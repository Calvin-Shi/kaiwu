#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import multiprocessing
from kaiwudrl.server.common.strategy import IPredictorStrategy
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
from common_python.ipc.zmq_util import ZmqServer, ZmqConfig


class OnPolicyStrategy(IPredictorStrategy):
    """
    OnPolicy的策略实现
    """

    def __init__(self, predictor):
        self.master_conn = []
        self.slave_conn = None
        self.current_sync_model_version_from_learner = 0

        # 传递了predictor对象, 则该类里可以复用, 而不是每次调用都传predictor对象
        self.predictor = predictor

        # 下面是统计告警指标
        self.on_policy_pull_from_modelpool_error_cnt = 0
        self.on_policy_pull_from_modelpool_success_cnt = 0
        self.actor_change_model_version_error_count = 0
        self.actor_change_model_version_success_count = 0

    def before_run(self, context):
        """
        on_policy的before_run逻辑
        """

        """
        model_file_sync_wrapper, actor和learner之间的Model文件同步, 采用单独的进程
        如果是on-policy算法则需要保存下来learner同步过来最新的model文件ID, 如果是off-policy则不需要
        为了编程方便, 都统一设置下
        """
        self.predictor.model_file_sync_wrapper.ckpt_sync_warper.make_model_dirs(self.predictor.logger)

        # 需要文件锁
        if context:
            self.model_file_sync_wrapper_lock = context.model_file_sync_wrapper_lock
        else:
            self.model_file_sync_wrapper_lock = multiprocessing.Lock()

        self.zmq_config = ZmqConfig(
            zmq_io_threads_server=CONFIG.zmq_io_threads_server,
            zmq_io_threads_client=CONFIG.zmq_io_threads_client,
            tcp_keep_alive=CONFIG.tcp_keep_alive,
            tcp_keep_alive_idle=CONFIG.tcp_keep_alive_idle,
            tcp_keep_alive_intvl=CONFIG.tcp_keep_alive_intvl,
            tcp_keep_alive_cnt=CONFIG.tcp_keep_alive_cnt,
            sock_buff_size=CONFIG.sock_buff_size,
            backlog_size=CONFIG.backlog_size,
            tcp_immediate=CONFIG.tcp_immediate,
            zmq_ops_sendhwm=CONFIG.zmq_ops_sendhwm,
            zmq_ops_recvhwm=CONFIG.zmq_ops_recvhwm,
        )

        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            # 只有第一个主进程才需要启动zmq_server，从进程不需要
            if not self.predictor.index:
                zmq_server_port = int(CONFIG.zmq_server_port) + 100
                self.zmq_server = ZmqServer(CONFIG.ip_address, zmq_server_port, self.zmq_config)
                self.zmq_server.bind()
                self.predictor.logger.info(
                    f"predict zmq server on-policy bind at {CONFIG.ip_address}:{zmq_server_port} for learner"
                )

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            sorted_items = sorted(self.predictor.policy_agent_wrapper_maps.items(), key=lambda item: item[0])
            sorted_keys = [key for key, value in sorted_items]
            key_index = sorted_keys.index(self.predictor.policy_name)
            zmq_server_port = int(CONFIG.zmq_server_port) + (self.predictor.index + 1) * 100 + key_index

            self.zmq_server = ZmqServer(CONFIG.aisrv_ip_address, zmq_server_port, self.zmq_config)
            self.zmq_server.bind()
            self.predictor.logger.info(
                f"predict zmq server on-policy bind at {CONFIG.aisrv_ip_address}:{zmq_server_port} for learner"
            )

        else:
            pass

        self.predictor.predict_common_object.set_current_sync_model_version_from_learner(
            self.current_sync_model_version_from_learner
        )

        self.predictor.logger.info("predict OnPolicyStrategy before_run success")

    def predict_stat(self):
        """
        on_policy里的特有统计指标
        """
        monitor_data = {
            KaiwuDRLDefine.ON_POLICY_PULL_FROM_MODELPOOL_ERROR_CNT: self.on_policy_pull_from_modelpool_error_cnt,
            KaiwuDRLDefine.ON_POLICY_PULL_FROM_MODELPOOL_SUCCESS_CNT: self.on_policy_pull_from_modelpool_success_cnt,
            KaiwuDRLDefine.ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_ERROR_COUNT: self.actor_change_model_version_error_count,
            KaiwuDRLDefine.ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_SUCCESS_COUNT: self.actor_change_model_version_success_count,
            KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_COST_MS: self.predictor.load_model_common_object.get_actor_load_last_model_error_cnt(),
            KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_SUCC_CNT: self.predictor.load_model_common_object.get_actor_load_last_model_succ_cnt(),
            KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_ERROR_CNT: self.predictor.load_model_common_object.get_actor_load_last_model_error_cnt(),
        }

        if CONFIG.use_prometheus:
            self.predictor.monitor_proxy.put_data({self.predictor.current_pid: monitor_data})

    def process_policy_specific(self):
        """
        on_policy特定的逻辑
        """

        """
        actor上执行on-policy流程
        1. actor的predict的0号进程作为预测主进程, 处理on-policy的事务
        2. actor的predict的其他进程作为预测从进程, 只是处理模型加载事务
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            if not self.predictor.index:
                self.actor_predict_on_policy_process_detail()
            else:
                self.actor_predict_on_policy_process_slave()
        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            self.actor_predict_on_policy_process_detail()
        else:
            pass

    def actor_predict_on_policy_process_slave(self):
        """
        actor的非预测主进程, 进行下面操作:
        1. 轮询是否需要加载model文件
        1.1 如果需要加载即从预测主进程发送的管道里有model_change_version值即代表需要加载则加载
        1.1.1 加载成功, 返回预测主进程成功信息
        1.1.2 加载失败, 返回预测主进程失败信息
        1.2 如果不需要加载, 本次不做操作
        """
        if self.slave_conn and self.slave_conn.poll(0):
            model_change_version = self.slave_conn.recv()
            if model_change_version:
                self.predictor.load_model_common_object.standard_load_last_new_model_by_framework(CONFIG.policy_name)
                self.current_sync_model_version_from_learner = model_change_version

                # 更新下predict_common里的model_version
                self.predictor.predict_common_object.set_current_sync_model_version_from_learner(
                    self.current_sync_model_version_from_learner
                )

                # 返回该预测从进程加载model文件成功的消息
                self.slave_conn.send(True)

    # actor重新从modelpool获取文件, 因为是learner才push到modelpool, 这里加上重试机制
    def actor_get_model_from_modelpool(self):
        all_pull_model_success = False
        current_available_model_files = []
        retry_count = 0

        while not all_pull_model_success and retry_count < int(CONFIG.on_policy_error_retry_count_when_modelpool):
            (
                pull_model_success,
                current_available_model_files,
            ) = self.predictor.model_file_sync_wrapper.ckpt_sync_warper.pull_checkpoint_from_model_pool_by_on_policy(
                self.predictor.logger, self.model_file_sync_wrapper_lock
            )
            if not pull_model_success:
                # 如果本次失败, 则sleep下再重试, 这里重试的间隔设置大些
                time.sleep(CONFIG.idle_sleep_second * 1000)
            else:
                all_pull_model_success = True
                self.predictor.logger.info(f"predict actor pull_checkpoint_from_model_pool success")
                break

            retry_count += 1

        return all_pull_model_success, current_available_model_files

    # 从的predictor进程需要设置model_version
    def set_model_version(self, model_version):
        self.current_sync_model_version_from_learner = model_version

    def append_predictor_master_conn(self, master_conn):
        self.master_conn.append(master_conn)

    def set_predictor_slave_conn(self, slave_conn):
        self.slave_conn = slave_conn

    def actor_predict_on_policy_process_master(self, model_version):
        """
        actor上单个predict进程的处理逻辑, 注意是有多个predict进程的场景, 见actor配置actor_predict_process_num
        1. 第0个进程作为主进程, 主进程的工作:
        1.1 拉取最新的model文件,如果成功, 继续剩余流程; 否则失败退出
        1.2 加载最新model文件
        1.3 等待其他从进程加载最新model文件的响应
        1.4 回复learner的on-policy流程成功
        2. 剩余的进程作为从进程
        2.1 加载最新model文件
        2.2 给预测主进程返回响应
        """

        """
        actor重新从modelpool获取文件, 因为是learner才push到modelpool, 这里加上重试机制
        """
        actor_get_model_file_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            success, current_available_model_files = self.actor_get_model_from_modelpool()
            if success:
                actor_get_model_file_success = True
                break

        """
        根据actor从modelpool拉取model文件执行下面流程:
        1. 成功, actor加载最新model文件, 更新当前self.current_sync_model_version_from_learner值, 回复learner响应
        2. 失败, actor告警指标++, 此时返回到learner处, learner本次失败不直接退出, 等待下一轮的on_policy流程
        """
        actor_execute_on_policy_success = False
        # 存在返回aisrv_get_model_file_success为True, 但是current_available_model_files为空的情况
        if not actor_get_model_file_success or not current_available_model_files:
            self.predictor.logger.warning(
                f"predict actor pull_checkpoint_from_model_pool failed, skip this round, "
                f"not change model_version: {model_version}, but return false to continue on-policy flow"
            )
            self.on_policy_pull_from_modelpool_error_cnt += 1
        else:
            self.on_policy_pull_from_modelpool_success_cnt += 1

            # actor的主predictor进程加载最新model文件
            self.predictor.load_model_common_object.standard_load_last_new_model_by_framework(
                policy_name=CONFIG.policy_name, models_path=current_available_model_files[-1]
            )

            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
                # 等待其他predictor进程的更新model文件和model_version, 采用异步方式
                all_predictor_on_policy_success = True
                for conn in self.master_conn:
                    conn.send(model_version)

                # 等待所有的从进程确认加载最新model文件完成
                for conn in self.master_conn:
                    retries = 0
                    while retries < CONFIG.on_policy_error_max_retry_rounds:
                        change_model_version = conn.recv()
                        if change_model_version:
                            break
                        else:
                            time.sleep(CONFIG.idle_sleep_second * 1000)
                            retries += 1

                    # 达到超时条件跳出整个循环, 说明此时某个预测从进程是异常的情况
                    if retries == CONFIG.on_policy_error_max_retry_rounds:
                        all_predictor_on_policy_success = False
                        break

                if not all_predictor_on_policy_success:
                    actor_execute_on_policy_success = False
                else:
                    actor_execute_on_policy_success = True

            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                actor_execute_on_policy_success = True

        return actor_execute_on_policy_success

    def actor_predict_on_policy_process_detail(self):
        """
        actor上的单个predict进程的on-policy的处理流程:
        1. 同步model_version请求
        1.1 获取来自learner的 model文件同步请求
        1.2 actor重新从modelpool获取文件
        1.2.1 如果成功则继续剩余流程
        1.2.2 失败则返回learner的明确失败的结果, learner根据情况决定是否让aisrv执行更新model_version操作, actor等待下一次model_version改变再走该流程
        1.2.2.1 如果actor返回给learner执行model_version失败, 则learner不能让aisrv执行修改model_version操作
        1.2.2.2 如果actor返回给learner执行model_version成功, 则learner让aisrv执行修改model_version操作
        1.3 actor加载最新model文件
        1.4 朝learner发送model文件同步响应
        2. 心跳请求
        2.1 心跳响应
        """

        try:
            # 获取来自learner的 model文件同步请求
            client_id, message = self.zmq_server.recv(block=False, binary=False)
            if message:
                if (
                    message[KaiwuDRLDefine.MESSAGE_TYPE]
                    == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST
                ):

                    """
                    predictor主进程走on-policy的流程
                    """
                    model_version = message[KaiwuDRLDefine.MESSAGE_VALUE]
                    actor_execute_on_policy_success = self.actor_predict_on_policy_process_master(model_version)
                    if not actor_execute_on_policy_success:
                        # 接入告警统计
                        self.predictor.logger.warning(
                            f"predict learner ask actor to set model_version: {model_version} failed"
                        )
                        self.actor_change_model_version_error_count += 1
                    else:
                        self.current_sync_model_version_from_learner = model_version
                        self.actor_change_model_version_success_count += 1

                        # 更新下predict_common里的model_version
                        self.predictor.predict_common_object.set_current_sync_model_version_from_learner(
                            self.current_sync_model_version_from_learner
                        )

                        self.predictor.logger.info(
                            f"predict learner ask actor to set model_version: {model_version} success"
                        )

                    # actor朝learner发送model文件同步响应
                    send_data = {
                        KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE,
                        KaiwuDRLDefine.MESSAGE_VALUE: actor_execute_on_policy_success,
                    }

                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.predictor.logger.info(
                        f"predict learner ask actor to {message[KaiwuDRLDefine.MESSAGE_TYPE]} success, result is {actor_execute_on_policy_success}"
                    )

                elif message[KaiwuDRLDefine.MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST:

                    # 心跳采用最简单方式即可
                    send_data = {
                        KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE,
                        KaiwuDRLDefine.MESSAGE_VALUE: True,
                    }

                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.predictor.logger.debug(
                        f"predict learner ask actor to {message[KaiwuDRLDefine.MESSAGE_TYPE]} success"
                    )

                else:
                    self.predictor.logger.error(
                        f"predict learner learner_model_sync_req not support "
                        f"message_type {message[KaiwuDRLDefine.MESSAGE_TYPE]}, so return"
                    )
                    return

        except Exception as e:
            pass

    def strategy_name(self):
        """
        策略特定的名字
        """
        return "on_policy"

    def cleanup(self):
        """
        OnPolicy清理操作
        """
        pass
