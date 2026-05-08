#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import multiprocessing
import os
import time
import numpy as np
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
import datetime
import copy
from kaiwudrl.common.utils.common_func import TimeIt
from kaiwudrl.common.config.app_conf import AppConf

from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.common_func import (
    get_uuid,
    compress_data,
    get_mean_and_max,
)
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.server.common.load_model_common import LoadModelCommon

from kaiwudrl.common.algorithms.agent_wrapper_common import (
    create_standard_agent_wrapper,
)
from kaiwudrl.server.common.predict_common import PredictCommon
from kaiwudrl.server.common.actor_to_aisrv_response_common import (
    ActorToAisrvResponseCommon,
)
from kaiwudrl.server.common.strategy import create_strategy
from common_python.worker.worker import Worker, WorkerConfig


class PredictorLocal(Worker):
    def __init__(self, policy_name, index, actor_addr, context) -> None:
        # 进程pid
        self.current_pid = os.getpid()
        worker_config = WorkerConfig(
            worker_name="predict",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        self.policy_name = policy_name
        """
        支持业务自定义和从alloc获取的情况
        1. 默认端口
        2. alloc服务下发的IP和端口
        3. 从配置文件读取的IP和端口
        """
        # 获取ip
        self.actor_address = actor_addr[0]
        # 获取port
        self.actor_port = actor_addr[1]

        """
        aisrv <--> actor之间, actor是支持多个aisrv的, 故actor需要知道各个aisrv的client_id, 才能准确回包, 故这里采用uuid方式
        该值会透传给actor, 在actor采用zmq进行回包时, 带上该client_id参数
        """
        self.client_id = get_uuid()

        # 标志该index是多少, 主要用于看主和从关系
        self.index = index

        # slots
        self.slots = context.slots
        self.slot_group_name = f"{policy_name}_predict_{self.index}"
        self.slots.register_group(self.slot_group_name)

        # send msg queue
        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 统计数目
        self.send_to_actor_succ_cnt = 0
        self.send_to_actor_error_cnt = 0

        self.recv_from_actor_succ_cnt = 0
        self.recv_from_actor_error_cnt = 0

        # 采用压缩算法时, 压缩耗时, 解压缩耗时, 压缩大小
        self.max_compress_time = 0
        self.max_decompress_time = 0
        self.max_compress_size = 0
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_from_zmq_queue_size = 0

        self.current_sync_model_version_from_learner = 0

        # 本地预测分段耗时
        self.get_and_predict_cost_time_ms = 0
        self.send_cost_time_ms = 0

        # 设置最后处理时间
        self.last_predict_stat_time = 0
        self.last_load_last_new_model_time = 0

        # policy和agent_wrapper对象的map, 为了支持多agent
        self.policy_agent_wrapper_maps = {}

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = None

        # 设置公共的加载model文件类, 便于每次加载时使用
        self.load_model_common_object = None

        # 通过组合引入策略
        self.strategy = create_strategy(self)
        self.context = context

        # 由外界传入model_file_sync_wrapper对象
        self.model_file_sync_wrapper = None
        if CONFIG.remote_agent_default_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT:
            if CONFIG.need_to_start_learner:
                self.model_file_sync_wrapper = context.model_file_sync_wrapper

    # 需要区分是哪个agent发送的请求
    def put_predict_data(self, slot_id, agent_id, message_id, model_version, agent_main_id, predict_data) -> None:
        if not predict_data or self.msg_queue.full():
            return False

        """
        因为on-policy情况下需要修改model_version, 如果这里的compose_id是tuple则转换为list修改耗费性能, 故设计为list
        """
        self.msg_queue.put(
            (
                [slot_id, agent_id, message_id, model_version, agent_main_id],
                predict_data,
            )
        )
        return True

    def get_predict_data(self, slot_id):

        input_pipe = self.slots.get_input_pipe(self.slot_group_name, slot_id)

        # 设置queue的超时时间
        if input_pipe.poll(CONFIG.queue_wait_timeout):
            return input_pipe.recv()

        return None

    def standard_serialize_buffer_data(self):
        """
        针对不同的数据类型, 进行序列化
        消息的数据格式:
        (actor_id, slot_id, agent_id) | data
        1. pickle, 可以直接组装msg
        2. protobuf, 需要对KaiwuServerRequest类赋值
        """
        # 序列化 pre_req(业务State定义的类里的key的顺序) + client_id + compose_id(agent_id, slot_id)

        assert self.cur_buf_size == 1
        # self.logger.info(f'predict cur_buf_size is:{self.cur_buf_size}', g_not_server_label)

        # 采用pickle序列化
        msg = {
            "data": self.buffer_data,
            "client_id": self.client_id_buf[: self.cur_buf_size],
            "compose_id": self.compose_id_buf[: self.cur_buf_size],
        }

        # 在单个进程类不需要进行压缩和解压缩
        return msg

    def serialize_buffer_data(self):
        """
        针对不同的数据类型, 进行序列化
        消息的数据格式:
        (actor_id, slot_id, agent_id) | data
        1. pickle, 可以直接组装msg
        """

        # 序列化 pre_req(业务State定义的类里的key的顺序) + client_id + compose_id(agent_id, slot_id)

        assert self.cur_buf_size == 1
        # self.logger.info(f'predict cur_buf_size is:{self.cur_buf_size}', g_not_server_label)

        """
        如果是标准化直接按照整个buff传输
        """
        pred_req = self.buffer_data

        msg = {
            "data": pred_req,
            "client_id": self.client_id_buf[: self.cur_buf_size],
            "compose_id": self.compose_id_buf[: self.cur_buf_size],
        }

        # 在单个进程类不需要进行压缩和解压缩
        return msg

    def compress_request_data(self, msg):

        # 增加lz4的压缩
        with TimeIt() as ti:
            compress_msg = compress_data(msg)

        # 压缩耗时和压缩包大小
        if self.max_compress_time < ti.interval:
            self.max_compress_time = ti.interval

        compress_msg_len = len(compress_msg)
        if self.max_compress_size < compress_msg_len:
            self.max_compress_size = compress_msg_len

        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            return dump_arrays(compress_msg)

        return compress_msg

    def predict_stat_reset(self):
        self.predict_common_object.set_actor_batch_predict_cost_time_ms(0)
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_from_zmq_queue_size = 0
        self.predict_common_object.set_actor_load_last_model_cost_ms(0)

        self.max_decompress_time = 0
        self.max_compress_size = 0
        self.max_compress_time = 0
        self.send_to_actor_succ_cnt = 0
        self.send_to_actor_error_cnt = 0
        self.actor_to_aisrv_response_common_object.set_recv_from_actor_succ_cnt(0)
        self.recv_from_actor_error_cnt = 0
        self.get_and_predict_cost_time_ms = 0
        self.send_cost_time_ms = 0

    def predict_stat(self):
        """
        这里增加predict的统计项
        """

        predict_count = self.predict_common_object.predict_stat()

        if int(CONFIG.use_prometheus) and not self.index and CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:

            # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
            try:
                msg_queue_size = self.msg_queue.qsize()
            except NotImplementedError:
                msg_queue_size = 0
            except Exception as e:
                msg_queue_size = 0

            actor_batch_predict_cost_time_ms = self.predict_common_object.get_actor_batch_predict_cost_time_ms()
            recv_from_actor_succ_cnt = self.actor_to_aisrv_response_common_object.get_recv_from_actor_succ_cnt()

            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT: predict_count,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE: self.actor_from_zmq_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_COST_TIME_MS: self.actor_from_zmq_queue_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_BATCH_PREDICT_COST_TIME_MS: actor_batch_predict_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME: self.max_decompress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME: self.max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE: self.max_compress_size,
                KaiwuDRLDefine.MONITOR_AISRV_ACTOR_PROXY_QUEUE_LEN: msg_queue_size,
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT: self.send_to_actor_succ_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT: self.send_to_actor_error_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT: recv_from_actor_succ_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT: self.recv_from_actor_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_GET_AND_PREDICT_COST_MS: self.get_and_predict_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_BATCH_COST_TIME_MS: self.send_cost_time_ms,
            }

            # 针对aisrv发出去的请求, 有响应包的场景, 只是计算最大值和平均值时延
            if int(CONFIG.use_prometheus):
                mean_value, max_value = get_mean_and_max(self.time_cost_map.values())

                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ACTOR_MEAN_TIME_COST] = mean_value
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ACTOR_MAX_TIME_COST] = max_value

            self.time_cost_map.clear()

            # 针对aisrv发出去的请求, 没有响应包的场景
            timeout_cnt = 0
            for key in list(self.timeout_map.keys()):
                value = self.timeout_map.get(key)

                # 计算下来是s为单位
                time_dela = (int(round(time.time() * 1000)) - value) / 1000

                if time_dela > CONFIG.aisrv_actor_timeout_second_threshold:
                    timeout_cnt += 1
                    self.logger.error(
                        f"predict message id {key} timeout after {time_dela} seconds",
                        g_not_server_label,
                    )

                    del self.timeout_map[key]
                    continue

            if int(CONFIG.use_prometheus):
                monitor_data[
                    f"{KaiwuDRLDefine.MONITOR_AISRV_ACTOR_TIMEOUT_GT}{CONFIG.aisrv_actor_timeout_second_threshold}"
                ] = timeout_cnt

                self.monitor_proxy.put_data({self.current_pid: monitor_data})

            # 策略特定的统计
            self.strategy.predict_stat()

        # 指标复原, 计算的是周期性的上报指标
        self.predict_stat_reset()

        self.logger.info(f"predict now predict count is {predict_count}")

    def after_run(self) -> bool:
        pass

    def before_run(self) -> None:
        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        # 填充client_id
        self.client_id_buf = np.empty((CONFIG.proxy_batch_size * 2), np.int32)
        self.client_id_buf.fill(self.client_id)

        # buff 的处理, 填充了COMPOSE_ID(agent_id, slot_id, message_id, model_version, agent_main_id), 由于需要存储不同的字段类型故采用list
        self.compose_id_buf = [None] * CONFIG.proxy_batch_size * 2

        """
        如果是标准化直接按照整个buff传输
        """
        self.buffer_data = {}
        self.cur_buf_size = 0

        """
        下面的操作是做下原本在actor上类似的操作, 移植过来
        """
        # 加载配置文件kaiwudrl/conf/algo_conf.json
        AlgoConf.load_conf(CONFIG.algo_conf)

        # 加载配置文件kaiwudrl/conf/app_conf.json
        AppConf.load_conf(CONFIG.app_conf)

        # policy_name 主要是和kaiwudrl/conf/app_conf.json设置一致
        self.policy_conf = AppConf.get_app_conf(CONFIG.app, "policies")

        # agent_wrapper, 无论是remote, local, none这里都需要执行下面操作
        create_standard_agent_wrapper(
            self.policy_conf,
            self.policy_agent_wrapper_maps,
            None,
            self.logger,
            self.monitor_proxy,
        )

        # 注册定时器任务
        # set_schedule_event(
        #    CONFIG.prometheus_stat_per_minutes, self.predict_stat)

        # 设置公共的加载文件类, 便于每次加载文件时调用
        self.load_model_common_object = LoadModelCommon(self.logger)
        self.load_model_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.load_model_common_object.set_policy_agent_wrapper_maps(self.policy_agent_wrapper_maps)

        # 如果是在eval模式下下则执行第一次加载
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            # 单机单进程的版本是在aisrv上预测这里不需要
            if CONFIG.wrapper_type != KaiwuDRLDefine.WRAPPER_LOCAL:
                if not self.load_model_common_object.standard_load_last_new_model_by_framework(self.policy_name):
                    return False

        # 预先加载模型文件模式, 只有在train训练模式下预加载才有效
        if int(CONFIG.preload_model):
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                self.load_model_common_object.preload_model_file(self.policy_agent_wrapper_maps)
            else:
                self.logger.warning(f"predict only run_mode is {KaiwuDRLDefine.RUN_MODE_TRAIN} support preload model")

        """
        用于做超时控制的, key为aisrv --> actor的message_id, value为发送时间
        1. 发送时, 将message_id和发送时间放在map里
        2. 当响应包回来, 则当前时间 - 发送时间, 即耗时
        3. 如果在一定时间里没有响应包回来, 则开始删除map里的key, 并且记录ERROR日志
        """
        self.timeout_map = {}
        self.time_cost_map = {}

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = PredictCommon(self.policy_name, self.monitor_proxy, self.logger)
        self.predict_common_object.set_policy_agent_wrapper_maps(self.policy_agent_wrapper_maps)
        self.predict_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.predict_common_object.set_policy_conf(self.policy_conf)

        # 检查 MultiModelManager 初始化是否成功
        if not self.predict_common_object.is_multi_model_manager_init_success():
            self.logger.error(
                f"predict MultiModelManager initialization failed in before_run, process will exit",
                g_not_server_label,
            )
            return False

        # 设置公共的aisrv/actor朝aisrv回包的处理类, 便于每次处理回包时调用
        self.actor_to_aisrv_response_common_object = ActorToAisrvResponseCommon(self.logger)
        self.actor_to_aisrv_response_common_object.set_slots(self.slots)
        self.actor_to_aisrv_response_common_object.set_slot_group_name(self.slot_group_name)
        self.actor_to_aisrv_response_common_object.set_zmq_server_ip(self.get_zmq_server_ip())

        # 策略特定的初始化
        self.strategy.before_run(self.context)

        # 在before run最后打印启动成功日志
        self.logger.info(
            (
                f"predict policy_name: {self.policy_name}, start success at pid {self.current_pid}, "
                f"on-policy/off-policy is {self.strategy.strategy_name()}, "
                f"actor_receive_cost_time_ms: {CONFIG.actor_receive_cost_time_ms}, "
                f"predict_batch_size: {CONFIG.predict_batch_size}"
            ),
            g_not_server_label,
        )

        return True

    def get_predict_request_data_by_direct(self):
        """
        aisrv需要预测的数据, 采用队列形式返回
        """
        with TimeIt() as ti:
            msgs = []

            # 按照时间间隔和批处理大小收包
            start_time = time.time()
            while len(msgs) < int(CONFIG.predict_batch_size):
                msg = self.standard_get_data_from_predict_data_queue()
                if msg:
                    msgs.append(copy.deepcopy(msg))
                    self.send_to_actor_succ_cnt += 1

                # 收包超时时强制退出, 平滑处理
                if (time.time() - start_time) * 1000 > int(CONFIG.actor_receive_cost_time_ms):
                    break

        msgs_length = len(msgs)
        if not msgs_length:
            return msgs

        # 获取采集周期里的最大值
        if self.actor_from_zmq_queue_size < msgs_length:
            self.actor_from_zmq_queue_size = msgs_length

        if self.actor_from_zmq_queue_cost_time_ms < ti.interval * 1000:
            self.actor_from_zmq_queue_cost_time_ms = ti.interval * 1000

        return msgs

    def standard_get_data_from_predict_data_queue(self):
        """
        这里的分2种情况:
        1. 如果是on-polciy的, 注意需要采用非阻塞的, 否则阻塞了predict主流程
        2. 如果是非on-polciy的, 注意采用阻塞的, 性能要好于非阻塞的, 但是对于battlesrv数量远远小于了batch_size则需要设置超时时间
        """
        tmp_data = None
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            try:
                tmp_data = self.msg_queue.get_nowait()
            except Exception as e:
                pass
        else:
            tmp_data = self.msg_queue.get()

        if not tmp_data:
            return None

        if isinstance(tmp_data, tuple):
            if CONFIG.aisrv_actor_protocol in (KaiwuDRLDefine.PROTOCOL_PICKLE, KaiwuDRLDefine.PROTOCOL_MSGPACK):
                compose_id, data = tmp_data

                self.compose_id_buf[self.cur_buf_size] = compose_id

                # 所有的预测数据作为一个整体
                self.buffer_data = data

                self.cur_buf_size += 1

                is_predict_request = data.get(KaiwuDRLDefine.MESSAGE_TYPE) == KaiwuDRLDefine.MESSAGE_PREDICT

                """
                注意:
                1. 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
                2. slot_id, agent_id, message_id作为存放时延的key, 不能加入model_version, 该值可能被actor返回的值修改掉
                3. 如果是管理流请求不能放入, 因为没有回包
                """
                # 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
                [
                    slot_id,
                    agent_id,
                    message_id,
                    model_version,
                    agent_main_id,
                ] = compose_id

                if message_id != 1 and is_predict_request:
                    self.timeout_map[(slot_id, agent_id, message_id)] = int(round(time.time() * 1000))

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        (
                            f"predict distributed_tracing compose_id {compose_id} "
                            f"will send to actor {self.get_zmq_server_ip()}"
                        ),
                        g_not_server_label,
                    )

                msg = self.standard_serialize_buffer_data()

                self.cur_buf_size = 0

            return msg

    def get_zmq_server_ip(self):
        """
        返回zmq server的IP和端口
        """

        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
            return f"{self.actor_address}:{CONFIG.zmq_server_port}"
        elif CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            return f"{self.actor_address}:{CONFIG.zmq_server_op_port}"
        else:
            return None

    # 周期性的操作
    def periodic_operations(self):
        # 记录发送给actor成功失败数目, 包括发出去和收回来的请求
        # schedule.run_pending()

        now = time.time()
        if now - self.last_predict_stat_time > CONFIG.prometheus_stat_per_minutes * 60:
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                self.predict_stat()
                self.last_predict_stat_time = now

    def handle_message_timeouts(self, compose_id_indexs):
        """
        处理响应回包里超时情况, 处理步骤如下:
        1. 按照message_id计算出耗时, 放入time_cost_map
        2. 删除timeout_map对应的message_id项

        主要去掉第一帧耗时大的
        """
        for index in compose_id_indexs:
            if index in self.timeout_map:
                now = int(round(time.time() * 1000))
                cost_time = now - self.timeout_map.get(index)
                self.time_cost_map[index] = cost_time

                del self.timeout_map[index]

    def run_once(self) -> None:
        """
        单次run_once, 采用串行操作
        """

        # 周期性的操作, 放在最前面, 规避因为没有请求而阻塞
        self.periodic_operations()

        # 步骤2, 策略特定处理
        self.strategy.process_policy_specific()

        # 读取预测请求并且预测
        with TimeIt() as ti:
            msgs = self.get_predict_request_data_by_direct()
            if msgs:

                # 执行预测
                size, pred = self.predict_common_object.predict(msgs)

            self.process_run_idle_count += 1

        # 获取采集周期里的最大值
        if self.get_and_predict_cost_time_ms < ti.interval * 1000:
            self.get_and_predict_cost_time_ms = ti.interval * 1000

        with TimeIt() as it:
            if msgs:
                compose_id_indexs = []

                # 发送预测响应
                compose_id_indexs = (
                    self.actor_to_aisrv_response_common_object.standard_send_response_to_aisrv_simple_fast_by_aisrv(
                        size, pred
                    )
                )

                # 处理响应的回包里超时控制
                if compose_id_indexs:
                    self.handle_message_timeouts(compose_id_indexs)

        if self.send_cost_time_ms < it.interval * 1000:
            self.send_cost_time_ms = it.interval * 1000

    def run(self):
        if not self.before_run():
            self.logger.error(f"predict before_run failed, so return", g_not_server_label)
            return

        # 无论多个policy还是单个policy, 第1个policy是获取得到的
        agent_wrapper = next(iter(self.policy_agent_wrapper_maps.values()))
        while not agent_wrapper.should_stop():
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题, 减少CPU损耗
                if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_idle_count = 0

            except ValueError as e:
                self.logger.exception(
                    f"predict run error: {str(e)}",
                    g_not_server_label,
                )
            except Exception as e:
                self.logger.exception(
                    f"predict run error: {str(e)}",
                    g_not_server_label,
                )

        for policy, agent_wrapper in self.policy_agent_wrapper_maps.items():
            agent_wrapper.close()
            self.logger.info(f"predict {policy} agent_wrapper.close success")
