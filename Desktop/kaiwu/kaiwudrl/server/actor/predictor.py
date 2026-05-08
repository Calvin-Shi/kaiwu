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
from kaiwudrl.common.checkpoint.model_file_save import ModelFileSave
from common_python.config.config_control import CONFIG
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
import os
import schedule
import datetime
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
    decompress_data,
    decompress_data_parallel,
)

from common_python.alloc.alloc_proxy import AllocProxy
from kaiwudrl.common.algorithms.agent_wrapper_common import (
    create_standard_agent_wrapper,
)
from kaiwudrl.server.common.predict_common import PredictCommon
from kaiwudrl.server.common.actor_to_aisrv_response_common import (
    ActorToAisrvResponseCommon,
)
from kaiwudrl.server.common.load_model_common import LoadModelCommon
from kaiwudrl.server.common.strategy import create_strategy
from common_python.worker.worker import Worker, WorkerConfig


class Predictor(Worker):
    def __init__(self, send_server, recv_server):
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

        self.send_server = send_server
        self.recv_server = recv_server

        # policy_name 主要是和kaiwudrl/conf/app_conf.json设置一致
        self.policy_conf = AppConf.get_app_conf(CONFIG.app, "policies")

        # 通过组合引入策略
        self.strategy = create_strategy(self)

        # 进程启动的序号
        self.index = -1

        """
        actor采用批处理从zmq_server获取, 故记录了此时队列长度, 从队列里获取的耗时,
        为了减少损耗, 只是记录统计周期最后一次的值
        1. actor从zmq-server获取的队列长度, 最大为配置值, 需要查看平时是多少
        2. 从zmq-server的队列里获取数据时批处理耗时
        3. actor批处理预测耗时
        4. actor将预测结果发送给aisrv的批处理耗时
        5. actor加载最新的Model文件耗时
        """
        self.actor_from_zmq_queue_size = 0
        self.actor_from_zmq_queue_cost_time_ms = 0

        self.max_decompress_time = 0

        """
        从actor_server获取的需要预测的数据, 每次处理完成需要清空
        因为存在每次按照batch_size或者按照超时时间来读取, 那这里采用单独的线程来读取数据, 规避超时时间的限制

        """
        if CONFIG.pipeline_process_sync:
            self.predict_request_queue = multiprocessing.Queue(CONFIG.queue_size)
            self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)

        if CONFIG.actor_server_predict_server_different_queue:
            self.predict_request_queue_from_actor_server = None

        # policy和model对象的map, 为了支持多agent
        self.policy_agent_wrapper_maps = {}

        self.model_file_sync_wrapper = None

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = None

    # 下面append_predictor_master_conn和set_predictor_slave_conn函数主要是由on_policy的场景调用
    def append_predictor_master_conn(self, master_conn):
        self.strategy.append_predictor_master_conn(master_conn)

    def set_predictor_slave_conn(self, slave_conn):
        self.strategy.set_predictor_slave_conn(slave_conn)

    # 返回predict_request_queue
    def get_predict_request_queue(self):
        if CONFIG.pipeline_process_sync or CONFIG.actor_server_predict_server_different_queue:
            return self.predict_request_queue

        return None

    # 返回predict_result_queue
    def get_predict_result_queue(self):
        if CONFIG.pipeline_process_sync or CONFIG.actor_server_predict_server_different_queue:
            return self.predict_result_queue

        return None

    # actor周期性的加载七彩石修改配置, 主要包括进程独有的和公共的
    def rainbow_activate(self):
        self.rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
        self.rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

    def predict_stat_reset(self):
        self.predict_common_object.set_actor_batch_predict_cost_time_ms(0)
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_from_zmq_queue_size = 0
        self.predict_common_object.set_actor_load_last_model_cost_ms(0)

        self.max_decompress_time = 0
        self.actor_to_aisrv_response_common_object.set_max_compress_size(0)
        self.actor_to_aisrv_response_common_object.set_max_compress_time(0)

    # 这里增加predict的统计项
    def predict_stat(self):

        predict_count = self.predict_common_object.predict_stat()

        if CONFIG.use_prometheus and CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:

            predict_request_queue_size = 0
            predict_result_queue_size = 0
            try:
                predict_request_queue_size = self.predict_request_queue.qsize()
                predict_result_queue_size = self.predict_result_queue.qsize()
            except Exception as e:
                pass

            actor_batch_predict_cost_time_ms = self.predict_common_object.get_actor_batch_predict_cost_time_ms()
            max_compress_time = self.actor_to_aisrv_response_common_object.get_max_compress_time()
            max_compress_size = self.actor_to_aisrv_response_common_object.get_max_compress_size()
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT: predict_count,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE: self.actor_from_zmq_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_COST_TIME_MS: self.actor_from_zmq_queue_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_BATCH_PREDICT_COST_TIME_MS: actor_batch_predict_cost_time_ms,
                # KaiwuDRLDefine.ACTOR_TCP_AISRV: actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME: self.max_decompress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_REQUEST_QUEUE_SIZE: predict_request_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_RESULT_QUEUE_SIZE: predict_result_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME: max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE: max_compress_size,
            }

            if CONFIG.use_prometheus:
                self.monitor_proxy.put_data({self.current_pid: monitor_data})

            # 策略特定的统计
            self.strategy.predict_stat()

        # 指标复原, 计算的是周期性的上报指标
        self.predict_stat_reset()

        self.logger.info(f"predict now predict count is {predict_count}")

    def start_actor_process_by_type(self):
        """
        根据不同的启动方式进行处理:
        1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
        2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动
        """
        if CONFIG.start_actor_learner_process_type:
            # 按照需要引入ModelFileSave
            self.model_file_saver = ModelFileSave()
            self.model_file_saver.start_actor_process_by_type(self.logger)

    # 外界设置下model_file_sync_wrapper
    def set_model_file_sync_wrapper(self, model_file_sync_wrapper):
        self.model_file_sync_wrapper = model_file_sync_wrapper

    def before_run(self):
        # 支持间隔N分钟, 动态修改配置文件
        if CONFIG.use_rainbow:
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            self.rainbow_wrapper = RainbowWrapper(self.logger)

            # 第一次配置主动从七彩石拉取, 后再设置为周期性拉取
            self.rainbow_activate()
            set_schedule_event(CONFIG.rainbow_activate_per_minutes, self.rainbow_activate)

        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        # 根据不同启动方式来进行处理
        self.start_actor_process_by_type()

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            self.process_pid_list = []

        # 需要引入业务自定义的workflow, 即while True循环
        self.workflow = None

        # agent_wrapper, 无论是remote, local, none这里都需要执行下面操作
        create_standard_agent_wrapper(
            self.policy_conf,
            self.policy_agent_wrapper_maps,
            None,
            self.logger,
            self.monitor_proxy,
        )

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            if CONFIG.actor_server_async:
                self.process_pid_list.append(self.send_server.pid)
                self.process_pid_list.append(self.recv_server.pid)
            else:
                self.process_pid_list.append(self.send_server.pid)

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc) and self.index == 0:
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
                self.process_pid_list.append(self.alloc_proxy.pid)

        # 注册定时器任务
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.predict_stat)

        # 设置公共的加载文件类, 便于每次加载文件时调用
        self.load_model_common_object = LoadModelCommon(self.logger)
        self.load_model_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.load_model_common_object.set_policy_agent_wrapper_maps(self.policy_agent_wrapper_maps)

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            if not self.load_model_common_object.standard_load_last_new_model_by_framework(CONFIG.policy_name):
                return False

        # 预先加载模型文件模式, 只有在train训练模式下预加载才有效
        if int(CONFIG.preload_model):
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                self.load_model_common_object.preload_model_file(self.policy_agent_wrapper_maps)
            else:
                self.logger.warning(f"predict only run_mode is {KaiwuDRLDefine.RUN_MODE_TRAIN} support preload model")

        # 获取本机IP
        self.host = get_host_ip()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = PredictCommon(CONFIG.policy_name, self.monitor_proxy, self.logger)
        self.predict_common_object.set_policy_agent_wrapper_maps(self.policy_agent_wrapper_maps)
        self.predict_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.predict_common_object.set_policy_conf(self.policy_conf)

        # 检查 MultiModelManager 初始化是否成功
        if not self.predict_common_object.is_multi_model_manager_init_success():
            self.logger.error(
                f"predict MultiModelManager initialization failed in before_run, process will exit",
            )
            return False

        # 设置公共的aisrv/actor朝aisrv回包的处理类, 便于每次处理回包时调用
        self.actor_to_aisrv_response_common_object = ActorToAisrvResponseCommon(self.logger)
        self.actor_to_aisrv_response_common_object.set_zmq_send_server(self.send_server)
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.actor_to_aisrv_response_common_object.set_zmq_server(self.zmq_server)

        # 策略特定的初始化
        context = None
        self.strategy.before_run(context)

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"predict process start success at pid is {self.current_pid}, "
            f"on-policy/off-policy is {self.strategy.strategy_name()}, "
            f"actor_receive_cost_time_ms: {CONFIG.actor_receive_cost_time_ms}, "
            f"predict_batch_size: {CONFIG.predict_batch_size}"
        )

        return True

    def predict_tensorrt_direct(self):
        """
        流程:
        判断GPU队列里是否为空:
        1. 队列为空, 等待下次操作
        2. 队列非空, 开始处理预测请求, 并且返回actor_server预测响应
        """
        size = 0
        pred = None

        # 处理actor --> aisrv的回包
        self.send_server.put_predict_result_data([size, pred])

    # 从actor_server进程提供的队列收集预测数据, 以线程形式, 暂时不用
    def get_predict_data_from_actor_server_by_threading(self):
        while True:
            self.get_predict_data_from_actor_server()

    # actor_server获取的预测请求数据放入到predictor里
    def put_to_predict_queue(self, predict_data):
        if not predict_data:
            return

        if self.predict_request_queue.full():
            return

        self.predict_request_queue.put(predict_data)

    # predictor的预测结果数据放入到本地后, actor_server从本地拿走
    def get_predict_result_data(self):
        return self.predict_result_queue.get()

    def get_predict_data_from_actor_server(self):
        """
        从actor_server进程提供的队列收集预测数据, 以函数形式
        1. 如果是pipeline_process_sync为False则从actor_server队列里获取
        2. 如果是pipeline_process_sync为True则从本地队列里获取
        控制条件依据pipeline_process_sync的值:
        1. 如果是False:
            1.1 单次批处理predict_batch_size
            1.2 设置的超时时间
        2. 如果是True:
            2.1 尽最大努力获取数据
            2.2 超过predict_batch_size跳出, 平滑操作
        """

        datas = []

        with TimeIt() as it:
            if not CONFIG.pipeline_process_sync:

                # 按照时间间隔和批处理大小收包
                start_time = time.time()
                while len(datas) < int(CONFIG.predict_batch_size):

                    # 区分从哪里获取数据
                    data = None
                    if not CONFIG.actor_server_predict_server_different_queue:
                        data = self.recv_server.get_from_to_predict_queue()
                    else:
                        try:
                            data = self.predict_request_queue_from_actor_server.get()
                        except Exception as e:
                            pass

                    if data:
                        # 增加压缩和解压缩耗时
                        with TimeIt() as ti:
                            decompressed_data = decompress_data(data)

                        if self.max_decompress_time < ti.interval:
                            self.max_decompress_time = ti.interval

                        datas.append(decompressed_data)

                    # 收包超时时强制退出, 平滑处理
                    if (time.time() - start_time) * 1000 > int(CONFIG.actor_receive_cost_time_ms):
                        break

            else:

                # 最大限度收包
                while not self.predict_request_queue.empty():
                    datas.append(self.predict_request_queue.get())

                    # 最大predict_batch_size的跳出去, 平滑处理
                    if len(datas) > int(CONFIG.predict_batch_size):
                        break

        # 如果本次没有数据, 提前返回, 不需要进行处理
        datas_length = len(datas)
        if not datas_length:
            self.process_run_idle_count += 1
            return datas

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing get_predict_data_from_actor_server end")

        # 获取采集周期里的最大值
        if self.actor_from_zmq_queue_size < datas_length:
            self.actor_from_zmq_queue_size = datas_length

        if self.actor_from_zmq_queue_cost_time_ms < it.interval * 1000:
            self.actor_from_zmq_queue_cost_time_ms = it.interval * 1000

        return datas

    def get_predict_data_from_actor_server_parallel(self):
        """
        从actor_server进程提供的队列收集预测数据, 以函数形式, 并行处理
        1. 如果是pipeline_process_sync为False则从actor_server队列里获取
        2. 如果是pipeline_process_sync为True则从本地队列里获取
        控制条件依据pipeline_process_sync的值:
        1. 如果是False:
            1.1 单次批处理predict_batch_size
            1.2 设置的超时时间
        2. 如果是True:
            2.1 尽最大努力获取数据
            2.2 超过predict_batch_size跳出, 平滑操作
        """

        datas = []

        with TimeIt() as it:
            if not CONFIG.pipeline_process_sync:

                # 按照时间间隔和批处理大小收包
                start_time = time.time()
                data_from_queues = []
                while len(data_from_queues) < int(CONFIG.predict_batch_size):

                    # 区分从哪里获取数据
                    data = None
                    if not CONFIG.actor_server_predict_server_different_queue:
                        data = self.recv_server.get_from_to_predict_queue()
                    else:
                        try:
                            data = self.predict_request_queue_from_actor_server.get()
                        except Exception as e:
                            pass

                    if data:
                        data_from_queues.append(data)

                    # 收包超时时强制退出, 平滑处理
                    if (time.time() - start_time) * 1000 > int(CONFIG.actor_receive_cost_time_ms):
                        break

                # 批量处理数据
                if data_from_queues:
                    with TimeIt() as ti:
                        decompressed_data = decompress_data_parallel(data_from_queues)

                    # 增加压缩和解压缩耗时
                    if self.max_decompress_time < ti.interval:
                        self.max_decompress_time = ti.interval

                    datas = decompressed_data

            else:

                # 最大限度收包
                while not self.predict_request_queue.empty():
                    datas.append(self.predict_request_queue.get())

                    # 最大predict_batch_size的跳出去, 平滑处理
                    if len(datas) > int(CONFIG.predict_batch_size):
                        break

        # 如果本次没有数据, 提前返回, 不需要进行处理
        datas_length = len(datas)
        if not datas_length:
            self.process_run_idle_count += 1
            return datas

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing get_predict_data_from_actor_server end")

        # 获取采集周期里的最大值
        if self.actor_from_zmq_queue_size < datas_length:
            self.actor_from_zmq_queue_size = datas_length

        if self.actor_from_zmq_queue_cost_time_ms < it.interval * 1000:
            self.actor_from_zmq_queue_cost_time_ms = it.interval * 1000

        return datas

    # actor采用tensorrt前提下流水线处理
    def run_once_tesnorrt(self):
        # 步骤1, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, 进行预测, 并且获取预测响应
        self.predict_tensorrt_direct()

    def run_once(self):

        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, 策略特定处理
        self.strategy.process_policy_specific()

        # 步骤3, 从zmq/zmq-ops上获取data/tensor进行预测, 这里按照批处理获取数据, 尽最大努力去拿取队列里的数据, 如果没有则跳出该循环
        datas = self.get_predict_data_from_actor_server()
        if datas:

            # 步骤4, 预测
            size, pred = self.predict_common_object.predict(datas)

            # 步骤5, actor朝aisrv回包
            if CONFIG.distributed_tracing:
                self.logger.info(f"predict distributed_tracing predict put actor_server predict result start")

            """
            处理actor->aisrv的响应回包
            """
            if not CONFIG.pipeline_process_sync:
                if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
                    self.actor_to_aisrv_response_common_object.send_response_to_aisrv_by_actor(size, pred)
                else:
                    self.actor_to_aisrv_response_common_object.standard_send_response_to_aisrv_simple_fast_by_actor(
                        size, pred
                    )
            else:
                self.predict_result_queue.put([size, pred])

            if CONFIG.distributed_tracing:
                self.logger.info(f"predict distributed_tracing predict put actor_server predict result end")

        # Model文件同步操作, learner --> actor, 采用单独的进程处理

    def set_index(self, index):
        self.index = index

    def set_predict_request_queue_from_actor_server(self, predict_request_queue_from_actor_server):
        if not predict_request_queue_from_actor_server:
            return

        self.predict_request_queue_from_actor_server = predict_request_queue_from_actor_server

    def after_run(self) -> bool:
        pass

    def run(self):
        if not self.before_run():
            self.logger.error(f"predict before_run failed, so break")
            return

        # 无论多个policy还是单个policy, 第1个policy是获取得到的
        agent_wrapper = next(iter(self.policy_agent_wrapper_maps.values()))
        while not agent_wrapper.should_stop():
            try:
                self.run_once()

                # 因为在pipeline_process_sync模式下一直从本地收包容易导致CPU100%, 而在非pipeline_process_sync模式下有收包超时时间反而不容易发生
                if CONFIG.pipeline_process_sync:
                    # 短暂sleep, 规避容器里进程CPU使用率100%问题, 由于存在actor的按照时间间隔去预测, 故这里不休眠, 后期修改为事件机制
                    if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                        time.sleep(CONFIG.idle_sleep_second)

                        # process_run_count置0, 规避溢出
                        self.process_run_idle_count = 0

            except Exception as e:
                self.logger.exception(f"predict failed to run predict. exit. Error is: {e}, ")

        if CONFIG.actor_server_async:
            self.send_server.stop()
            self.recv_server.stop()
            self.logger.info("predict self.send_server.stop self.recv_server.stop success")
        else:
            self.send_server.stop()
            self.logger.info("predict self.send_server.stop success")

        for policy, agent_wrapper in self.policy_agent_wrapper_maps.items():
            agent_wrapper.close()
            self.logger.info(f"predict {policy} agent_wrapper.close success")

        self.model_file_sync_wrapper.stop()
        self.logger.info("predict self.model_file_sync_wrapper.stop success")
