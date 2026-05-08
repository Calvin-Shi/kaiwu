#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import multiprocessing
import schedule
import datetime
import time
from common_python.config.config_control import CONFIG
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.common_func import (
    set_schedule_event,
    get_random,
    decompress_data,
)
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from common_python.ipc.zmq_util import ZmqServer, ZmqConfig
import numpy as np
from guppy import hpy
import psutil
from kaiwudrl.common.config.algo_conf import AlgoConf
from common_python.worker.worker import Worker, WorkerConfig
from kaiwudrl.server.learner.trainer import Trainer


class LearnerServer(Worker):
    """
    数据流程如下:
    1. aisrv zmq_client --> learner zmq_server
    2. learner zmq_server --> learner reverb_client
    3. learner reverb_client --> learner reverb_server

    由于learner使用的python版本的reverb性能比aisrv使用的C++版本的zmq性能差, 出现收发速度不匹配问题, 故这里learner上的zmq和reverb进程情况如下:
    1. zmq_server, 1个, 端口设置为CONFIG.reverb_svr_port - 1, 需要考虑到learner_server和reverb_server同时启动的场景
    2. reverb client server, N个, N由配置文件项决定
    """

    def __init__(self, shared_mem_buffer=None) -> None:
        # 进程pid
        self.current_pid = os.getpid()

        worker_config = WorkerConfig(
            worker_name="learner_server",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        zmq_config = ZmqConfig(
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
        self.zmq_server = ZmqServer(CONFIG.ip_address, str(int(CONFIG.reverb_svr_port) - 1), zmq_config, pull_mode=True)

        self.process_run_count = 0

        # 停止标志位
        self.exit_flag = multiprocessing.Value("b", False)

        """
        具体处理样本的类, 支持负载均衡
        1. 如果是reverb则是reverb_server
        2. 如果是zmq则是zmq_server
        """
        self.sample_send_server_wrappers = []

        # learner从aisrv收到的包的个数
        self.learner_recv_success_sample_count_from_aisrv = 0
        self.learner_recv_fail_sample_count_from_aisrv = 0
        self.last_run_schedule_time_by_stat = time.time()

        # 跨进程共享的底层 MemBuffer 对象
        self.shared_mem_buffer = shared_mem_buffer

        # 全局的PB解析对象
        # self.pb_req = AisrvLearnerRequest()

    def get_data_and_send_to_queue(self):
        """
        从网络上收到数据, 并且放入到本地队列, PULL模式不需要回包
        """

        # get sample data
        try:
            data = self.zmq_server.recv(block=True, binary=True)
            if not data:
                return None

            if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
                train_data = [{"input_datas": np.array(sample, dtype=np.float32)} for sample in data]
            else:
                train_data = data

            # 随机选择发送给sample_send_server_wrappers列表
            idx = get_random(0, len(self.sample_send_server_wrappers) - 1)
            self.sample_send_server_wrappers[idx].put_data(train_data)

            # 统计值, 只能是接收到的包大小
            self.learner_recv_success_sample_count_from_aisrv += 1

        except Exception as e:
            # 这里暂时没有请求aisrv请求是正常现象, 下一个循环接着处理
            self.logger.exception(
                f"learner_server get_data_and_send_to_queue error: {str(e)}, ",
                g_not_server_label,
            )

            return None

    # 返回reverb server的IP和端口
    def get_zmq_ip(self):
        return f"{self.ip_address}:{int(CONFIG.reverb_svr_port)-1}"

    def before_run(self):
        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        self.zmq_server.bind()
        self.logger.info(f"learner_server start at: {CONFIG.ip_address}:{int(CONFIG.reverb_svr_port) - 1}")

        self.process_run_count = 0

        # 定时器采用schedule, need pip install schedule
        # self.zmq_server_stat_schedule()

        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            for i in range(int(CONFIG.learner_send_sample_server_count)):
                learner_server_reverb = LearnerServerReverb(i)
                learner_server_reverb.start()
                self.sample_send_server_wrappers.append(learner_server_reverb)

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            # 直接创建 LearnerServerZmq 子进程，它们会在各自的 before_run 中创建 replay_buffer_wrapper
            for i in range(int(CONFIG.learner_send_sample_server_count)):
                learner_server_zmq = LearnerServerZmq(i, self.shared_mem_buffer)
                learner_server_zmq.start()
                self.sample_send_server_wrappers.append(learner_server_zmq)

        else:
            pass

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"learner_server start success at pid {self.current_pid}, use protocl flatbuffer",
            g_not_server_label,
        )

        return True

    # 周期性打印内存占用情况
    def zmq_server_stat(self):
        self.logger.info(
            f"learner_server learner_recv_success_sample_count_from_aisrv: "
            f"{self.learner_recv_success_sample_count_from_aisrv}",
            g_not_server_label,
        )

        # h = hpy()
        # self.logger.info(h.heap())
        # self.logger.info(f'learner_server gc count {gc.get_count()}')
        # self.logger.info(f'learner_server memory: {psutil.virtual_memory()}')
        # self.logger.info(f'learner_server msg count: {self.zmq_server.get_cache_message_count()}')

    # 定时器采用schedule, need pip install schedule
    def zmq_server_stat_schedule(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_server_stat)

    def run_tasks_periodically(self):
        """
        主要是周期性的调用统计函数
        """
        now = time.time()
        if now - self.last_run_schedule_time_by_stat >= CONFIG.prometheus_stat_per_minutes * 60:
            self.zmq_server_stat()
            self.last_run_schedule_time_by_stat = now

    def run_once(self):

        # get sample data
        self.get_data_and_send_to_queue()

        # 启动记录发送成功失败的数目的定时器
        # schedule.run_pending()
        self.run_tasks_periodically()

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("learner_server LearnerServerZmq stop success", g_not_server_label)

    def after_run(self) -> bool:
        pass

    def run(self) -> None:
        if not self.before_run():
            self.logger.error("learner_server before_run failed", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                """
                由于LearnerServerZmq进程里都是IO操作比较多, 这里减少休息时间
                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0
                """

            except Exception as e:
                self.logger.exception(
                    f"learner_server_zmq run error: {str(e)}",
                    g_not_server_label,
                )


class LearnerServerZmq(Worker):
    """
    该类主要是采用zmq进行通信, 处理aisrv<-->learner之间的数据
    """

    def __init__(self, idx, shared_mem_buffer=None) -> None:
        # 进程pid
        self.current_pid = os.getpid()

        worker_config = WorkerConfig(
            worker_name="learner_server_zmq",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        self.process_run_count = 0

        # 停止标志位
        self.exit_flag = multiprocessing.Value("b", False)

        self.sample_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 接收到的样本个数
        self.learner_recv_success_sample_count_from_aisrv = 0
        self.learner_recv_fail_sample_count_from_aisrv = 0
        self.last_run_schedule_time_by_stat = time.time()

        self.idx = idx

        """
        因为训练进程和learner_server进程需要采用相同的replay_buffer_wrapper, 否则会因为不同进程之间内存隔离而导致读取和写入样本数据问题
        解决方案: 在 LearnerServerZmq 进程中创建新的 replay_buffer_wrapper，但共享底层的 mem_buffer（mem_buffer 使用 multiprocessing.Array 实现跨进程共享）
        """
        self.shared_mem_buffer = shared_mem_buffer
        self.replay_buffer_wrapper = None

    def set_replay_buffer_wrapper(self, replay_buffer_wrapper):
        self.replay_buffer_wrapper = replay_buffer_wrapper

    def put_data(self, data):
        if not data:
            return False

        self.sample_queue.put(data)
        return True

    def get_data(self):
        """
        从网络上收到数据, 并且放入到本地队列
        """
        return self.sample_queue.get()

    def before_run(self):
        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        self.process_run_count = 0

        # 在子进程中创建 replay_buffer_wrapper，使用共享的 mem_buffer
        if self.shared_mem_buffer is not None:
            # 获取 tensor 配置信息
            trainer_instance = Trainer()
            tensor_names = trainer_instance.tensor_names
            tensor_dtypes = trainer_instance.tensor_dtypes

            # 创建新的 ReplayBufferWrapper
            from kaiwudrl.common.replay_buffer.replay_buffer_wrapper import ReplayBufferWrapper

            self.replay_buffer_wrapper = ReplayBufferWrapper(tensor_names, tensor_dtypes, self.logger)

            # 使用共享的 mem_buffer 进行初始化
            self.replay_buffer_wrapper.init_with_shared_mem_buffer(self.shared_mem_buffer)
        else:
            self.logger.error(
                f"learner_server_zmq {self.idx} shared_mem_buffer is None! Cannot create replay_buffer_wrapper!",
                g_not_server_label,
            )
            return False

        # 定时器采用schedule, need pip install schedule
        # self.zmq_server_stat_schedule()

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"learner_server_zmq start success at pid {self.current_pid}, use protocl flatbuffer",
            g_not_server_label,
        )

        return True

    # 周期性打印内存占用情况
    def zmq_server_stat(self):
        self.logger.info(
            f"learner_server_zmq learner recv sample from aisrv by zmq: "
            f"{self.learner_recv_success_sample_count_from_aisrv}, idx: {self.idx }",
            g_not_server_label,
        )

        # h = hpy()
        # self.logger.info(h.heap())
        # self.logger.info(f'learner_server_zmq gc count {gc.get_count()}')
        # self.logger.info(f'learner_server_zmq memory: {psutil.virtual_memory()}')
        # self.logger.info(f'learner_server_zmq msg count: {self.zmq_server.get_cache_message_count()}')

    # 定时器采用schedule, need pip install schedule
    def zmq_server_stat_schedule(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_server_stat)

    def send_sample_data_by_mem_buffer(self, data):
        if not data:
            return False

        """
        lz4 decompress + pb, lz4解压缩大小设置需要和aisrv的learner_proxy对齐
        1. lz4压缩比20
        2. 320帧样本PB序列化大小为11MB
        3. 按照1和2的结果, 则设置为300MB比较安全
        """
        try:
            data = decompress_data(data, uncompressed_size=CONFIG.lz4_learner_uncompressed_size)
            self.replay_buffer_wrapper.add_sample(data)

            # 增加统计值, 此时是样本条数
            self.learner_recv_success_sample_count_from_aisrv += len(data)

            return True

        except Exception as e:
            self.logger.exception(
                f"learner_server_zmq get_data error: {str(e)}",
                g_not_server_label,
            )
            return False

    def run_tasks_periodically(self):
        """
        主要是周期性的调用统计函数
        """
        now = time.time()
        if now - self.last_run_schedule_time_by_stat >= CONFIG.prometheus_stat_per_minutes * 60:
            self.zmq_server_stat()
            self.last_run_schedule_time_by_stat = now

    def run_once(self):

        # get sample data
        datas = self.get_data()
        if datas:
            self.send_sample_data_by_mem_buffer(datas)

        # 启动记录发送成功失败的数目的定时器
        # schedule.run_pending()
        self.run_tasks_periodically()

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("learner_server_zmq LearnerServerZmq stop success", g_not_server_label)

    def after_run(self) -> bool:
        pass

    def run(self) -> None:
        if not self.before_run():
            self.logger.error("learner_server_zmq before_run failed", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                """
                由于LearnerServerZmq进程里都是IO操作比较多, 这里减少休息时间
                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0
                """

            except Exception as e:
                self.logger.exception(
                    f"learner_server_zmq run error: {str(e)}",
                    g_not_server_label,
                )


class LearnerServerReverb(Worker):
    """
    该类主要是将数据采用reverb_client发送出去
    """

    def __init__(self, idx) -> None:
        # 进程pid
        self.current_pid = os.getpid()

        worker_config = WorkerConfig(
            worker_name="learner_server_reverb",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        self.learner_addr = "localhost"
        self.learner_port = int(CONFIG.reverb_svr_port)
        self.process_run_count = 0

        # 停止标志位
        self.exit_flag = multiprocessing.Value("b", False)

        # 需要发送给learner的样本数据
        self.train_data = None

        # reverb 工具类, aisrv上采用reverb client将数据发送给learn进程上的reverb server
        self.reverb_table_names = None

        # 进程是否退出, 用于在对端异常条件下, 主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

        # 单个reverb_util对象
        self.reverb_util = None

        # index, 只是对第1个进行上报处理
        self.idx = idx

        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)

    def put_data(self, train_data):
        if not train_data or self.msg_queue.full():
            return False

        self.msg_queue.put(train_data)
        return True

    def get_data(self):
        # 判断队列为空self.msg_queue.empty()时, 可能出现报错Connection reset by peer, 需要使用try-except形式
        try:
            if not self.msg_queue.empty():
                self.train_data = self.msg_queue.get()

        except Exception as e:
            self.train_data = None

    # 返回reverb server的IP和端口
    def get_reverb_ip(self):
        return f"{self.learner_addr}:{self.learner_port}"

    def before_run(self):
        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        self.logger.info(
            f"learner_server_reverb {self.idx} start at pid {self.current_pid}",
            g_not_server_label,
        )

        # 必须放在这里赋值, 否则reverb client会卡住
        from common_python.ipc.reverb_util import ReverbUtil, ReverbUtilConfig

        config = ReverbUtilConfig(
            reverb_sampler=CONFIG.reverb_sampler,
            reverb_client_max_sequence_length=CONFIG.reverb_client_max_sequence_length,
            reverb_client_chunk_length=CONFIG.reverb_client_chunk_length,
        )
        self.reverb_util = ReverbUtil(f"{self.learner_addr}:{self.learner_port}", config, self.logger)

        self.reverb_table_names = [
            "{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))
        ]
        self.logger.info(
            f"learner_server_reverb {self.idx} send reverb server tables is {self.reverb_table_names}",
            g_not_server_label,
        )

        # 因为需要打印统计日志代表进程存活, 故都需要设置下
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
            self.send_to_reverb_server_stat()

        self.process_run_count = 0

        # aisrv朝learner发送的最大样本大小
        self.max_sample_size = 0

        return True

    def reverb_server_stat(self):
        (
            total_succ_cnt,
            total_error_cnt,
        ) = self.reverb_util.get_send_to_reverb_server_stat()

        # 只有第一个才上报普罗米修斯
        if int(CONFIG.use_prometheus) and self.idx == 0:

            # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
            try:
                msg_queue_size = self.msg_queue.qsize()
            except Exception as e:
                msg_queue_size = 0

            monitor_data = {
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_SUCC_CNT: total_succ_cnt,
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_ERR_CNT: total_error_cnt,
                KaiwuDRLDefine.MONITOR_MAX_SAMPLE_SIZE: self.max_sample_size,
                KaiwuDRLDefine.MONITOR_LEARNER_ZMQ_REVERB_QUEUE_LEN: msg_queue_size,
            }

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        # 打印日志, 是为了确保进程正常, 1分钟打印1次性能可控
        self.logger.info(
            f"learner_server_reverb {self.idx} send reverb server stat, "
            f"succ_cnt is {total_succ_cnt}, error_cnt is {total_error_cnt}",
            g_not_server_label,
        )

    # 定时器采用schedule, need pip install schedule
    def send_to_reverb_server_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.reverb_server_stat)

    def run_once(self):

        # get sample data
        self.get_data()

        # use reverb client send sample data to reverb server
        self.send_msg_use_reverb_client()

        # 重新设置self.train_data为None
        self.train_data = None

        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info(
            f"learner_server_reverb {self.idx} LearnerServerReverb stop success",
            g_not_server_label,
        )

    def after_run(self) -> bool:
        pass

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(
                f"learner_server_reverb {self.idx} before_run failed, so return",
                g_not_server_label,
            )
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(
                    f"learner_server_reverb {self.idx} run error: {str(e)}, ",
                    g_not_server_label,
                )

    # use reverb client send msg to reverb server
    def send_msg_use_reverb_client(self):
        if not self.train_data:
            return

        # reverb_client发送
        self.reverb_util.write_to_reverb_server_simple(self.reverb_table_names, self.train_data)

        # 更新最大样本大小
        input_datas_list = self.train_data
        sample_size = 0
        for agent in input_datas_list:
            sample_size += agent["input_datas"].nbytes

        # 更新最大样本大小
        if sample_size > self.max_sample_size:
            self.max_sample_size = sample_size
