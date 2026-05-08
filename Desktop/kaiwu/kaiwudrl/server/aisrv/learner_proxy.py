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
from multiprocessing.shared_memory import SharedMemory
import schedule
import datetime
import time
import numpy as np
from common_python.config.config_control import CONFIG
from common_python.logging.kaiwu_logger import g_not_server_label
from kaiwudrl.common.utils.common_func import (
    set_schedule_event,
    compress_data,
    get_uuid,
    get_host_ip,
)
from common_python.ipc.zmq_util import ZmqClient, ZmqConfig
from kaiwudrl.common.utils.shared_memory import SharedMemoryExtend
from kaiwudrl.common.replay_buffer.shared_memory_replay_buffer import SharedMemoryReplayBuffer
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from common_python.worker.worker import Worker, WorkerConfig


class LearnerProxy(Worker):
    def __init__(self, policy_name, learner_addr, context) -> None:
        # 进程pid
        self.current_pid = os.getpid()
        worker_config = WorkerConfig(
            worker_name="learner_proxy",
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
        self.learner_addr = learner_addr[0]
        self.learner_port = learner_addr[1]

        """
        aisrv里主线程放进该Queue, learnproxy采用reverb client发送给reverb server
        1. 采用multiprocessing.Manager().Queue()发现在单条数据比较大时耗时较多, 而采用multiprocessing.Queue()耗时比较少
        2. multiprocessing.Queue()是稳定的, 只是在内存紧张时会报错, 故需要先解决内存紧张的问题
        3. 采用multiprocessing.Manager().Queue()会启动单独的进程处理, 增加CPU消耗

        综合上面的情况, 采用multiprocessing.Queue()
        """
        if (
            CONFIG.workflow_learner_proxy_communication
            == KaiwuDRLDefine.WORKFLOW_LEARNER_PROXY_COMMUNICATION_SHARED_MEMORY
        ):
            # SharedMemory高性能零拷贝通信
            buffer_size = CONFIG.replay_buffer_capacity * CONFIG.max_sample_size

            # 仅创建SharedMemory对象，不立即映射（避免SIGBUS）
            self.shm_data = SharedMemory(create=True, size=buffer_size * 4)
            self.shm_prioritized = SharedMemory(create=True, size=buffer_size * 4)
            self.shm_metadata = SharedMemory(create=True, size=CONFIG.replay_buffer_capacity * 256 * 4)

            # 延迟映射标记
            self._shm_data_buf = None
            self._shm_prioritized_buf = None
            self._shm_metadata_buf = None
            self._mapping_lock = multiprocessing.Lock()

            # 同步原语
            self.write_idx = multiprocessing.Value("i", 0)
            self.read_idx = multiprocessing.Value("i", 0)
            self.count = multiprocessing.Value("i", 0)
            self.write_lock = multiprocessing.Lock()

        else:
            self.msg_queue_data = multiprocessing.Queue(CONFIG.queue_size)

        self.msg_queue_control = multiprocessing.Queue(CONFIG.queue_size)

        self.context = context

        # 根据不同的replay_buffer_type设置不同的对象
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            # reverb 工具类, aisrv上采用reverb client将数据发送给learn进程上的reverb server
            self.reverb_util = None
            self.reverb_table_names = None

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            self.zmq_client = None
            self.client_id = None

            self.send_to_learner_err_cnt = 0
            self.send_to_learner_succ_cnt = 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            self.shared_memory = None
            self.send_to_learner_err_cnt = 0
            self.send_to_learner_succ_cnt = 0
        else:
            pass

        # zmq_client, 某些管理流需要从aisrv传到learner上去执行, 故采用zmq, 因为与前面的self.zmq_client可能命名冲突, 然后定义其他名字
        self.client_id_for_learner = 0
        self.zmq_client_for_learner = None

        # 进程是否退出, 用于在对端异常条件下, 主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

    @property
    def shm_data_buf(self):
        """延迟初始化SharedMemory映射（线程安全）"""
        if self._shm_data_buf is None:
            with self._mapping_lock:
                if self._shm_data_buf is None:
                    self._shm_data_buf = np.ndarray(
                        (CONFIG.replay_buffer_capacity, CONFIG.max_sample_size),
                        dtype=np.float32,
                        buffer=self.shm_data.buf,
                    )
        return self._shm_data_buf

    @property
    def shm_prioritized_buf(self):
        """延迟初始化SharedMemory映射（线程安全）"""
        if self._shm_prioritized_buf is None:
            with self._mapping_lock:
                if self._shm_prioritized_buf is None:
                    self._shm_prioritized_buf = np.ndarray(
                        (CONFIG.replay_buffer_capacity, CONFIG.max_sample_size),
                        dtype=np.float32,
                        buffer=self.shm_prioritized.buf,
                    )
        return self._shm_prioritized_buf

    @property
    def shm_metadata_buf(self):
        """延迟初始化SharedMemory映射（线程安全）"""
        if self._shm_metadata_buf is None:
            with self._mapping_lock:
                if self._shm_metadata_buf is None:
                    self._shm_metadata_buf = np.ndarray(
                        (CONFIG.replay_buffer_capacity, 256), dtype=np.int32, buffer=self.shm_metadata.buf
                    )
        return self._shm_metadata_buf

    # 不用区分哪个agent发送的样本
    def put_data(self, slot_id, train_data):
        if (
            CONFIG.workflow_learner_proxy_communication
            == KaiwuDRLDefine.WORKFLOW_LEARNER_PROXY_COMMUNICATION_SHARED_MEMORY
        ):
            try:
                message_value = train_data.get(KaiwuDRLDefine.MESSAGE_VALUE)
                if not message_value:
                    return False

                actual_train_data = message_value.get("train_data")
                actual_train_data_prioritized = message_value.get("train_data_prioritized")

                if actual_train_data is None or actual_train_data_prioritized is None:
                    return False

                # 性能优化：批量处理numpy数组列表
                # 假设所有样本维度相同（来自definition.py的标准化输出）
                if isinstance(actual_train_data, (list, tuple)) and len(actual_train_data) > 0:
                    if isinstance(actual_train_data[0], np.ndarray):
                        # 快速路径：所有样本是同维度的numpy数组
                        # 使用np.vstack一次性堆叠，然后ravel展平
                        stacked = np.vstack(actual_train_data)
                        sample_dim = (
                            actual_train_data[0].shape[0]
                            if actual_train_data[0].ndim == 1
                            else actual_train_data[0].size
                        )
                        sample_lengths_data = [sample_dim] * len(actual_train_data)
                        flattened_data = stacked.ravel()
                    else:
                        # 兜底：逐个处理
                        flattened_parts_data = []
                        sample_lengths_data = []
                        for sample in actual_train_data:
                            arr = np.asarray(sample, dtype=np.float32).flatten()
                            sample_lengths_data.append(len(arr))
                            flattened_parts_data.append(arr)
                        flattened_data = np.concatenate(flattened_parts_data)
                elif isinstance(actual_train_data, np.ndarray):
                    flattened_data = actual_train_data.flatten()
                    sample_lengths_data = [len(flattened_data)]
                else:
                    flattened_data = np.asarray(actual_train_data, dtype=np.float32).flatten()
                    sample_lengths_data = [len(flattened_data)]

                # 同样优化prioritized数据
                if isinstance(actual_train_data_prioritized, (list, tuple)) and len(actual_train_data_prioritized) > 0:
                    if isinstance(actual_train_data_prioritized[0], np.ndarray):
                        stacked_prio = np.vstack(actual_train_data_prioritized)
                        sample_dim_prio = (
                            actual_train_data_prioritized[0].shape[0]
                            if actual_train_data_prioritized[0].ndim == 1
                            else actual_train_data_prioritized[0].size
                        )
                        sample_lengths_prio = [sample_dim_prio] * len(actual_train_data_prioritized)
                        flattened_prioritized = stacked_prio.ravel()
                    else:
                        flattened_parts_prio = []
                        sample_lengths_prio = []
                        for sample in actual_train_data_prioritized:
                            arr = np.asarray(sample, dtype=np.float32).flatten()
                            sample_lengths_prio.append(len(arr))
                            flattened_parts_prio.append(arr)
                        flattened_prioritized = np.concatenate(flattened_parts_prio)
                elif isinstance(actual_train_data_prioritized, np.ndarray):
                    flattened_prioritized = actual_train_data_prioritized.flatten()
                    sample_lengths_prio = [len(flattened_prioritized)]
                else:
                    flattened_prioritized = np.asarray(actual_train_data_prioritized, dtype=np.float32).flatten()
                    sample_lengths_prio = [len(flattened_prioritized)]

                # 检查限制
                if len(flattened_data) > CONFIG.max_sample_size or len(flattened_prioritized) > CONFIG.max_sample_size:
                    return False

                with self.write_lock:
                    if self.count.value >= CONFIG.replay_buffer_capacity:
                        return False

                    idx = self.write_idx.value

                    # 零拷贝写入
                    self.shm_data_buf[idx, : len(flattened_data)] = flattened_data
                    self.shm_prioritized_buf[idx, : len(flattened_prioritized)] = flattened_prioritized

                    # 元数据：[total_len_data, total_len_prio, num_samples_data, num_samples_prio, len1, len2, ...]
                    # 前4个固定位置，后面存储每个样本的长度
                    self.shm_metadata_buf[idx, 0] = len(flattened_data)
                    self.shm_metadata_buf[idx, 1] = len(flattened_prioritized)
                    self.shm_metadata_buf[idx, 2] = len(sample_lengths_data)
                    self.shm_metadata_buf[idx, 3] = len(sample_lengths_prio)

                    # 批量写入样本长度（使用numpy切片赋值）
                    self.shm_metadata_buf[idx, 4 : 4 + len(sample_lengths_data)] = sample_lengths_data
                    offset = 4 + len(sample_lengths_data)
                    self.shm_metadata_buf[idx, offset : offset + len(sample_lengths_prio)] = sample_lengths_prio

                    self.write_idx.value = (idx + 1) % CONFIG.replay_buffer_capacity
                    self.count.value += 1

                return True

            except Exception as e:
                import traceback

                print(f"learner_proxy put_data error: {str(e)}\n{traceback.format_exc()}")
                return False
        else:
            if self.msg_queue_data.full():
                return False

            self.msg_queue_data.put(train_data)
            return True

    def put_data_control(self, slot_id, control_data):
        if self.msg_queue_control.full():
            return False

        self.msg_queue_control.put(control_data)
        return True

    # 返回参数是train_data或者control_data
    def get_data(self):
        """
        优先管理流, 其次数据流
        """
        control_data = self.get_control_data()
        if control_data is not None:
            return control_data

        try:
            msg_queue_size = self.msg_queue_data.qsize()
        except Exception as e:
            msg_queue_size = 0

        if msg_queue_size > 0:
            return self.msg_queue_data.get()

        return None

    def get_control_data(self):
        try:
            msg_queue_size = self.msg_queue_control.qsize()
        except Exception as e:
            msg_queue_size = 0

        if msg_queue_size > 0:
            return self.msg_queue_control.get()

        return None

    def get_data_batch(self, batch_size=32):
        """
        批量读取数据，提升吞吐量
        返回: list of train_data_dict，如果没有数据返回空列表
        优先管理流, 其次数据流
        """
        batch = []

        control_data = self.get_control_data()
        if control_data is not None:
            batch.append(control_data)
            return batch

        try:
            with self.write_lock:
                available = min(self.count.value, batch_size)
                if available == 0:
                    return batch

                for _ in range(available):
                    idx = self.read_idx.value

                    # 读取元数据
                    total_len_data = int(self.shm_metadata_buf[idx, 0])
                    total_len_prio = int(self.shm_metadata_buf[idx, 1])
                    num_samples_data = int(self.shm_metadata_buf[idx, 2])
                    num_samples_prio = int(self.shm_metadata_buf[idx, 3])

                    # 读取扁平数据
                    flattened_data = self.shm_data_buf[idx, :total_len_data].copy()
                    flattened_prioritized = self.shm_prioritized_buf[idx, :total_len_prio].copy()

                    # 优化：使用numpy切片读取样本长度
                    sample_lengths_data = self.shm_metadata_buf[idx, 4 : 4 + num_samples_data].astype(int)
                    offset = 4 + num_samples_data
                    sample_lengths_prio = self.shm_metadata_buf[idx, offset : offset + num_samples_prio].astype(int)

                    # 优化：使用np.split恢复列表结构（如果所有样本长度相同）
                    if num_samples_data > 0:
                        if num_samples_data == 1:
                            train_data_list = [flattened_data]
                        elif np.all(sample_lengths_data == sample_lengths_data[0]):
                            # 所有样本维度相同，使用reshape + list
                            sample_dim = sample_lengths_data[0]
                            train_data_list = list(flattened_data.reshape(num_samples_data, sample_dim))
                        else:
                            # 样本维度不同，使用np.split
                            split_indices = np.cumsum(sample_lengths_data[:-1])
                            train_data_list = np.split(flattened_data, split_indices)
                    else:
                        train_data_list = []

                    if num_samples_prio > 0:
                        if num_samples_prio == 1:
                            train_data_prio_list = [flattened_prioritized]
                        elif np.all(sample_lengths_prio == sample_lengths_prio[0]):
                            sample_dim_prio = sample_lengths_prio[0]
                            train_data_prio_list = list(
                                flattened_prioritized.reshape(num_samples_prio, sample_dim_prio)
                            )
                        else:
                            split_indices_prio = np.cumsum(sample_lengths_prio[:-1])
                            train_data_prio_list = np.split(flattened_prioritized, split_indices_prio)
                    else:
                        train_data_prio_list = []

                    batch.append(
                        {
                            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_TRAIN,
                            KaiwuDRLDefine.MESSAGE_VALUE: {
                                "train_data": train_data_list,
                                "train_data_prioritized": train_data_prio_list,
                            },
                        }
                    )

                    self.read_idx.value = (idx + 1) % CONFIG.replay_buffer_capacity
                    self.count.value -= 1

        except Exception as e:
            self.logger.exception(f"learner_proxy get_data_batch error: {str(e)}")

        return batch

    # 返回reverb server的IP和端口
    def get_reverb_ip(self):
        return f"{self.learner_addr}:{self.learner_port}"

    def before_run(self):
        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

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

        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
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
                f"learner_proxy reverb server is {self.learner_addr}:{self.learner_port}, "
                f"table_names is {self.reverb_table_names}",
                g_not_server_label,
            )
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:

            """
            aisrv <--> learner之间, learner是支持多个aisrv的, 故learner需要知道各个aisrv的client_id, 故这里采用uuid方式
            """
            port = int(CONFIG.reverb_svr_port) - 1
            self.client_id = get_uuid()
            self.zmq_client = ZmqClient(
                str(self.client_id),
                self.learner_addr,
                str(port),
                self.zmq_config,
                push_mode=True,
            )
            self.zmq_client.connect()

            self.logger.info(
                f"learner_proxy send reverb server use zmq, connect to {self.learner_addr}, "
                f"port is {port}, client_id is {self.client_id}",
                g_not_server_label,
            )
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            # 获取aisrv的所在的ip
            ip = get_host_ip()
            self.shared_memory = SharedMemoryReplayBuffer(
                base_name=KaiwuDRLDefine.SHARED_MEMORY_NAME,
                is_producer=True,
                producer_identifier=ip,
                discovery_interval=10.0,
                size=4 * 1024 * 1024,
                create=True,
                unlink_on_close=False,
                logger=self.logger,
            )
            self.logger.info(
                f"learner_proxy send reverb server use shared_memory, shared_memory_name {KaiwuDRLDefine.SHARED_MEMORY_NAME}, ",
                g_not_server_label,
            )
        else:
            pass

        # aisrv与learner之间的zmq管理流通信, 无论什么场景都需要使用
        self.client_id_for_learner = get_uuid()
        port = int(CONFIG.reverb_svr_port) - 2
        self.zmq_client_for_learner = ZmqClient(
            str(self.client_id_for_learner),
            self.learner_addr,
            port,
            self.zmq_config,
        )
        self.zmq_client_for_learner.connect()
        self.logger.info(
            f"learner_proxy zmq client connect at {self.learner_addr}:{port} "
            f"with client_id {self.client_id_for_learner}",
            g_not_server_label,
        )

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
            self.send_to_sample_server_stat()

        self.process_run_count = 0

        # aisrv朝learner发送的最大样本大小
        self.max_sample_size = 0

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"learner_proxy policy_name: {self.policy_name}, start success at pid {self.current_pid}",
            g_not_server_label,
        )

        return True

    def sample_server_stat(self):
        """
        获取发送样本的统计情况
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            succ_cnt, error_cnt = self.reverb_util.get_send_to_reverb_server_stat()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            succ_cnt, error_cnt = (
                self.send_to_learner_succ_cnt,
                self.send_to_learner_err_cnt,
            )
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            succ_cnt, error_cnt = (
                self.send_to_learner_succ_cnt,
                self.send_to_learner_err_cnt,
            )
        else:
            succ_cnt, error_cnt = 0, 0

        if int(CONFIG.use_prometheus):

            if (
                CONFIG.workflow_learner_proxy_communication
                == KaiwuDRLDefine.WORKFLOW_LEARNER_PROXY_COMMUNICATION_SHARED_MEMORY
            ):
                msg_queue_size = self.count.value
            else:
                # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
                try:
                    msg_queue_size = self.msg_queue_data.qsize()
                except Exception as e:
                    msg_queue_size = 0

            monitor_data = {
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_SUCC_CNT: succ_cnt,
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_ERR_CNT: error_cnt,
                KaiwuDRLDefine.MONITOR_MAX_SAMPLE_SIZE: self.max_sample_size,
                KaiwuDRLDefine.MONITOR_AISRV_LEARNER_PROXY_QUEUE_LEN: msg_queue_size,
            }

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        self.logger.info(
            f"learner_proxy send sample stat, succ_cnt is {succ_cnt}, error_cnt is {error_cnt}",
            g_not_server_label,
        )

    # 定时器采用schedule, need pip install schedule
    def send_to_sample_server_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.sample_server_stat)

    # use shard memory
    def send_msg_use_shared_memory(self, train_data):
        if train_data is None or len(train_data) == 0:
            return

        train_data_size = len(train_data)
        try:
            # 采用共享内存放入数据
            self.shared_memory.try_send(train_data)
            self.send_to_learner_succ_cnt += train_data_size
        except Exception as e:
            self.send_to_learner_err_cnt += train_data_size

        self.after_send_train_data_simple(train_data)

    def run_once(self):
        """
        支持情况:
        1. Queue模式, 稳定性优先
        2. SharedMemory模式, 性能优先
        """
        if (
            CONFIG.workflow_learner_proxy_communication
            == KaiwuDRLDefine.WORKFLOW_LEARNER_PROXY_COMMUNICATION_SHARED_MEMORY
        ):
            return self.run_once_shared_memory_mode()
        else:
            return self.run_once_queue_mode()

    def msg_process_save_model(self, train_data):
        self.zmq_client_for_learner.send(train_data, binary=False)
        self.logger.info(
            f"learner_proxy send save_model data to learner",
            g_not_server_label,
        )

        result = self.zmq_client_for_learner.recv(binary=False)
        if (
            result
            and result.get(KaiwuDRLDefine.MESSAGE_TYPE) == KaiwuDRLDefine.MESSAGE_SAVE_MODEL
            and result.get(KaiwuDRLDefine.MESSAGE_VALUE)
        ):
            self.logger.info(
                f"learner_proxy recv save_model data result from learner success",
                g_not_server_label,
            )
        else:
            self.logger.error(
                f"learner_proxy recv save_model data result from learner failed",
                g_not_server_label,
            )

    def msg_process_process_stop(self, train_data):
        self.zmq_client_for_learner.send(train_data, binary=False)
        self.logger.info(
            f"learner_proxy send process_stop data to learner",
            g_not_server_label,
        )

        result = self.zmq_client_for_learner.recv(binary=False)
        if (
            result
            and result.get(KaiwuDRLDefine.MESSAGE_TYPE) == KaiwuDRLDefine.MESSAGE_PROCESS_STOP
            and result.get(KaiwuDRLDefine.MESSAGE_VALUE)
        ):
            self.logger.info(
                f"learner_proxy recv process_stop data result from learner success",
                g_not_server_label,
            )
        else:
            self.logger.error(
                f"learner_proxy recv process_stop data result from learner failed",
                g_not_server_label,
            )

    def run_once_queue_mode(self):
        """
        Queue模式的run_once实现
        每次处理一条数据，立即发送
        """

        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()

        # get sample data
        train_data = self.get_data()
        if train_data is None or len(train_data) == 0:
            return False

        """
        根据不同的协议发送不同的操作:
        1. 如果是训练, 则按照样本发送的逻辑, 采用reverb, 如果不能采用reverb则采用zmq
        2. 如果是保留模型文件, 则按照发送模型文件的逻辑, 采用zmq
        """
        message_type = train_data.get(KaiwuDRLDefine.MESSAGE_TYPE)
        message_value = train_data.get(KaiwuDRLDefine.MESSAGE_VALUE)
        if message_type == KaiwuDRLDefine.MESSAGE_TRAIN:
            td = message_value.get("train_data")
            td_prioritized = message_value.get("train_data_prioritized")
            if td is not None and td_prioritized is not None:
                if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
                    # use reverb client send sample data to reverb server
                    self.send_msg_use_reverb_client(td, td_prioritized)

                elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
                    self.send_msg_use_zmq_client(td)

                elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
                    self.send_msg_use_shared_memory(td)

                else:
                    return False
            else:
                return False

        elif message_type == KaiwuDRLDefine.MESSAGE_SAVE_MODEL:
            self.msg_process_save_model(train_data)

        elif message_type == KaiwuDRLDefine.MESSAGE_PROCESS_STOP:
            self.msg_process_process_stop(train_data)

        else:
            self.logger.error(
                f"learner_proxy recv un support message_type: {message_type}, please check",
                g_not_server_label,
            )
            return False

        return True

    def run_once_shared_memory_mode(self):
        """
        SharedMemory模式的run_once实现（批量优化）
        批量读取、批量处理、批量发送
        """

        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()

        # 性能优化：根据实际测试调优的参数
        max_samples_per_call = 512  # 减少单次处理量，提高循环频率
        batch_size = 256  # 增大批量，减少create_item调用次数

        reverb_batch_data = []  # 收集需要批量发送的reverb数据
        reverb_batch_prioritized = []
        total_processed = 0

        # 循环读取，直到队列为空或达到单次处理上限
        while total_processed < max_samples_per_call:
            train_data_batch = self.get_data_batch(batch_size)

            # 如果批量读取为空
            if not train_data_batch:
                break

            # 批量处理每条数据
            for train_data in train_data_batch:
                """
                根据不同的协议发送不同的操作:
                1. 如果是训练, 则按照样本发送的逻辑, 采用reverb, 如果不能采用reverb则采用zmq
                2. 如果是保留模型文件, 则按照发送模型文件的逻辑, 采用zmq
                """
                message_type = train_data.get(KaiwuDRLDefine.MESSAGE_TYPE)
                message_value = train_data.get(KaiwuDRLDefine.MESSAGE_VALUE)
                if message_type == KaiwuDRLDefine.MESSAGE_TRAIN:
                    td = message_value.get("train_data")
                    td_prioritized = message_value.get("train_data_prioritized")
                    if td is not None and td_prioritized is not None:
                        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
                            # 收集数据，批量发送减少flush次数
                            reverb_batch_data.extend(td)
                            reverb_batch_prioritized.extend(td_prioritized)

                        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
                            self.send_msg_use_zmq_client(td)

                        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
                            self.send_msg_use_shared_memory(td)

                        else:
                            pass

                elif message_type == KaiwuDRLDefine.MESSAGE_SAVE_MODEL:
                    self.msg_process_save_model(train_data)

                elif message_type == KaiwuDRLDefine.MESSAGE_PROCESS_STOP:
                    self.msg_process_process_stop(train_data)

                else:
                    self.logger.error(
                        f"learner_proxy recv un support message_type: {message_type}, please check",
                        g_not_server_label,
                    )

                total_processed += 1

        # 批量发送所有积累的reverb数据（关键性能优化：减少flush次数）
        if reverb_batch_data and CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            self.send_msg_use_reverb_client_batch(reverb_batch_data, reverb_batch_prioritized)

        return True  # 有数据处理

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True

        # 优雅退出：等待一段时间让进程自然结束
        self.join(timeout=5)

        # 如果5秒后还没结束，强制终止
        if self.is_alive():
            self.logger.warning(f"learner_proxy process still alive after 5s, terminating forcefully")
            self.terminate()
            self.join(timeout=2)

            # 如果terminate也不行，使用kill
            if self.is_alive():
                self.logger.error(f"learner_proxy process still alive after terminate, killing")
                self.kill()
                self.join(timeout=1)

        # 清理SharedMemory资源
        if (
            CONFIG.workflow_learner_proxy_communication
            == KaiwuDRLDefine.WORKFLOW_LEARNER_PROXY_COMMUNICATION_SHARED_MEMORY
        ):
            try:
                self.shm_data.close()
                self.shm_data.unlink()
                self.shm_prioritized.close()
                self.shm_prioritized.unlink()
                self.shm_metadata.close()
                self.shm_metadata.unlink()
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"learner_proxy cleanup shared memory error: {str(e)}")

        self.logger.info("learner_proxy LearnerProxy stop success", g_not_server_label)

    def after_run(self) -> bool:
        pass

    def run(self) -> None:
        if not self.before_run():
            self.logger.error("learner_proxy before_run failed", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                # 持续消费队列，只有没数据时才sleep
                has_data = self.run_once()

                if not has_data:
                    # 队列为空，短暂休眠避免CPU空转
                    time.sleep(CONFIG.idle_sleep_second)

            except Exception as e:
                self.logger.exception(
                    f"learner_proxy run error: {str(e)}",
                    g_not_server_label,
                )

    # 发送样本时, 可以对样本进行预处理操作
    def before_send_train_data(self, train_data, train_data_prioritized):
        if train_data is None or len(train_data) == 0:
            return

        # 暂时删除step维度
        if "s" in train_data.keys():
            del train_data["s"]

        # 增加lz4压缩
        # compress_train_data = lz4.block.compress(train_data, store_size=False)

    def before_send_train_data_simple(self, train_data, train_data_prioritized):
        """
        在发送样本开始时处理, 主要是压缩/解压缩, 主要是对train_data做检测, train_data_prioritized有些场景可能没有
        """
        if train_data is None or len(train_data) == 0:
            return None

        # 增加lz4压缩
        compress_train_data = compress_data(train_data)
        return compress_train_data

    def after_send_train_data_simple(self, train_data):
        """
        在发送样本后的处理, 主要是做统计
        """
        if train_data is None or len(train_data) == 0:
            return

        sample_size = 0
        if isinstance(train_data, np.ndarray):
            sample_size = train_data.nbytes
        elif isinstance(train_data, (list, tuple)):
            input_datas_list = train_data
            sample_size = 0
            for agent in input_datas_list:
                sample_size += agent.nbytes

            # 转换成MB
            sample_size = round(sample_size / (1024 * 1024), 2)

        # 更新最大样本大小
        if sample_size > self.max_sample_size:
            self.max_sample_size = sample_size

    def send_msg_use_zmq_client(self, train_data):
        """
        采用zmq_client发送请求, PUSH模式, 不等待回包
        """
        if train_data is None or len(train_data) == 0:
            return False

        train_data_size = len(train_data)

        try:
            data = self.before_send_train_data_simple(train_data, None)
            if data:
                self.zmq_client.send(data, binary=True)
                self.send_to_learner_succ_cnt += train_data_size
            else:
                self.send_to_learner_err_cnt += train_data_size

        except Exception as e:
            self.logger.exception(
                f"learner_proxy send to zmq_server {self.get_reverb_ip()} failed, "
                f"client_id is {self.client_id}, run error: {str(e)}, ",
                g_not_server_label,
            )
            self.send_to_learner_err_cnt += train_data_size

    # use reverb client send msg to reverb server (单条发送，已废弃)
    def send_msg_use_reverb_client(self, train_data, train_data_prioritized):
        if train_data is None or len(train_data) == 0:
            return

        # 发给reverb server, 没有进行样本发送前的处理是由于reverb暂时不支持lz4压缩/解压缩
        self.reverb_util.write_to_reverb_server_simple(self.reverb_table_names, train_data, train_data_prioritized)

        self.after_send_train_data_simple(train_data)

    # 批量发送reverb数据（减少flush次数，提升性能）
    def send_msg_use_reverb_client_batch(self, train_data_list, train_data_prioritized_list):
        if not train_data_list:
            return

        # 批量发送，在一个writer context内完成多个样本的append+flush
        self.reverb_util.write_to_reverb_server_simple(
            self.reverb_table_names, train_data_list, train_data_prioritized_list
        )

        # 统计（遍历原始数据，每个元素可能包含多个样本）
        for train_data in train_data_list:
            self.after_send_train_data_simple(train_data)

    # 注意，这里实际是在aisrv进程上的learner_proxy，而不是learner_proxy进程
    def get_training_metrics(self, msg):

        # 由于是多个进程可能调用, 故每个进程传递的client_id不一样的, client_id设置为self.client_id_for_learner + self.current_pid
        if self.zmq_client_for_learner is None:
            port = int(CONFIG.reverb_svr_port) - 2
            self.zmq_client_for_learner = ZmqClient(
                str(self.client_id_for_learner + self.current_pid),
                self.learner_addr,
                port,
                self.zmq_config,
            )
            self.zmq_client_for_learner.connect()

        message_type = msg.get(KaiwuDRLDefine.MESSAGE_TYPE)
        message_value = msg.get(KaiwuDRLDefine.MESSAGE_VALUE)

        if message_type == KaiwuDRLDefine.MESSAGE_GET_TRAINING_METRICS:
            self.zmq_client_for_learner.send(msg, binary=False)

            result = self.zmq_client_for_learner.recv(binary=False)
            training_metrics = result.get(KaiwuDRLDefine.MESSAGE_VALUE)
        return training_metrics
