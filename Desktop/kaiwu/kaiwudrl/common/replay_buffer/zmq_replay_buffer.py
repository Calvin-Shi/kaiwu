#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
import ctypes
import threading
import time
import random
from collections import deque
from multiprocessing import Value, Array, Queue
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
from kaiwudrl.common.replay_buffer.replay_buffer_base import ReplayBufferBase


class BatchManager(object):
    """
    因为每条数据是写入到了共享内存里, 但是为了批量的返回数据, 这里采用了类来处理
    """

    def __init__(self, batch_size, sample_size, process_num, logger):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.process_num = process_num
        self.c_data_type = ctypes.c_float
        self.data_type = np.float32
        self.data = Array(self.c_data_type, process_num * 2 * sample_size * batch_size, lock=False)
        self.state = Array(ctypes.c_int, process_num * 2 + 1, lock=False)
        for index in range(len(self.state)):
            self.state[index] = 1
        self.last_get = Value("i", process_num * 2, lock=False)

        self.logger = logger

    def set_batch_sample(self, sample, batch_index):
        if not (isinstance(sample, np.ndarray) or sample.shape == (1, self.sample_size * self.batch_size)):
            self.logger.info(f"set_batch_sample error batch {sample.shape}")
            return False
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(self.process_num * 2, self.batch_size, self.sample_size)
        nparray[batch_index] = sample
        return True

    def set_one_sample(self, sample, batch_index, sample_index):
        if not (isinstance(sample, np.ndarray) or sample.shape == (1, self.sample_size)):
            self.logger.info(f"set_one_sample error sample {sample.shape}")
            return False
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(self.process_num * 2 * self.batch_size, self.sample_size)
        nparray[batch_index * self.batch_size + sample_index] = sample
        return True

    def get_batch_sample(self, batch_index):
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(self.process_num * 2 * self.batch_size, self.sample_size)
        value = nparray[batch_index * self.batch_size : batch_index * self.batch_size + self.batch_size]
        return value

    def set_state(self, index):
        self.state[self.last_get.value] = 1
        self.last_get.value = index

    def clear(self):
        for index in range(len(self.state)):
            self.state[index] = 1
        self.last_get.value = 2 * self.process_num


class BatchProcess(object):
    def __init__(self, batch_size, sample_size, process_num, logger):
        self.batch_size = batch_size
        self.process_num = process_num

        # batch_idx有效
        self.batch_queue = Queue()
        self.free_queue = Queue()
        self.logger = logger

        self.batch_manager = BatchManager(
            batch_size=batch_size,
            sample_size=sample_size,
            process_num=process_num,
            logger=self.logger,
        )
        self.pids = []
        self.last_get_index = None

    def __process_run(self, process_index, get_sample_func, full_queue, free_queue):
        self.logger.info("[BatchProcess::__process_run] process_index:{} pid:{}".format(process_index, os.getpid()))
        while True:
            batch_index = free_queue.get()
            for sample_index in range(self.batch_size):
                sample = get_sample_func()
                self.batch_manager.set_one_sample(sample, batch_index, sample_index)
            full_queue.put(batch_index)

    def process(self, get_data_func):
        for batch_index in range(self.process_num * 2):
            self.free_queue.put(batch_index)
        for process_index in range(self.process_num):
            pid = Process(
                target=self.__process_run,
                args=(
                    process_index,
                    get_data_func,
                    self.batch_queue,
                    self.free_queue,
                ),
            )
            pid.daemon = True
            pid.start()
            self.pids.append(pid)

    def get_batch_data(self):
        batch_index = self.batch_queue.get()
        sample = self.batch_manager.get_batch_sample(batch_index)
        return batch_index, sample

    def put_free_data(self, batch_index):
        self.free_queue.put(batch_index)

    def exit(self):
        for pid in self.pids:
            pid.join()


class ZmqReplayBuffer(ReplayBufferBase):
    def __init__(self, data_spec, logger, mem_buffer=None):

        # 参数定义
        capacity = CONFIG.replay_buffer_capacity

        self.batch_size = CONFIG.train_batch_size
        self.data_shapes = data_spec
        self.logger = logger

        """
        如果外部传入了 mem_buffer，则直接使用（跨进程共享场景）
        否则创建新的 mem_buffer（单进程场景）
        """
        if mem_buffer is not None:
            # 跨进程共享场景：直接使用传入的 mem_buffer
            self.mem_buffer = mem_buffer
            self.logger.info("ZmqReplayBuffer using shared mem_buffer from external source")
        else:
            # 单进程场景：创建新的 mem_buffer
            if CONFIG.reverb_rate_limiter == KaiwuDRLDefine.REVERB_RATE_LIMITER_SAMPLE_TO_INSERT_RATIO:
                from kaiwudrl.common.utils.mem_buffer_ratio import MemBuffer

                self.mem_buffer = MemBuffer(
                    capacity,
                    self.logger,
                    samples_per_insert=CONFIG.reverb_samples_per_insert,
                    error_buffer=CONFIG.reverb_error_buffer,
                )
            else:
                from kaiwudrl.common.utils.mem_buffer import MemBuffer

                self.mem_buffer = MemBuffer(capacity, self.logger)

            self.logger.info("ZmqReplayBuffer created new mem_buffer")

        self.last_batch_index = -1

        # 后台预取: 双缓冲队列 + 后台线程
        # 预取线程只做 numpy 操作(从共享内存批量读取), 不做 torch 转换, 确保 C 层面释放 GIL
        # 训练线程从缓冲区取 numpy array 后再做 torch.from_numpy().to(device)
        self._prefetch_buffer = deque(maxlen=2)
        self._prefetch_lock = threading.Lock()
        self._prefetch_event = threading.Event()
        self._stop_event = threading.Event()
        self._prefetch_thread = None

        # 设备信息, 用于训练线程做 tensor 转换
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _start_prefetch_thread(self):
        """启动后台预取线程（首次调用 next_by_batch_size 时触发）"""
        if self._prefetch_thread is None:
            self._stop_event.clear()
            self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self._prefetch_thread.start()
            self.logger.info("ZmqReplayBuffer prefetch thread started")

    def _get_numpy_samples(self, N):
        """
        从共享内存中批量读取 N 条样本, 只返回 numpy array (不做 torch 转换)
        直接调用 mem_buffer / mem_buffer_ratio 的 get_numpy_samples 方法
        """
        return self.mem_buffer.get_numpy_samples(N)

    def _prefetch_loop(self):
        """后台线程: 持续预取 batch 到缓冲队列 (只做 numpy 操作, 不持 GIL)"""
        while not self._stop_event.is_set():
            # 缓冲区未满时预取
            with self._prefetch_lock:
                buffer_full = len(self._prefetch_buffer) >= self._prefetch_buffer.maxlen
            if buffer_full:
                self._prefetch_event.wait(timeout=0.01)
                self._prefetch_event.clear()
                continue

            try:
                # 直接从共享内存读 numpy array, 不经过 mem_buffer.get_samples 避免 torch 操作
                batch_np = self._get_numpy_samples(self.batch_size)
                if batch_np is None:
                    time.sleep(CONFIG.idle_sleep_second)
                    continue
                with self._prefetch_lock:
                    self._prefetch_buffer.append(batch_np)
            except Exception as e:
                self.logger.error(f"ZmqReplayBuffer prefetch error: {e}")
                time.sleep(CONFIG.idle_sleep_second)

    def add_sample(self, datas):
        if not datas:
            return False

        for data in datas:
            self.mem_buffer.append(data)

        return True

    def init(self):
        pass

        # 从MemBuff里获取样本
        # self.batch_process.process(self.mem_buffer.get_sample)

    def get_next_batch(self):
        """
        获取下一批的数据记录
        """
        batch_index, sample_buff = self.batch_process.get_batch_data()
        if self.last_batch_index >= 0:
            self.batch_process.put_free_data(self.last_batch_index)
        self.last_batch_index = batch_index

        return sample_buff

    def next(self):
        """
        获取下一条数据记录, 采用预先批处理
        """

        return torch.from_numpy(self.get_next_batch())

    def next_by_batch_size(self):
        """
        获取下一条数据记录, 采用后台预取+双缓冲方式
        后台线程异步预取 numpy batch, 训练线程取到后做 torch 转换
        """
        # 启动后台预取线程（首次调用时）
        self._start_prefetch_thread()

        # 从预取缓冲区获取 numpy batch
        while True:
            with self._prefetch_lock:
                if len(self._prefetch_buffer) > 0:
                    batch_np = self._prefetch_buffer.popleft()
                    # 通知后台线程可以继续预取
                    self._prefetch_event.set()

                    # 防御性检查: 跳过 None (采样超时等异常情况)
                    if batch_np is None:
                        continue

                    # 训练线程做 torch 转换 (这里才用到 torch)
                    return torch.from_numpy(batch_np).to(self._device)

            # 缓冲区为空, 短暂等待后重试
            time.sleep(CONFIG.idle_sleep_second)

    def next_by_for(self):
        """
        获取下一条数据记录, 采用for循环方式
        """
        batch_samples = []
        for _ in range(CONFIG.train_batch_size):
            sample = self.mem_buffer.get_sample()
            batch_samples.append(sample)

        return batch_samples

    def get_insert_speed(self):
        speed, add_sample_count = self.mem_buffer.get_speed()
        return speed, add_sample_count

    def total_size(self):
        speed, add_sample_count = self.mem_buffer.get_speed()
        return add_sample_count

    def __del__(self):
        self._stop_event.set()
        self._prefetch_event.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5)
