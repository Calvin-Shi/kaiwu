#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import random
import copy
import ctypes
from multiprocessing import Value, Array, Queue
import sys
import time
import numpy as np
from common_python.config.config_control import CONFIG


class MemBuffer(object):
    """
    random repeat get sample
    """

    def __init__(self, max_sample_num, logger):
        self._maxlen = int(max_sample_num)
        # 存储float32
        self._c_data_type = ctypes.c_float
        self._data_type = np.float32

        # max_sample_size: 预分配的最大样本大小
        # _sample_size: 实际使用的样本大小（第一次append时确定）
        self._max_sample_size = int(CONFIG.max_sample_size)
        self._sample_size = Value("i", -1)  # -1 表示未确定

        # 关键：在主进程就预分配足够大的共享内存
        self._data_queue = Array(self._c_data_type, max_sample_num * self._max_sample_size, lock=False)

        # 每个 slot 的状态锁, True 表示数据有效可读
        self._data_status = [Value(ctypes.c_bool, False, lock=True) for _ in range(max_sample_num)]

        self.next_idx = Value("i", 0)
        self._len = Value("i", 0)
        self.recv_samples = Value("i", 0)
        # 共享的起始样本数
        self.start_sample_num = Value("i", 0)
        # 共享的起始时间
        self.start_time = Value("d", time.time())
        self.last_speed = 0

        self.logger = logger

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

    def __len__(self):
        length = self._len.value
        return length

    def set_logger(self, logger):
        self.logger = logger

    def append(self, data):
        data_size = len(data)

        # 第一次append时，确定实际的sample_size
        if self._sample_size.value == -1:
            with self._sample_size.get_lock():
                if self._sample_size.value == -1:  # 双重检查
                    if data_size > self._max_sample_size:
                        if self.logger:
                            self.logger.error(
                                f"Sample size {data_size} exceeds max_sample_size {self._max_sample_size}!"
                            )
                        return
                    self._sample_size.value = data_size
                    if self.logger:
                        self.logger.info(f"Set actual sample_size={data_size} (max={self._max_sample_size})")

        sample_size = self._sample_size.value
        with self.next_idx.get_lock():
            idx = self.next_idx.value
            self.next_idx.value = (self.next_idx.value + 1) % self._maxlen

        with self._data_status[idx].get_lock():
            nparray = np.frombuffer(self._data_queue, dtype=self._data_type).reshape(
                self._maxlen, self._max_sample_size
            )
            nparray[idx][:sample_size] = data
            self._data_status[idx].value = True

        with self._len.get_lock():
            if self._len.value < self._maxlen:
                self._len.value += 1

        with self.recv_samples.get_lock():
            self.recv_samples.value += 1

    def _wait_ready(self, N):
        """等待共享内存准备就绪: sample_size已确定、预加载完成、样本数>=N"""
        error_index = 0
        while self._sample_size.value == -1:
            error_index += 1
            time.sleep(CONFIG.idle_sleep_second)
            if error_index % 10000 == 0:
                if self.logger:
                    self.logger.debug("Waiting for first sample to determine sample_size...")

        error_index = 0
        while self.__len__() < int(self._maxlen * CONFIG.preload_ratio):
            error_index += 1
            time.sleep(CONFIG.idle_sleep_second)
            if error_index % 10000 == 0:
                if self.logger:
                    self.logger.debug(
                        "The sample is less than half the capacity {} {}".format(self.__len__(), self._maxlen)
                    )

        while self._len.value < N:
            time.sleep(CONFIG.idle_sleep_second)
            if self.logger:
                self.logger.debug("sample_num < N {}".format(self._len.value))

    def get_numpy_samples(self, N):
        """
        一次性返回N条记录, 返回 numpy array (N, sample_size)
        不做 torch 转换, 适合预取线程使用
        逐条加锁读取, 确保不会读到正在被写入的脏数据
        """
        self._wait_ready(N)
        sample_size = self._sample_size.value
        nparray = np.frombuffer(self._data_queue, dtype=self._data_type).reshape(self._maxlen, self._max_sample_size)

        current_len = self.__len__()
        candidates = random.sample(range(current_len), min(N + N // 2, current_len))

        result = np.empty((N, sample_size), dtype=self._data_type)
        collected = 0
        for idx in candidates:
            if collected >= N:
                break
            with self._data_status[idx].get_lock():
                if self._data_status[idx].value:
                    result[collected] = nparray[idx, :sample_size]
                    collected += 1

        # 极端情况兜底: 候选不够, 无锁读取（概率极低, 仅保证不阻塞）
        if collected < N:
            fallback_indices = random.sample(range(current_len), N - collected)
            for idx in fallback_indices:
                result[collected] = nparray[idx, :sample_size].copy()
                collected += 1

        return result

    def get_samples(self, N):
        """
        一次性返回N条记录
        Return a batch tensor with shape (N, sample_size)
        返回形状为 (N, sample_size) 的批量张量
        """
        batch = self.get_numpy_samples(N)
        return torch.from_numpy(batch).to(self.device)

    def get_sample(self):
        """
        一次性返回1条记录, 加锁读取确保数据完整性
        """
        self._wait_ready(1)

        sample_size = self._sample_size.value
        nparray = np.frombuffer(self._data_queue, dtype=self._data_type).reshape(self._maxlen, self._max_sample_size)

        for _ in range(5):
            i = random.randint(0, self.__len__() - 1)
            if i < 0 or i >= self._maxlen:
                continue
            with self._data_status[i].get_lock():
                if self._data_status[i].value:
                    return torch.from_numpy(nparray[i][:sample_size].copy()).to(self.device)

        # 兜底: 无锁读取（极低概率走到这里）
        i = random.randint(0, self.__len__() - 1)
        return torch.from_numpy(nparray[i][:sample_size].copy()).to(self.device)

    def clear(self):
        with self._len.get_lock():
            self._len.value = 0
        with self.next_idx.get_lock():
            self.next_idx.value = 0
        with self.recv_samples.get_lock():
            self.recv_samples.value = 0

    def get_speed(self):
        # 加锁读取 recv_samples
        with self.recv_samples.get_lock():
            total_sample = self.recv_samples.value
            if total_sample < 0:
                self.recv_samples.value = 0
                return self.last_speed, 0

        # 加锁读取时间窗口基准
        with self.start_sample_num.get_lock(), self.start_time.get_lock():
            end_time = time.time()
            time_diff = end_time - self.start_time.value
            sample_diff = total_sample - self.start_sample_num.value

            # 仅在样本增加时计算速度并更新基准
            if sample_diff > 0 and time_diff > 1e-6:
                speed = int(sample_diff / time_diff)
                self.start_sample_num.value = total_sample
                self.start_time.value = end_time
                self.last_speed = speed
            else:
                # 无新样本时返回上一次速度
                speed = self.last_speed

        return speed, total_sample
