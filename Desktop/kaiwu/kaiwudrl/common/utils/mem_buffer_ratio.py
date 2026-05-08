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
from multiprocessing import Value, Array, Lock
import sys
import time
import numpy as np
from common_python.config.config_control import CONFIG


class MemBuffer(object):
    """
    支持SampleToInsertRatio样本消耗比控制的内存缓冲区
    """

    def __init__(self, max_sample_num, logger, samples_per_insert=10, error_buffer=10.0):
        self._maxlen = int(max_sample_num)
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
        self.recv_samples = Value("i", 0)  # 总插入数（累计）
        self.sampled_count = Value("i", 0)  # 总采样数（累计）
        self.start_sample_num = Value("i", 0)
        self.start_time = Value("d", time.time())
        self.last_speed = 0
        self.logger = logger

        # 速率限制器参数（严格按照 reverb SampleToInsertRatio 语义实现）
        # reverb 原版两阶段:
        #   Stage 1: len < min_size_to_sample 时，允许插入，阻塞采样
        #   Stage 2: len >= min_size_to_sample 后，按比例动态平衡
        #     error = (inserts - min_size) * samples_per_insert - samples
        #     允许采样条件: error >= -error_buffer
        #     允许插入条件: error <= error_buffer
        self.samples_per_insert = samples_per_insert
        self.error_buffer = float(error_buffer) if isinstance(error_buffer, (float, int)) else error_buffer
        if self.logger:
            self.logger.info(f"ratio {CONFIG.preload_ratio},  self._maxlen { self._maxlen}")
        self.min_size_to_sample = self._maxlen * CONFIG.preload_ratio
        # 校验参数合法性
        self._validate_rate_limiter_params()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 多进程锁（确保插入/采样计数操作原子性）
        self.rate_lock = Lock()

    def _validate_rate_limiter_params(self):
        """校验速率限制器参数合法性"""
        if self.samples_per_insert <= 0:
            raise ValueError(f"samples_per_insert必须>0，实际为{self.samples_per_insert}")
        if self.min_size_to_sample < 1:
            raise ValueError(f"min_size_to_sample必须≥1，实际为{self.min_size_to_sample}")
        if self.error_buffer < max(1.0, self.samples_per_insert):
            raise ValueError(f"error_buffer过小({self.error_buffer})，至少需要{max(1.0, self.samples_per_insert)}")

    def __len__(self):
        return self._len.value

    def _can_insert(self, num_inserts=1):
        """
        判断是否允许插入操作
        reverb 原版实现:
          - Python 层: offset = samples_per_insert * min_size_to_sample
                       max_diff = offset + error_buffer
          - C++ 层:    diff = inserts * spi - samples
                       CanInsert: diff <= max_diff
          - 展开:      inserts * spi - samples <= spi * min_size + error_buffer
                    => (inserts - min_size) * spi - samples <= error_buffer
        """
        with self.rate_lock:
            current_inserts = self.recv_samples.value
            current_samples = self.sampled_count.value
            current_len = self.__len__()

            # Stage 1: 若插入后仍未达到最小采样尺寸，允许插入
            if current_len + num_inserts <= self.min_size_to_sample:
                return True

            # Stage 2: 计算插入后的 error (含 offset)
            new_inserts = current_inserts + num_inserts
            error = (new_inserts - self.min_size_to_sample) * self.samples_per_insert - current_samples
            return error <= self.error_buffer

    def _can_sample(self, num_samples=1):
        """
        判断是否允许采样操作
        reverb 原版实现:
          - Python 层: offset = samples_per_insert * min_size_to_sample
                       min_diff = offset - error_buffer
          - C++ 层:    diff = inserts * spi - samples
                       CanSample: diff >= min_diff
          - 展开:      inserts * spi - samples >= spi * min_size - error_buffer
                    => (inserts - min_size) * spi - samples >= -error_buffer
        """
        with self.rate_lock:
            current_inserts = self.recv_samples.value
            current_samples = self.sampled_count.value
            current_len = self.__len__()

            # Stage 1: 缓冲区大小不足，禁止采样
            if current_len < self.min_size_to_sample:
                return False

            # Stage 2: 计算采样后的 error (含 offset)
            new_samples = current_samples + num_samples
            error = (current_inserts - self.min_size_to_sample) * self.samples_per_insert - new_samples
            return error >= -self.error_buffer

    def _max_sampleable(self):
        """
        计算当前允许采样的最大条数（不加锁版本, 调用方需在 rate_lock 内使用）
        reverb 原版公式（展开含 offset）:
          error_after = (inserts - min_size) * spi - (samples + N) >= -error_buffer
          解出 N <= (inserts - min_size) * spi - samples + error_buffer
        """
        current_inserts = self.recv_samples.value
        current_samples = self.sampled_count.value
        current_len = self.__len__()

        if current_len < self.min_size_to_sample:
            return 0

        max_n = int(
            (current_inserts - self.min_size_to_sample) * self.samples_per_insert - current_samples + self.error_buffer
        )
        return max(0, max_n)

    def append(self, data, timeout=5.0):
        """插入数据（带速率限制）"""
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
                        return False
                    self._sample_size.value = data_size
                    if self.logger:
                        self.logger.info(f"Set actual sample_size={data_size} (max={self._max_sample_size})")

        sample_size = self._sample_size.value
        if data_size != sample_size:
            if self.logger:
                self.logger.error(f"Sample size mismatch: expected {sample_size}, got {data_size}")
            return False

        start_time = time.time()
        # 等待插入权限（超时退出）
        while not self._can_insert():
            if time.time() - start_time > timeout:
                if self.logger:
                    self.logger.debug("插入操作超时，缓冲区可能已满或采样不足")
                return False
            time.sleep(CONFIG.idle_sleep_second)

        # 执行插入操作
        with self.next_idx.get_lock():
            idx = self.next_idx.value
            self.next_idx.value = (self.next_idx.value + 1) % self._maxlen

        # 写入数据（加锁, 确保读取端不会读到半写入的数据）
        with self._data_status[idx].get_lock():
            nparray = np.frombuffer(self._data_queue, dtype=self._data_type).reshape(
                self._maxlen, self._max_sample_size
            )
            nparray[idx][:sample_size] = data
            self._data_status[idx].value = True

        # 对齐 reverb 的 Insert 逻辑
        with self._len.get_lock():
            if self._len.value < self._maxlen:
                self._len.value += 1

        with self.recv_samples.get_lock():
            self.recv_samples.value += 1

        return True

    def _collect_samples(self, N, timeout=200.0):
        """
        核心采样逻辑: 从共享内存批量读取 N 条样本（带速率限制）
        直接计算可采样数量, 避免试探式减半浪费时间
        返回 numpy array 列表和总采集数, 供 get_samples / get_numpy_samples 复用
        """
        if N <= 0:
            return [], 0

        # 等待第一个样本被写入（确定sample_size）
        while self._sample_size.value == -1:
            time.sleep(CONFIG.idle_sleep_second)

        # 等待预加载完成
        while self.__len__() < self.min_size_to_sample:
            time.sleep(CONFIG.idle_sleep_second)

        total_collected = 0
        collected_np = []
        start_time = time.time()

        # 预先获取不变量, 避免循环内重复访问
        sample_size = self._sample_size.value
        nparray = np.frombuffer(self._data_queue, dtype=self._data_type).reshape(self._maxlen, self._max_sample_size)

        while total_collected < N:
            remaining = N - total_collected

            # 在 rate_lock 内一次性完成: 计算配额 + 确认 + 扣减计数, 消除竞态窗口
            batch_size = 0
            with self.rate_lock:
                available = self._max_sampleable()
                if available > 0:
                    batch_size = min(remaining, available, self.__len__())
                    if batch_size > 0:
                        # 在锁内直接扣减 sampled_count, 保证计算和扣减的原子性
                        self.sampled_count.value += batch_size

            if batch_size <= 0:
                if time.time() - start_time > timeout:
                    if self.logger:
                        self.logger.debug(f"采样超时（总耗时{time.time()-start_time:.2f}s），已采集{total_collected}/{N}个")
                    return collected_np, total_collected
                time.sleep(CONFIG.idle_sleep_second)
                continue

            # 逐条加锁读取, 确保不会读到正在被写入的脏数据
            current_len = self.__len__()
            actual_batch = min(batch_size, current_len)
            candidates = random.sample(range(current_len), min(actual_batch + actual_batch // 2, current_len))

            batch_result = np.empty((actual_batch, sample_size), dtype=self._data_type)
            collected_in_batch = 0
            for idx in candidates:
                if collected_in_batch >= actual_batch:
                    break
                with self._data_status[idx].get_lock():
                    if self._data_status[idx].value:
                        batch_result[collected_in_batch] = nparray[idx, :sample_size]
                        collected_in_batch += 1

            if collected_in_batch > 0:
                collected_np.append(batch_result[:collected_in_batch])
                total_collected += collected_in_batch

            if self.logger:
                self.logger.debug(f"已采集{total_collected}/{N}个样本（本次批次大小：{actual_batch}）")

        return collected_np, total_collected

    def get_numpy_samples(self, N, timeout=200.0):
        """
        批量获取样本, 返回 numpy array (N, sample_size)
        带速率限制和动态批次调整, 不做 torch 转换, 适合预取线程使用
        超时时返回已采集到的部分数据, 一条都没采到则返回 None
        """
        collected_np, total_collected = self._collect_samples(N, timeout)
        if not collected_np:
            if self.logger:
                self.logger.warning(f"get_numpy_samples 未采集到任何样本 (请求{N}条, 超时{timeout}s)")
            return None
        if total_collected < N and self.logger:
            self.logger.warning(f"get_numpy_samples 采样不足: 请求{N}条, 实际采集{total_collected}条")
        if len(collected_np) == 1:
            return collected_np[0]
        return np.concatenate(collected_np, axis=0)

    def get_samples(self, N, timeout=200.0):
        """
        批量获取样本（带动态批次调整，避免因批次过大导致阻塞）
        Return a stacked batch tensor with shape (N, sample_size)
        返回形状为 (N, sample_size) 的堆叠批量张量
        """
        collected_np, total_collected = self._collect_samples(N, timeout)
        if not collected_np:
            return None
        batch_np = np.concatenate(collected_np, axis=0) if len(collected_np) > 1 else collected_np[0]
        return torch.from_numpy(batch_np).to(self.device)

    def get_sample(self, timeout=5.0):
        """
        获取单个样本（带速率限制）
        Returns a single tensor sample or None
        返回单个张量样本或 None
        """
        samples = self.get_samples(1, timeout)
        if samples is not None and samples.shape[0] > 0:
            return samples[0]
        return None

    def clear(self):
        """清空缓冲区并重置计数"""
        with self._len.get_lock():
            self._len.value = 0
        with self.next_idx.get_lock():
            self.next_idx.value = 0
        with self.recv_samples.get_lock():
            self.recv_samples.value = 0
        with self.sampled_count.get_lock():
            self.sampled_count.value = 0

    def get_speed(self):
        """计算插入速度（保持原有逻辑）"""
        with self.recv_samples.get_lock():
            total_sample = self.recv_samples.value
            if total_sample < 0:
                self.recv_samples.value = 0
                return self.last_speed, 0

        with self.start_sample_num.get_lock(), self.start_time.get_lock():
            end_time = time.time()
            time_diff = end_time - self.start_time.value
            sample_diff = total_sample - self.start_sample_num.value

            if sample_diff > 0 and time_diff > 1e-6:
                speed = int(sample_diff / time_diff)
                self.start_sample_num.value = total_sample
                self.start_time.value = end_time
                self.last_speed = speed
            else:
                speed = self.last_speed

        return speed, total_sample
