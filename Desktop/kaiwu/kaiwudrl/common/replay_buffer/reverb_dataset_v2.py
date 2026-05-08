#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from collections import deque
import torch
import numpy as np
import reverb
from common_python.config.config_control import CONFIG
import threading
import time


class ReverbDataset(torch.utils.data.IterableDataset):
    """
    主动采用 reverb_client 从 reverb_server 读取数据
    优化点：
      - 在 _process_batch 使用 np.stack / torch.stack 得到连续内存
      - 返回 pin_memory 的 CPU tensor(可选)，并在内部用一个 copy stream 预取到 GPU(异步)
    """

    def __init__(self, logger):
        super().__init__()
        self._table_names = ["{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]

        # device 判定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        # 双缓冲队列（线程安全，使用单锁保护 active/backup buffer 的读写和交换）
        self.buffer_lock = threading.Lock()

        # 主缓冲
        self.active_buffer = deque(maxlen=CONFIG.train_batch_size * CONFIG.replay_buffer_cache_multiplier)
        # 后台缓冲
        self.backup_buffer = deque(maxlen=CONFIG.train_batch_size * CONFIG.replay_buffer_cache_multiplier)

        # 线程停止信号
        self._stop_event = threading.Event()

        self.client = None

        # 后台填充线程
        self._fill_thread = None

        # 用于 Host->GPU 的异步拷贝 stream（仅当有 CUDA）
        self.copy_stream = None
        if self.device.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device=self.device)

        # pin_memory 支持开关：
        # 仅在有 CUDA 的机器上尝试 pin_memory；无 GPU 时直接禁用
        self._pin_memory_supported = False
        if self.device.type == "cuda":
            # 仅在有 CUDA 时才做一次轻量检测
            try:
                _ = torch.empty(1).pin_memory()
                self._pin_memory_supported = True
            except Exception as e:
                self._pin_memory_supported = False
                self.logger.warning(f"pin_memory 预检测失败，已禁用 pin 操作: {str(e)}")
        else:
            # 没有 GPU，则不尝试 pin
            self.logger.info("检测到无 GPU, pin_memory 已禁用")

    def start_background_filler(self):
        """启动后台填充线程"""
        if self._fill_thread is None:
            self._stop_event.clear()

            self.client = reverb.Client(f"localhost:{CONFIG.reverb_svr_port}")
            self._fill_thread = threading.Thread(target=self._fill_buffers_loop, daemon=True)
            self._fill_thread.start()
            self.logger.info(
                f"start_background_filler success, reverb.Client connect at localhost:{CONFIG.reverb_svr_port}"
            )

    def _fill_buffers_loop(self):
        """后台线程持续填充缓冲区"""
        while not self._stop_event.is_set():
            # 填充备用缓冲区
            self._fill_buffer(self.backup_buffer, target_fill=0.75)

            # 交换缓冲区（原子操作）
            with self.buffer_lock:
                if len(self.active_buffer) < CONFIG.train_batch_size:
                    self.active_buffer, self.backup_buffer = self.backup_buffer, self.active_buffer
                    self.backup_buffer.clear()

            # 避免高频切换
            time.sleep(CONFIG.idle_sleep_second)

    def _fill_buffer(self, target_buffer, target_fill):
        """填充指定缓冲区至目标大小"""
        while len(target_buffer) < target_buffer.maxlen * target_fill and not self._stop_event.is_set():
            try:
                remaining = int(target_buffer.maxlen - len(target_buffer))
                num_to_fetch = min(CONFIG.train_batch_size, remaining)

                if num_to_fetch <= 0:
                    break

                data = self.client.sample(
                    table=self._table_names[0],
                    num_samples=num_to_fetch,
                )
                if data:
                    target_buffer.extend(data)
                else:
                    time.sleep(CONFIG.idle_sleep_second)
            except Exception as e:
                self.logger.error(f"后台填充线程错误: {str(e)}")
                time.sleep(CONFIG.idle_sleep_second)

    def __iter__(self):
        # 启动后台线程从 reverb 读取数据
        self.start_background_filler()

        # prefetch_gpu: 已在 copy_stream 上发起异步拷贝、下一轮要产出的 GPU tensor
        prefetch_gpu = None

        while True:
            # 从主缓冲取数据
            with self.buffer_lock:
                if len(self.active_buffer) >= CONFIG.train_batch_size:
                    batch = [self.active_buffer.popleft() for _ in range(CONFIG.train_batch_size)]
                else:
                    batch = None

            if batch is None:
                # 缓冲区数据不足，如果已有预取的结果则先产出
                if prefetch_gpu is not None:
                    torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
                    yield prefetch_gpu
                    prefetch_gpu = None
                time.sleep(CONFIG.idle_sleep_second)
                continue

            # 提取原始数据列表（假设每个sample[0].data[0]是单个元素）
            raw_batch = [sample[0].data[0] for sample in batch]

            # 构建 CPU batch（并在可能时 pin 内存以支持异步拷贝）
            batch_cpu = self._process_batch(raw_batch, pin=True)

            # 如果没有 CUDA，则直接产出 CPU tensor
            if self.device.type != "cuda":
                yield batch_cpu
                continue

            # ---- CUDA 双缓冲流水线 ----
            if prefetch_gpu is not None:
                # 有上一轮预取的结果：等待其拷贝完成并产出
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
                yield prefetch_gpu

            # 在 copy_stream 上发起当前 batch 的异步 H2D 拷贝（预取）
            with torch.cuda.stream(self.copy_stream):
                prefetch_gpu = batch_cpu.to(self.device, non_blocking=True)

    def _process_batch(self, batch_data, pin=False):
        """支持多类型数据的批量处理优化
        参数:
          - batch_data: list of samples (np.ndarray / torch.Tensor)
          - pin: 是否尝试返回 pin_memory 的 CPU tensor(仅在有 GPU 时才尝试)
        返回:
          - 返回 CPU torch.Tensor(可pin, 若不可用则退回普通 CPU tensor)
        """
        if not batch_data:
            raise ValueError("输入批次数据为空")

        first_element = batch_data[0]

        def try_pin(tensor):
            """仅在确有 GPU 且 pin 被预先检测为可用时尝试 pin_memory"""
            if not (pin and self.device.type == "cuda" and self._pin_memory_supported):
                return tensor
            try:
                return tensor.pin_memory()
            except Exception as e:
                # 发生 pin 相关错误后，禁用后续 pin 操作并记录一次日志
                self._pin_memory_supported = False
                self.logger.warning(f"pin_memory 在运行时失败，已禁用后续 pin: {e}")

                # 返回未 pin 的 tensor
                return tensor

        # Torch tensor 分支
        if isinstance(first_element, torch.Tensor):
            cpu_tensors = []
            for t in batch_data:
                if not isinstance(t, torch.Tensor):
                    raise TypeError("混合类型：期望全是 torch.Tensor")
                tt = t.detach()
                if tt.device.type != "cpu":
                    tt = tt.cpu()
                cpu_tensors.append(tt)
            out = torch.stack(cpu_tensors, dim=0)
            if not out.is_contiguous():
                out = out.contiguous()
            if out.dtype != torch.float32:
                out = out.to(torch.float32)
            out = try_pin(out)
            return out

        # numpy 分支
        elif isinstance(first_element, np.ndarray):
            try:
                out_np = np.stack(batch_data, axis=0)
            except Exception:
                out_np = np.stack([np.asarray(x) for x in batch_data], axis=0)
            # 确保 numpy 数组内存连续，避免 from_numpy 报错
            if not out_np.flags["C_CONTIGUOUS"]:
                out_np = np.ascontiguousarray(out_np)
            tensor = torch.from_numpy(out_np)
            if tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)
            else:
                # from_numpy 返回的 tensor 与 numpy 共享内存，
                # 当 dtype 已是 float32 时 .to() 不会复制，
                # 需要 clone 使 tensor 拥有独立内存以支持 pin_memory
                tensor = tensor.clone()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            tensor = try_pin(tensor)
            return tensor

        else:
            raise TypeError(f"不支持的数据类型: {type(first_element)}")

    def __del__(self):
        self._stop_event.set()
        if self._fill_thread is not None:
            self._fill_thread.join(timeout=5)

    def get_metrics(self):
        """获取性能指标"""
        return {"buffer_utilization": f"{len(self.active_buffer)}/{self.active_buffer.maxlen}"}
