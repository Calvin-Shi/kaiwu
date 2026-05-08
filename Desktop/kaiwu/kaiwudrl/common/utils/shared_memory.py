#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import ctypes
from multiprocessing import sharedctypes
import os
import sys
import re
import contextlib
import tempfile
import numpy as np
import struct
import fcntl
import time
import torch
from typing import Union
from multiprocessing import shared_memory


# 共享内存库, 采用sharedctypes
class SharedMemory(object):
    def __init__(self, type, size) -> None:

        self.size = size
        self.array = sharedctypes.RawArray(type, self.size)

    def check_idx_valid(self, idx):
        if 0 < idx or idx > self.size:
            return False

        return True

    def put(self, idx, data):
        if not self.check_idx_valid(idx) or not data:
            return

        self.array[idx] = data

    def get(self, idx):
        if not self.check_idx_valid(idx):
            return

        return self.array[idx]

    def size(self):
        return self.size


# 扩展数据类型映射, 支持numpy和tensor数据类型
DTYPE_MAP = {
    # 基础类型（显式包含小端和大端）
    np.dtype(np.float32).newbyteorder(">"): 0x01,
    np.dtype(np.float32).newbyteorder("<"): 0x01,  # 统一映射
    np.dtype(np.int32).newbyteorder(">"): 0x02,
    np.dtype(np.int32).newbyteorder("<"): 0x02,
    np.dtype(np.uint8): 0x03,  # 无字节序类型
    # 添加常见类型
    np.dtype(np.float64): 0x04,
    np.dtype(np.int64): 0x05,
    np.dtype(np.bool_): 0x06,
}

INV_DTYPE_MAP = {
    0x01: np.dtype(np.float32),
    0x02: np.dtype(np.int32),
    0x03: np.dtype(np.uint8),
    0x04: np.dtype(np.float64),
    0x05: np.dtype(np.int64),
    0x06: np.dtype(np.bool_),
}


class SharedMemoryExtend:
    """
    最终内存布局：
    全局头信息如下:
    | 总Tensor数 (2B) | [Tensor1元数据 | Tensor1数据] | [Tensor2元数据 | Tensor2数据] | ... |
    每个Tensor段的详细结构:
    | dtype (1B) | ndim (1B) | 维度1 (4B) | 维度2 (4B) | ... | 数据区 (变长) |
    """

    def __init__(self):
        self.lock_dir = os.path.join(tempfile.gettempdir(), "shm_locks")
        os.makedirs(self.lock_dir, exist_ok=True)

        # 避免每次新申请共享内存
        self.shm_pool = {}

        self.read_success_count = 0
        self.write_success_count = 0

    def _get_lock_path(self, shm_name: str) -> str:
        """生成合法锁文件路径"""
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", shm_name)
        return os.path.join(self.lock_dir, f"shm_lock_{safe_name}")

    def _acquire_lock(self, shm_name: str, mode: str = "exclusive"):
        """mode: 'exclusive'(默认) 或 'shared'"""
        lock_type = fcntl.LOCK_EX if mode == "exclusive" else fcntl.LOCK_SH
        lock_path = self._get_lock_path(shm_name)

        class LockContext:
            # 动态选择锁类型
            def __enter__(self_):
                self_.lock_file = open(lock_path, "w")
                fcntl.flock(self_.lock_file, lock_type)
                return self_

            def __exit__(self_, *args):
                fcntl.flock(self_.lock_file, fcntl.LOCK_UN)
                self_.lock_file.close()

        return LockContext()

    def write_dynamic_tensors(self, shm_name: str, tensors: list):
        # 维度校验
        assert all(t.ndim == 2 for t in tensors), "输入必须为二维tensor"
        base_shape = tensors[0].shape
        assert all(t.shape == base_shape for t in tensors), "所有张量形状必须一致"

        # 优化内存传输
        cpu_tensors = []
        for t in tensors:
            if t.is_cuda:
                cpu_t = t.cpu().pin_memory() if t.is_pinned() else t.cpu()
            else:
                cpu_t = t.detach()
            cpu_tensors.append(cpu_t)

        # 计算元数据
        combined_shape = (len(tensors), *base_shape)
        total_elements = np.prod(combined_shape)
        dtype = cpu_tensors[0].numpy().dtype
        header = struct.pack("!HBB3I", 1, DTYPE_MAP[dtype], 3, *combined_shape)
        header_size = len(header)

        # 关键修正点：精确计算数据尺寸
        data_bytes = total_elements * dtype.itemsize
        # 保持对齐
        total_size = ((header_size + data_bytes + 63) // 64) * 64

        with self._acquire_lock(shm_name, mode="exclusive"):
            # 内存管理
            shm = self.shm_pool.get(shm_name)
            if shm and shm.size >= total_size:
                buffer = memoryview(shm.buf)
            else:
                if shm:
                    shm.close()
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_size)
                self.shm_pool[shm_name] = shm
                buffer = memoryview(shm.buf)

            # 写入元数据
            buffer[0:header_size] = header

            # 关键修正点：添加count参数
            dest = np.frombuffer(buffer, dtype=dtype, offset=header_size, count=total_elements).reshape(combined_shape)

            # 数据拷贝
            for i, t in enumerate(cpu_tensors):
                src_np = t.numpy()
                # 自动进行形状校验
                dest[i] = src_np

            self.write_success_count += 1

    def read_dynamic_tensors(self, shm_name: str) -> list:
        with self._acquire_lock(shm_name, mode="shared"):
            shm = None
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
            except FileNotFoundError:
                time.sleep(0.05)
                return None

            try:
                with contextlib.ExitStack() as stack:
                    buffer = memoryview(shm.buf)
                    stack.callback(buffer.release)

                    # 元数据解析
                    header = bytes(buffer[:16])
                    num_tensors, dtype_code, ndim, *shape = struct.unpack("!HBB3I", header)
                    shape = tuple(shape)

                    if ndim != 3 or num_tensors != 1:
                        raise ValueError("Invalid tensor dimensions")

                    dtype = INV_DTYPE_MAP[dtype_code]
                    element_count = shape[0] * shape[1] * shape[2]
                    data_size = element_count * dtype.itemsize

                    # 直接创建numpy数组并拷贝
                    np_array = np.ndarray(shape=shape, dtype=dtype, buffer=buffer, offset=16).copy()  # 关键优化点1

                # 转换为PyTorch张量
                if torch.cuda.is_available():
                    tensor = torch.as_tensor(np_array, device="cuda")  # 关键优化点2
                else:
                    tensor = torch.from_numpy(np_array)

                # 批次分割优化
                batch_tensors = list(tensor.unbind(0))  # 关键优化点3

                self.read_success_count += 1
                return [batch_tensors]

            finally:
                if shm:
                    shm.close()
                    if hasattr(self, "_shm_cleanup_list"):
                        self._shm_cleanup_list.append(shm)
                    else:
                        self._shm_cleanup_list = [shm]

    def cleanup_shm(self):
        for name, shm in self.shm_pool.items():
            shm.close()
            shm.unlink()
        self.shm_pool.clear()

    def get_monitor_info(self):
        return {
            "read_success_count": self.read_success_count,
            "write_success_count": self.write_success_count,
        }
