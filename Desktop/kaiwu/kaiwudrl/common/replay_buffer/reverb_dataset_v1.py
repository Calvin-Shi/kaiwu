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
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
import threading
import time


class ReverbDataset(torch.utils.data.IterableDataset):
    """
    主动采用reverb_client从reverb_server读取数据
    """

    def __init__(self, logger):
        super().__init__()
        self._table_names = ["{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        # 双缓冲队列（线程安全）
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()

        # 主缓冲
        self.active_buffer = deque(maxlen=CONFIG.train_batch_size * CONFIG.replay_buffer_cache_multiplier)
        # 后台缓冲
        self.backup_buffer = deque(maxlen=CONFIG.train_batch_size * CONFIG.replay_buffer_cache_multiplier)

        # 线程停止信号
        self._stop_event = threading.Event()

        self.client = None

        # 后台填充线程
        self._fill_thread = None

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
            with self.write_lock:
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

        # 启动后台线程从reverb读取数据
        self.start_background_filler()

        while True:
            # 从主缓冲取数据
            with self.read_lock:
                if len(self.active_buffer) >= CONFIG.train_batch_size:
                    batch = [self.active_buffer.popleft() for _ in range(CONFIG.train_batch_size)]
                else:
                    batch = None

            if batch is None:
                time.sleep(CONFIG.idle_sleep_second)
                continue

            # 提取原始数据列表（假设每个sample[0].data[0]是单个元素）
            raw_batch = [sample[0].data[0] for sample in batch]

            # 批量处理代替逐元素循环
            batch_data = self._process_batch(raw_batch)

            if CONFIG.sample_data_return_data_type == KaiwuDRLDefine.SAMPLE_DATA_RETURN_DATA_TYPE_TENSOR:
                # 返回tensor（默认，去除冗余转换）
                yield batch_data.to(self.device)
            else:
                # 返回numpy
                if isinstance(batch_data, torch.Tensor):
                    yield batch_data.cpu().numpy()
                else:
                    yield batch_data

    def _process_batch(self, batch_data):
        """批量处理优化逻辑"""

        # 批量处理逻辑
        if isinstance(batch_data[0], torch.Tensor):
            # 使用torch.stack直接处理张量列表
            return torch.stack([x for x in batch_data])
        elif isinstance(batch_data[0], np.ndarray):
            # 合并numpy数组后统一转换
            np_batch = np.stack(batch_data)
            return torch.from_numpy(np_batch).float()
        else:
            raise TypeError(f"不支持的数据类型: {type(batch_data[0])}")

    def __del__(self):
        self._stop_event.set()
        if self._fill_thread is not None:
            self._fill_thread.join(timeout=5)

    def get_metrics(self):
        """获取性能指标"""
        return {"buffer_utilization": f"{len(self.active_buffer)}/{self.active_buffer.maxlen}"}
