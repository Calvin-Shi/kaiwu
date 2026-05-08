#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

LocalReplayBuffer - 通用本地经验回放缓冲区, 只是在单机单进程场景使用
采用字典存储，支持任意数据结构，完全由业务层定义
"""

import random
from typing import Dict, Any, List, Optional


class LocalReplayBuffer:
    """
    通用本地经验回放缓冲区

    设计理念：
    - 框架层不预设任何数据字段（obs/action/reward等）
    - 完全由业务层定义存储的数据结构
    - 使用字典存储，灵活通用
    - 支持循环缓冲区（满时覆盖最早数据）

    适用场景：
    - 任意强化学习算法的本地训练
    - 不同项目可以存储不同的数据字段
    - 支持字典、列表、张量等任意Python对象

    示例：
        # 不同项目可以存储不同的数据
        # 项目A存储: obs, action, reward
        buffer.add({"obs": [1,2,3], "action": 0, "reward": 1.0})

        # 项目B存储: state, hidden, done, info
        buffer.add({"state": np.array(...), "hidden": torch.tensor(...), "done": False, "info": {...}})
    """

    def __init__(self, capacity: int, logger, **kwargs):
        """
        初始化缓冲区

        Args:
            capacity: 缓冲区最大容量（最多存储多少条样本）
            logger: 日志句柄
            **kwargs: 其他可选参数（预留扩展）
        """
        self.capacity = capacity
        self._buffer: List[Dict[str, Any]] = []
        self._position = 0  # 当前写入位置（用于循环覆盖）
        self._is_full = False  # 是否已满
        self.logger = logger

    def add(self, sample: Dict[str, Any]) -> None:
        """
        添加一条样本到缓冲区

        Args:
            sample: 样本数据，字典格式，键值对完全由业务层定义
                例如: {
                    "obs": np.array(...),
                    "action": 0,
                    "reward": 1.0,
                    "done": False,
                    "next_obs": np.array(...),
                    # ... 任意其他字段
                }

        注意：
        - sample必须是字典类型
        - 字典的键值对完全由业务层定义，框架层不做任何假设
        - 当缓冲区满时，会覆盖最早的样本（循环缓冲区）
        """
        if not isinstance(sample, dict):
            raise TypeError(f"sample must be dict, got {type(sample)}")

        # 如果缓冲区未满，直接追加
        if len(self._buffer) < self.capacity:
            self._buffer.append(sample)
            self._position = len(self._buffer)
            if self._position >= self.capacity:
                self._is_full = True
        else:
            # 缓冲区已满，循环覆盖
            self._buffer[self._position] = sample
            self._position = (self._position + 1) % self.capacity

    def get(
        self,
        batch_size: Optional[int] = None,
        indices: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        从缓冲区获取样本

        Args:
            batch_size: 获取的样本数量
                - 如果为None且indices为None，返回所有样本
                - 如果指定，随机采样batch_size条样本
            indices: 指定获取的样本索引列表
                - 如果指定，则忽略batch_size参数
                - 索引范围: [0, len(buffer))

        Returns:
            samples: 样本列表，每个元素是字典
                [
                    {"obs": ..., "action": ..., "reward": ...},
                    {"obs": ..., "action": ..., "reward": ...},
                    ...
                ]

        Raises:
            ValueError: 如果缓冲区为空或参数非法

        示例：
            # 获取所有样本
            all_samples = buffer.get()

            # 随机采样32条
            batch = buffer.get(batch_size=32)

            # 获取指定索引的样本
            samples = buffer.get(indices=[0, 5, 10])
        """
        if len(self._buffer) == 0:
            raise ValueError("Buffer is empty. Call add() before get().")

        # 情况1: 指定索引获取
        if indices is not None:
            if not isinstance(indices, list):
                raise TypeError(f"indices must be list, got {type(indices)}")

            # 检查索引合法性
            max_idx = len(self._buffer) - 1
            for idx in indices:
                if idx < 0 or idx > max_idx:
                    raise ValueError(f"Index {idx} out of range [0, {max_idx}]")

            return [self._buffer[i] for i in indices]

        # 情况2: 获取所有样本
        if batch_size is None:
            return self._buffer.copy()

        # 情况3: 随机采样batch_size条
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive int, got {batch_size}")

        if batch_size > len(self._buffer):
            if self.logger:
                self.logger.warning(
                    f"batch_size ({batch_size}) > buffer size ({len(self._buffer)}), " f"returning all samples"
                )
            return self._buffer.copy()

        # 随机采样（不重复）
        sampled_indices = random.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[i] for i in sampled_indices]

    def clear(self) -> None:
        """
        清空缓冲区

        功能：
        - 清空所有样本
        - 重置写入位置
        - 重置满标志

        注意：
        - 不会改变容量配置
        - 清空后需要重新add样本
        """
        self._buffer.clear()
        self._position = 0
        self._is_full = False

    def __len__(self) -> int:
        """返回缓冲区当前样本数量"""
        return len(self._buffer)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """支持索引访问: buffer[0]"""
        return self._buffer[index]

    @property
    def is_full(self) -> bool:
        """缓冲区是否已满"""
        return self._is_full

    @property
    def size(self) -> int:
        """返回当前样本数量（等同于__len__）"""
        return len(self._buffer)

    def sample_indices(self, batch_size: int) -> List[int]:
        """
        随机采样索引（不返回样本本身）

        Args:
            batch_size: 采样数量

        Returns:
            indices: 索引列表

        用途：
        - 业务层可以先采样索引，再根据需要处理样本
        """
        if batch_size > len(self._buffer):
            batch_size = len(self._buffer)

        return random.sample(range(len(self._buffer)), batch_size)
