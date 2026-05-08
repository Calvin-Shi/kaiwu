#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import re
import time
import posix_ipc
import msgpack
import msgpack_numpy as m

m.patch()
from typing import Optional, Dict, Set
from common_python.ipc.duplex_posix_channel import SemaphoreSharedMemoryChannel
from common_python.config.config_control import CONFIG


class SharedMemoryReplayBuffer:
    """
    动态发现跨容器共享内存通道
    - Producer按照自己的唯一标识（如IP、容器名）创建channel
    - Consumer动态扫描并连接所有可用的channels
    """

    def __init__(
        self,
        base_name: str = "kaiwudrl_shared_memory",
        # Producer配置
        is_producer: bool = True,
        producer_identifier: Optional[str] = None,  # 如 "192.168.1.100" 或容器名
        # Consumer配置
        discovery_interval: float = 10.0,  # 定期扫描新channel的间隔（秒）
        # Channel配置
        size: int = 4 * 1024 * 1024,
        create: bool = True,
        unlink_on_close: bool = False,
        logger=None,
    ):
        """
        Producer模式：
            producer_identifier: 唯一标识（IP地址、容器名、UUID等）

        Consumer模式：
            自动发现所有匹配base_name的channels
        """
        self.base_name = base_name
        self.is_producer = is_producer
        self.discovery_interval = discovery_interval
        self.size = size
        self.create = create
        self.unlink_on_close = unlink_on_close
        self.logger = logger

        if is_producer:
            if not producer_identifier:
                raise ValueError("Producer must provide producer_identifier")

            # 标准化标识符：去除特殊字符，只保留字母数字和下划线
            self.producer_id = self._sanitize_identifier(producer_identifier)
            self.channel_name = f"{base_name}_{self.producer_id}"

            self.channel = SemaphoreSharedMemoryChannel(
                name_prefix=self.channel_name, size=size, create=create, unlink_on_close=unlink_on_close
            )

            self.logger.info(f"Producer created: identifier={producer_identifier}, " f"channel={self.channel_name}")

        else:
            # Consumer: 动态发现channels
            self.channels: Dict[str, SemaphoreSharedMemoryChannel] = {}
            self.last_discovery_time = 0

            # 立即执行一次发现
            self._discover_channels()

            self.logger.info(f"Consumer initialized: discovered {len(self.channels)} channels")

        # 设置跨进程计数器
        self._setup_shared_counters(base_name)

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        """标准化标识符：IP地址或容器名转换为合法的POSIX名称"""
        # 替换点号、冒号、短横线等为下划线
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", identifier)
        return sanitized

    def _discover_channels(self) -> Set[str]:
        """
        扫描/dev/shm目录，发现所有匹配的channel
        返回新发现的channel标识符集合
        """
        if self.is_producer:
            return set()

        new_channels = set()

        try:
            # 扫描/dev/shm目录
            shm_dir = "/dev/shm"
            if not os.path.exists(shm_dir):
                self.logger.info(f"Warning: {shm_dir} not found")
                return new_channels

            # 查找匹配的共享内存文件
            # POSIX共享内存文件命名规则：sem.{name} 或直接 {name}
            pattern = re.compile(rf"^{re.escape(self.base_name)}_([a-zA-Z0-9_]+)_shm$")

            for filename in os.listdir(shm_dir):
                match = pattern.match(filename)
                if match:
                    producer_id = match.group(1)

                    # 如果已经连接，跳过
                    if producer_id in self.channels:
                        continue

                    # 尝试连接该channel
                    try:
                        channel_name = f"{self.base_name}_{producer_id}"
                        ch = SemaphoreSharedMemoryChannel(
                            name_prefix=channel_name,
                            size=self.size,
                            create=False,  # Consumer不创建，只打开已存在的
                            unlink_on_close=False,
                        )

                        self.channels[producer_id] = ch
                        new_channels.add(producer_id)
                        self.logger.info(f"Discovered new channel: {producer_id}")

                    except posix_ipc.ExistentialError:
                        # 文件存在但信号量可能还未创建，跳过
                        pass
                    except Exception as e:
                        self.logger.error(f"Failed to connect to channel {producer_id}: {e}")

        except Exception as e:
            self.logger.error(f"Error during channel discovery: {e}")

        return new_channels

    def _maybe_discover(self, force: bool = False):
        """定期扫描新channels"""
        if self.is_producer:
            return

        current_time = time.time()
        if force or (current_time - self.last_discovery_time) >= self.discovery_interval:
            new_channels = self._discover_channels()
            if new_channels:
                self.logger.info(f"Discovered {len(new_channels)} new channels: {new_channels}")
            self.last_discovery_time = current_time

    # ============ Producer方法 ============
    def send(self, data, timeout: Optional[float] = None, await_ack: bool = False) -> None:
        """Producer发送数据"""
        if not self.is_producer:
            raise RuntimeError("Only producer can send")

        data_size = len(data)
        data_bytes = msgpack.packb(data, use_bin_type=True)
        self.channel.send(data_bytes, timeout=timeout, await_ack=await_ack)
        self._increment_insert_count(data_size)

    # 高性能使用try_send
    def try_send(self, data, timeout: float = 0.001) -> bool:
        """尝试发送，失败返回False"""
        if not self.is_producer:
            return False

        try:
            data_size = len(data)
            data_bytes = msgpack.packb(data, use_bin_type=True)
            self.channel.send(data_bytes, timeout=timeout, await_ack=False)
            self._increment_insert_count(data_size)
            return True
        except (TimeoutError, Exception):
            return False

    # ============ Consumer方法 ============
    def recv_batch(
        self,
        batch_size: int = 256,
        timeout: float = 0.0001,
        strategy: str = "round_robin",
        max_rounds: int = 3,
        auto_discover: bool = True,
    ):
        """
        Consumer批量接收数据

        Returns a stacked batch tensor with shape (batch_size, feature_dim)
        返回形状为 (batch_size, feature_dim) 的堆叠批量张量

        Args:
            auto_discover: 是否在接收前自动发现新channels
        """
        if self.is_producer:
            raise RuntimeError("Only consumer can receive")

        # 定期发现新channels
        if auto_discover:
            self._maybe_discover()

        # 如果没有任何channel，强制扫描一次
        if not self.channels:
            self._maybe_discover(force=True)
            if not self.channels:
                return []

        results = []
        rounds = 0
        received_in_round = False

        while len(results) < batch_size and rounds < max_rounds:
            received_in_round = False

            if strategy == "round_robin":
                for producer_id, ch in list(self.channels.items()):
                    if len(results) >= batch_size:
                        break
                    try:
                        data_bytes = ch.recv(timeout=timeout)
                        data = msgpack.unpackb(data_bytes, raw=False)
                        results.append(data)
                        received_in_round = True
                    except TimeoutError:
                        continue
                    except Exception as e:
                        # 可能channel已失效，记录并考虑移除
                        if rounds == 0:
                            self.logger.error(f"Error reading from producer {producer_id}: {e}")
                        continue

            elif strategy == "random":
                import random

                producer_ids = list(self.channels.keys())
                random.shuffle(producer_ids)

                for producer_id in producer_ids:
                    if len(results) >= batch_size:
                        break
                    try:
                        ch = self.channels[producer_id]
                        data_bytes = ch.recv(timeout=timeout)
                        data = msgpack.unpackb(data_bytes, raw=False)
                        results.append(data)
                        received_in_round = True
                    except TimeoutError:
                        continue
                    except Exception as e:
                        if rounds == 0:
                            self.logger.error(f"Error reading from producer {producer_id}: {e}")
                        continue

            rounds += 1
            if not received_in_round and rounds > 1:
                break

        self._increment_sample_count()

        # Stack list of data into a single batch tensor
        # 将数据列表堆叠成单个批量张量
        if results and len(results) > 0:
            import torch
            import numpy as np

            # Convert to tensors if needed
            # 如果需要，转换为张量
            tensor_list = []
            for data in results:
                if isinstance(data, torch.Tensor):
                    tensor_list.append(data)
                elif isinstance(data, np.ndarray):
                    tensor_list.append(torch.from_numpy(data).float())
                elif isinstance(data, (list, tuple)):
                    tensor_list.append(torch.tensor(data, dtype=torch.float32))
                else:
                    # Try to convert to tensor
                    # 尝试转换为张量
                    tensor_list.append(torch.tensor(data, dtype=torch.float32))

            # Stack into batch tensor (batch_size, feature_dim)
            # 堆叠成批量张量 (batch_size, feature_dim)
            return torch.stack(tensor_list, dim=0)

        return results

    def force_discover(self) -> int:
        """强制立即扫描新channels，返回新发现的数量"""
        new_channels = self._discover_channels()
        return len(new_channels)

    def get_monitor_info(self) -> dict:
        """获取统计信息"""
        if self.is_producer:
            return {
                "role": "producer",
                "producer_id": self.producer_id,
                "channel_name": self.channel_name,
                "product_count": self._read_insert_count(),
            }
        else:
            return {
                "role": "consumer",
                "total_channels": len(self.channels),
                "producer_ids": sorted(self.channels.keys()),
                "last_discovery": time.time() - self.last_discovery_time,
                "consumer_count": self._read_sample_count(),
                "product_count": self._read_insert_count(),
            }

    def total_size(self):
        return CONFIG.train_batch_size

    def _setup_shared_counters(self, base_name: str):
        """设置跨进程共享计数器（带锁保护）"""
        import mmap

        counter_shm_name = f"/{base_name}_stats"
        counter_lock_name = f"/{base_name}_stats_lock"

        try:
            # 创建或打开共享内存
            try:
                self.counter_shm = posix_ipc.SharedMemory(
                    counter_shm_name, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, size=16, mode=0o600
                )
                is_creator = True
            except posix_ipc.ExistentialError:
                self.counter_shm = posix_ipc.SharedMemory(counter_shm_name)
                is_creator = False

            # 映射内存
            self.counter_mm = mmap.mmap(self.counter_shm.fd, 16, access=mmap.ACCESS_WRITE)
            self.counter_shm.close_fd()

            # 创建或打开信号量作为锁（初始值=1，表示未锁定）
            try:
                self.counter_lock = posix_ipc.Semaphore(
                    counter_lock_name, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, initial_value=1, mode=0o600
                )
            except posix_ipc.ExistentialError:
                self.counter_lock = posix_ipc.Semaphore(counter_lock_name)

            # 如果是创建者，初始化为0
            if is_creator:
                self.counter_mm[0:8] = (0).to_bytes(8, "little")
                self.counter_mm[8:16] = (0).to_bytes(8, "little")
                self.counter_mm.flush()

            self.counter_shm_name = counter_shm_name
            self.counter_lock_name = counter_lock_name
            self.logger.info(f"Shared counters {'created' if is_creator else 'opened'}: {counter_shm_name}")

        except Exception as e:
            self.logger.warning(f"Failed to setup shared counters: {e}")
            self.counter_mm = None
            self.counter_shm = None
            self.counter_lock = None

    def _read_insert_count(self):
        """读取全局插入计数（读操作加锁更安全）"""
        if self.counter_mm is None:
            return 0

        try:
            self.counter_lock.acquire()
            count = int.from_bytes(self.counter_mm[0:8], "little")
            self.counter_lock.release()
            return count
        except Exception:
            return 0

    def get_insert_speed(self):
        count = self._read_insert_count()
        return count, count

    def _read_sample_count(self):
        """读取全局采样计数"""
        if self.counter_mm is None:
            return 0

        try:
            self.counter_lock.acquire()
            count = int.from_bytes(self.counter_mm[8:16], "little")
            self.counter_lock.release()
            return count
        except Exception:
            return 0

    def _increment_insert_count(self, n=1):
        """递增插入计数（原子操作）"""
        if self.counter_mm is None or self.counter_lock is None:
            return

        try:
            # 加锁
            self.counter_lock.acquire()

            # 读-改-写
            count = int.from_bytes(self.counter_mm[0:8], "little")
            count += n
            self.counter_mm[0:8] = count.to_bytes(8, "little")
            self.counter_mm.flush()

            # 解锁
            self.counter_lock.release()
        except Exception as e:
            self.logger.error(f"Failed to increment insert count: {e}")
            # 确保锁被释放
            try:
                self.counter_lock.release()
            except Exception:
                pass

    def _increment_sample_count(self, n=1):
        """递增采样计数（原子操作）"""
        if self.counter_mm is None or self.counter_lock is None:
            return

        try:
            self.counter_lock.acquire()

            count = int.from_bytes(self.counter_mm[8:16], "little")
            count += n
            self.counter_mm[8:16] = count.to_bytes(8, "little")
            self.counter_mm.flush()

            self.counter_lock.release()
        except Exception as e:
            self.logger.error(f"Failed to increment sample count: {e}")
            try:
                self.counter_lock.release()
            except Exception:
                pass

    def close(self):
        if self.is_producer:
            if hasattr(self, "channel"):
                self.channel.close()
        else:
            for ch in self.channels.values():
                try:
                    ch.close()
                except Exception:
                    pass

        # 关闭共享计数器
        if hasattr(self, "counter_mm") and self.counter_mm:
            try:
                self.counter_mm.close()
            except Exception:
                pass

        # 关闭锁（不unlink）
        if hasattr(self, "counter_lock") and self.counter_lock:
            try:
                # 注意：不要unlink，让OS在所有进程退出后清理
                pass
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
