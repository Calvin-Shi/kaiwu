#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import mmap
import msgpack
from typing import Optional
import posix_ipc
import time


class SemaphoreSharedMemoryChannel:
    """
    单向通道：
    - 使用命名共享内存承载数据
    - 使用三个命名信号量（empty/full/ack）做单槽生产者-消费者同步
      empty: 初值1；发送前 acquire；接收后 release
      full:  初值0；发送后 release；接收前 acquire
      ack:   初值0；接收后 release；发送可选等待 acquire 以“确认已读”

    共享内存布局：
    [0:4) -> uint32 小端长度
    [4:4+len) -> 负载数据
    """

    LEN_OFFSET = 0
    HEADER_SIZE = 4

    def __init__(
        self,
        name_prefix: str,
        size: int = 1024 * 1024,
        create: bool = True,
        unlink_on_close: bool = False,
    ):
        """
        name_prefix: 作为命名前缀，实际资源名为:
          shm:  /{name_prefix}_shm
          sems: /{name_prefix}_empty, /{name_prefix}_full, /{name_prefix}_ack
        size: 共享内存大小（包含4字节长度头）
        create: True 时尝试创建（若已存在则打开），False 仅打开已存在资源
        unlink_on_close: 关闭时是否尝试 unlink 资源（需确保对端已不使用）
        """
        if not name_prefix or any(c.isspace() for c in name_prefix):
            raise ValueError("name_prefix 不能为空或包含空白字符")

        self.size = max(int(size), self.HEADER_SIZE + 1)
        self.max_payload = self.size - self.HEADER_SIZE
        self.unlink_on_close = bool(unlink_on_close)

        # POSIX 命名对象要求以斜杠开头
        def norm(n: str) -> str:
            return n if n.startswith("/") else "/" + n

        self.shm_name = norm(f"{name_prefix}_shm")
        self.sem_empty_name = norm(f"{name_prefix}_empty")
        self.sem_full_name = norm(f"{name_prefix}_full")
        self.sem_ack_name = norm(f"{name_prefix}_ack")

        self._mm = None
        self._shm = None
        self._sem_empty = None
        self._sem_full = None
        self._sem_ack = None

        # 打开/创建共享内存与信号量
        self._open_resources(create=create)

    def _open_resources(self, create: bool):
        created_any = False

        # 共享内存：优先尝试 O_CREAT|O_EXCL 判断是否首次创建
        if create:
            try:
                self._shm = posix_ipc.SharedMemory(
                    self.shm_name, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, mode=0o600, size=self.size
                )
                created_any = True
            except posix_ipc.ExistentialError:
                # 已存在则打开
                self._shm = posix_ipc.SharedMemory(self.shm_name)
        else:
            self._shm = posix_ipc.SharedMemory(self.shm_name)

        # 内存映射
        # 注意：posix_ipc.SharedMemory 提供 .fd，需要 mmap 后关闭 fd
        self._mm = mmap.mmap(self._shm.fd, self.size, access=mmap.ACCESS_WRITE)
        self._shm.close_fd()  # 关闭文件描述符，映射仍然有效

        # 三把信号量
        def open_sem(name: str, initial: int):
            if create:
                try:
                    sem = posix_ipc.Semaphore(
                        name, flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL, initial_value=initial, mode=0o600
                    )
                    return sem, True
                except posix_ipc.ExistentialError:
                    sem = posix_ipc.Semaphore(name)  # 打开已存在的
                    return sem, False
            else:
                return posix_ipc.Semaphore(name), False

        self._sem_empty, created_e = open_sem(self.sem_empty_name, initial=1)
        self._sem_full, created_f = open_sem(self.sem_full_name, initial=0)
        self._sem_ack, created_a = open_sem(self.sem_ack_name, initial=0)

        # 记录是否由本实例首次创建，供 unlink 决策参考（但默认不 unlink）
        self._created = created_any or created_e or created_f or created_a

    @staticmethod
    def _acquire(sem: posix_ipc.Semaphore, timeout: Optional[float]) -> None:
        try:
            if timeout is None:
                sem.acquire()
            else:
                sem.acquire(timeout=float(timeout))
        except posix_ipc.BusyError:
            raise TimeoutError("semaphore acquire timeout")

    def send(self, data: bytes, timeout: Optional[float] = None, await_ack: bool = True) -> None:
        """
        发送一条消息（单槽缓冲区）。默认等待接收方确认（await_ack=True）。
        data: bytes 负载（长度不得超过 max_payload）
        timeout: 可选超时（秒），同时应用于 empty/ack 的等待
        await_ack: True 时等待对端读完再返回，语义与原实现一致
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data 必须是 bytes-like")
        if len(data) > self.max_payload:
            raise ValueError(f"payload 过大（{len(data)} > {self.max_payload}）")

        # P(empty)
        self._acquire(self._sem_empty, timeout)

        # 写入共享内存：长度 + 数据
        l = len(data)
        self._mm[self.LEN_OFFSET : self.LEN_OFFSET + 4] = int(l).to_bytes(4, "little")
        if l:
            self._mm[self.HEADER_SIZE : self.HEADER_SIZE + l] = data
        self._mm.flush()

        # V(full)
        self._sem_full.release()

        # 可选等待对端确认
        if await_ack:
            self._acquire(self._sem_ack, timeout)

    def recv(self, timeout: Optional[float] = None, return_memoryview: bool = False) -> bytes:
        """
        接收一条消息。
        timeout: 可选超时（秒），用于等待 full
        return_memoryview: True 时返回 memoryview（避免复制，生命周期需在下次 recv 前使用完）
        """
        # P(full)
        self._acquire(self._sem_full, timeout)

        # 读取长度与数据
        l = int.from_bytes(self._mm[self.LEN_OFFSET : self.LEN_OFFSET + 4], "little")
        if l < 0 or l > self.max_payload:
            # 数据损坏或协议不一致
            # 尽力恢复信号量状态，使用try-except确保不会因信号量操作失败而掩盖原始异常
            try:
                self._sem_ack.release()
                self._sem_empty.release()
            except Exception:
                # 信号量操作失败不应掩盖数据损坏异常
                pass
            raise ValueError(f"非法长度字段: {l}")

        if return_memoryview:
            payload_view = memoryview(self._mm)[self.HEADER_SIZE : self.HEADER_SIZE + l]
            # V(ack)，V(empty)
            self._sem_ack.release()
            self._sem_empty.release()
            return payload_view
        else:
            payload = bytes(self._mm[self.HEADER_SIZE : self.HEADER_SIZE + l])
            # V(ack)，V(empty)
            self._sem_ack.release()
            self._sem_empty.release()
            return payload

    def close(self):
        # 按正确顺序关闭资源：先关闭内存映射，再处理信号量和共享内存
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None

        # 注意：POSIX 命名对象的 unlink 应只在确认所有进程不再使用后进行
        if self.unlink_on_close:
            for obj, name in (
                (self._shm, self.shm_name),
                (self._sem_empty, self.sem_empty_name),
                (self._sem_full, self.sem_full_name),
                (self._sem_ack, self.sem_ack_name),
            ):
                if obj is None:
                    continue
                try:
                    # 共享内存与信号量对象有各自的 unlink 接口
                    if isinstance(obj, posix_ipc.SharedMemory):
                        obj.unlink()
                    elif isinstance(obj, posix_ipc.Semaphore):
                        obj.unlink()
                except posix_ipc.ExistentialError:
                    pass  # 已被对端或其他地方 unlink
                except Exception:
                    # 其他异常不应影响整体关闭流程
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class DuplexPosixChannel:
    """
    基于posix_ipc库（封装 POSIX 标准的共享内存与信号量 API），构建 “单向 / 全双工跨进程通信通道”，
    核心是通过POSIX 命名共享内存承载数据、POSIX 命名信号量实现同步，解决无继承关系的跨进程数据交互问题，且避免轮询带来的性能损耗。

    全双工通道：两条单向通道组成。
    out_path: 本端->对端 的通道资源名前缀
    in_path:  对端->本端 的通道资源名前缀
    采用 msgpack 序列化，send/recv 支持任意 Python 对象；binary=True 时直传 bytes。
    """

    def __init__(
        self,
        out_path: str,
        in_path: str,
        size: int = 1024 * 1024,
        create: bool = True,
        unlink_on_close: bool = False,
    ):
        # 记录配置，便于重连
        self._cfg = dict(
            out_path=out_path,
            in_path=in_path,
            size=size,
            create=create,
            unlink_on_close=unlink_on_close,
        )
        self._out = SemaphoreSharedMemoryChannel(
            name_prefix=out_path, size=size, create=create, unlink_on_close=unlink_on_close
        )
        self._in = SemaphoreSharedMemoryChannel(
            name_prefix=in_path, size=size, create=create, unlink_on_close=unlink_on_close
        )

    def send(self, data, timeout: Optional[float] = None, binary: bool = False, await_ack: bool = True) -> None:
        if binary:
            payload = data
        else:
            payload = msgpack.packb(data, use_bin_type=True, strict_types=True)
        self._out.send(payload, timeout=timeout, await_ack=await_ack)

    def recv(self, timeout: Optional[float] = None, binary: bool = False):
        payload = self._in.recv(timeout=timeout)
        if binary:
            return payload
        else:
            return msgpack.unpackb(payload, raw=False)

    def _try_open_channels(self, create: bool):
        """尝试按配置打开两条通道；失败时清理已打开的一侧并抛出异常。"""
        out = None
        inn = None
        try:
            out = SemaphoreSharedMemoryChannel(
                name_prefix=self._cfg["out_path"],
                size=self._cfg["size"],
                create=create,
                unlink_on_close=self._cfg["unlink_on_close"],
            )
            inn = SemaphoreSharedMemoryChannel(
                name_prefix=self._cfg["in_path"],
                size=self._cfg["size"],
                create=create,
                unlink_on_close=self._cfg["unlink_on_close"],
            )
            return out, inn
        except Exception:
            # 确保异常时正确清理已打开的资源
            for ch in [out, inn]:
                if ch is not None:
                    try:
                        # 临时禁用 unlink 以避免影响其他进程
                        old_flag = ch.unlink_on_close
                        ch.unlink_on_close = False
                        ch.close()
                        ch.unlink_on_close = old_flag
                    except Exception:
                        # 关闭过程中的异常不应掩盖原始异常
                        pass
            raise

    def reconnect(self, attempts: int = 3, delay: float = 0.2, create: Optional[bool] = None) -> None:
        """
        重连到同名的共享内存与信号量。
        attempts: 重试次数
        delay: 每次重试间隔（秒）
        create: 覆盖是否创建资源（默认沿用构造时的设置）
        说明：
        - 若另一端尚未创建资源且 create=False，可能出现不存在错误；可通过提高 attempts 或设 create=True 来处理。
        - 为避免在失败时影响现有通道，先成功打开新通道，再关闭旧通道并切换。
        """
        if create is None:
            create = self._cfg["create"]

        last_exc = None
        for i in range(int(attempts)):
            try:
                new_out, new_in = self._try_open_channels(create=create)

                # 成功打开后，关闭旧通道（不 unlink）
                def _close_wo_unlink(ch):
                    if ch is None:
                        return
                    old_flag = ch.unlink_on_close
                    ch.unlink_on_close = False
                    try:
                        ch.close()
                    finally:
                        ch.unlink_on_close = old_flag

                old_out, old_in = self._out, self._in
                self._out, self._in = new_out, new_in
                _close_wo_unlink(old_out)
                _close_wo_unlink(old_in)

                # 记录最新 create 策略
                self._cfg["create"] = create
                return
            except Exception as e:
                last_exc = e
                if i < attempts - 1:
                    time.sleep(delay)

        # 最终失败
        raise last_exc

    def close(self):
        try:
            if hasattr(self, "_out") and self._out:
                self._out.close()
        finally:
            if hasattr(self, "_in") and self._in:
                self._in.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
