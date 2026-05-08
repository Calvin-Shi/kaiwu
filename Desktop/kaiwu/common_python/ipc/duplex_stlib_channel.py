#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import msgpack
import time
from typing import Optional, Dict, Any
from multiprocessing import shared_memory, Semaphore


class SharedMemoryChannelStdlib:
    """
    完全基于 Python 标准库实现：
    - 使用命名共享内存承载数据（通过共享内存对象实现跨进程通信）
    - 使用三个 Semaphore（empty/full/ack）做同步
      empty: 初值1；发送前 acquire；接收后 release
      full:  初值0；发送后 release；接收前 acquire
      ack:   初值0；接收后 release；发送可选等待 acquire 以“确认已读”
    共享内存布局：
      [0:4) -> uint32 小端长度
      [4:4+len) -> 负载数据
    说明：
    - 如果在父进程中创建了 shm 与信号量，子进程可以通过传入 shm_name/对象以及三枚 Semaphore 对象来复用资源。
    - 也可以在子进程中仅传入名称来连接到已有资源（前提是父进程在创建时使用相同名称）。
    - 该实现不依赖服务器，依赖 fork/继承或对象传递来实现跨进程共享。
    """

    LEN_OFFSET = 0
    HEADER_SIZE = 4

    def __init__(
        self,
        name_prefix: str,
        size: int = 1024 * 1024,
        create: bool = True,
        unlink_on_close: bool = False,
        shm: Optional[shared_memory.SharedMemory] = None,
        shm_name: Optional[str] = None,
        semaphores: Optional[Dict[str, Semaphore]] = None,
        # 以下两个字段仅用于调试和兼容性
        _owner_created: bool = False,  # 由本进程创建的 shm/sem 是否需要在 close 时 unlink
        **kwargs: Any,
    ):
        """
        name_prefix: 用作共享内存的名字前缀，实际资源名为:
          shm:  /{name_prefix}_shm
        size: 共享内存大小（包含4字节长度头）
        create: True 时尝试创建（若已存在则打开），False 仅打开已存在资源
        unlink_on_close: 关闭时是否 unlink 资源（需确保对端已不使用）
        shm: 已存在的 SharedMemory 对象，可直接在父进程创建后传给子进程
        shm_name: 已存在的共享内存名称，子进程通过此名称连接已有资源
        semaphores: 已存在的三个 Semaphore，键为 'empty','full','ack'
        """
        if not name_prefix or any(c.isspace() for c in name_prefix):
            raise ValueError("name_prefix 不能为空或包含空白字符")

        self.size = max(int(size), self.HEADER_SIZE + 1)
        self.max_payload = self.size - self.HEADER_SIZE
        self.unlink_on_close = bool(unlink_on_close)

        # 共享内存名称需要以 '/' 开头，符合 POSIX/Common 命名习惯
        def norm(n: str) -> str:
            return n if n.startswith("/") else "/" + n

        self.shm_name = norm(f"{name_prefix}_shm")
        self._shm = None
        self._buf = None
        self._created = False  # 是否为本进程首次创建

        # 允许直接传入现有 shm 对象或名称
        if shm is not None:
            self._shm = shm
            self._buf = self._shm.buf
            self._created = False
        else:
            # 通过名称或创建/连接共享内存
            if shm_name:
                self.shm_name = shm_name
            if create:
                try:
                    self._shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.size)
                    self._created = True
                except FileExistsError:
                    self._shm = shared_memory.SharedMemory(name=self.shm_name)
                    self._created = False
            else:
                self._shm = shared_memory.SharedMemory(name=self.shm_name)
                self._created = False

            self._buf = self._shm.buf

        # 三个信号量
        if semaphores is not None:
            self._sem_empty = semaphores["empty"]
            self._sem_full = semaphores["full"]
            self._sem_ack = semaphores["ack"]
        else:
            # 无服务器场景：在父进程创建后通过 fork/传参给子进程共享
            self._sem_empty = Semaphore(1)
            self._sem_full = Semaphore(0)
            self._sem_ack = Semaphore(0)

        # 记录创建信息，便于 close 时决定是否 unlink
        self._owner_created = self._created

    def _acquire(self, sem: Semaphore, timeout: Optional[float]):
        if timeout is None:
            sem.acquire()
        else:
            acquired = sem.acquire(timeout=timeout)
            if not acquired:
                raise TimeoutError("semaphore acquire timeout")

    def send(self, data: bytes, timeout: Optional[float] = None, await_ack: bool = True) -> None:
        """
        发送一条消息（单槽缓冲区）。默认等待对端确认（await_ack=True）。
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
        self._buf[self.LEN_OFFSET : self.LEN_OFFSET + 4] = int(l).to_bytes(4, "little")
        if l:
            self._buf[self.HEADER_SIZE : self.HEADER_SIZE + l] = data
        # 不需要显式 flush，memoryview 会自动同步

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
        l = int.from_bytes(self._buf[self.LEN_OFFSET : self.LEN_OFFSET + 4], "little")
        if l < 0 or l > self.max_payload:
            # 数据损坏或协议不一致
            self._sem_ack.release()
            self._sem_empty.release()
            raise ValueError(f"非法长度字段: {l}")

        if return_memoryview:
            payload_view = memoryview(self._buf)[self.HEADER_SIZE : self.HEADER_SIZE + l]
            # V(ack)，V(empty)
            self._sem_ack.release()
            self._sem_empty.release()
            return payload_view
        else:
            payload = bytes(self._buf[self.HEADER_SIZE : self.HEADER_SIZE + l])
            # V(ack)，V(empty)
            self._sem_ack.release()
            self._sem_empty.release()
            return payload

    def close(self):
        try:
            if self._shm is not None:
                self._shm.close()
        finally:
            if self.unlink_on_close and self._owner_created:
                # unlink 共享内存对象（注意：若另一端仍在使用，不应 unlink，调用方应确保安全）
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class DuplexChannelStdlib:
    """
    全双工通道：两条单向通道组成。
    out_path: 本端->对端 的通道资源名前缀
    in_path:  对端->本端 的通道资源名前缀
    采用 msgpack 序列化，send/recv 支持任意 Python 对象；binary=True 时直传 bytes。
    重要：需要两条通道各自的共享内存和信号量，且在父进程创建后通过 fork 继承给子进程使用，或直接传递对象。
    """

    def __init__(
        self,
        out_path: str,
        in_path: str,
        size: int = 1024 * 1024,
        create: bool = True,
        unlink_on_close: bool = False,
        out_shm: Optional[shared_memory.SharedMemory] = None,
        in_shm: Optional[shared_memory.SharedMemory] = None,
        out_semaphores: Optional[Dict[str, Semaphore]] = None,
        in_semaphores: Optional[Dict[str, Semaphore]] = None,
        out_shm_name: Optional[str] = None,
        in_shm_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self._cfg = dict(
            out_path=out_path,
            in_path=in_path,
            size=size,
            create=create,
            unlink_on_close=unlink_on_close,
        )

        # 出站通道
        self._out = SharedMemoryChannelStdlib(
            name_prefix=out_path,
            size=size,
            create=create,
            unlink_on_close=unlink_on_close,
            shm=out_shm,
            shm_name=out_shm_name,
            semaphores=out_semaphores,
        )

        # 入站通道
        self._in = SharedMemoryChannelStdlib(
            name_prefix=in_path,
            size=size,
            create=create,
            unlink_on_close=unlink_on_close,
            shm=in_shm,
            shm_name=in_shm_name,
            semaphores=in_semaphores,
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
            out = SharedMemoryChannelStdlib(
                name_prefix=self._cfg["out_path"],
                size=self._cfg["size"],
                create=create,
                unlink_on_close=self._cfg["unlink_on_close"],
                shm=None,
                shm_name=None,
                semaphores=None,
            )
            inn = SharedMemoryChannelStdlib(
                name_prefix=self._cfg["in_path"],
                size=self._cfg["size"],
                create=create,
                unlink_on_close=self._cfg["unlink_on_close"],
                shm=None,
                shm_name=None,
                semaphores=None,
            )
            return out, inn
        except Exception:
            if out is not None:
                old = out.unlink_on_close
                out.unlink_on_close = False
                try:
                    out.close()
                finally:
                    out.unlink_on_close = old
            if inn is not None:
                old = inn.unlink_on_close
                inn.unlink_on_close = False
                try:
                    inn.close()
                finally:
                    inn.unlink_on_close = old
            raise

    def reconnect(self, attempts: int = 3, delay: float = 0.2, create: Optional[bool] = None) -> None:
        """
        重连到同名的共享内存与信号量。
        说明：
        - 该实现不包含信号量服务器，重连逻辑依赖于 fork/渡过同一父进程派生出的子进程场景或显式对象传递。
        """
        if create is None:
            create = self._cfg["create"]

        last_exc = None
        for i in range(int(attempts)):
            try:
                new_out, new_in = self._try_open_channels(create=create)

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

                self._cfg["create"] = create
                return
            except Exception as e:
                last_exc = e
                if i < attempts - 1:
                    time.sleep(delay)

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
