#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import msgpack
import time
import os
import mmap
from typing import Optional


class SharedMemoryChannel:
    STATUS_OFFSET = 0
    LEN_OFFSET = 4
    PAYLOAD_OFFSET = 8
    HEADER_SIZE = PAYLOAD_OFFSET

    STATUS_IDLE = 0
    STATUS_WRITTEN = 1
    STATUS_READ = 2

    def __init__(
        self, path: str, size: int = 1024 * 1024, role: str = "writer", create: bool = True, spin_delay: float = 0.001
    ):
        if role not in ("writer", "reader"):
            raise ValueError("role must be 'writer' or 'reader'")
        self.path = path
        self.size = max(size, self.HEADER_SIZE + 1)
        self.role = role
        self.spin_delay = spin_delay

        if create:
            if not os.path.exists(self.path):
                with open(self.path, "wb") as f:
                    f.write(b"\x00" * self.size)
            else:
                with open(self.path, "r+b") as f:
                    cur = os.path.getsize(self.path)
                    if cur < self.size:
                        f.truncate(self.size)

        self._f = open(self.path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), self.size, access=mmap.ACCESS_WRITE)
        self._set_status(self.STATUS_IDLE)

    def _read_status(self) -> int:
        return int.from_bytes(self._mm[self.STATUS_OFFSET : self.STATUS_OFFSET + 4], "little")

    def _set_status(self, v: int) -> None:
        self._mm[self.STATUS_OFFSET : self.STATUS_OFFSET + 4] = int(v).to_bytes(4, "little")

    def _read_len(self) -> int:
        return int.from_bytes(self._mm[self.LEN_OFFSET : self.LEN_OFFSET + 4], "little")

    def _read_payload(self, length: int) -> bytes:
        data = self._mm[self.PAYLOAD_OFFSET : self.PAYLOAD_OFFSET + length]
        return bytes(data)

    def _write_len_and_payload(self, payload: bytes) -> None:
        l = len(payload)
        self._mm[self.LEN_OFFSET : self.LEN_OFFSET + 4] = l.to_bytes(4, "little")
        self._mm[self.PAYLOAD_OFFSET : self.PAYLOAD_OFFSET + l] = payload

    def _wait_for_status(self, target_status: int, timeout: Optional[float]) -> bool:
        import time

        start = time.time()
        while True:
            if self._read_status() == target_status:
                return True
            if timeout is not None and (time.time() - start) >= timeout:
                return False
            time.sleep(self.spin_delay)

    def send(self, data: bytes, timeout: Optional[float] = None) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        if len(data) > self.size - self.PAYLOAD_OFFSET:
            raise ValueError("payload too large for shared memory region")

        if not self._wait_for_status(self.STATUS_IDLE, timeout):
            raise TimeoutError("send: timeout waiting for idle")

        self._mm[self.LEN_OFFSET : self.LEN_OFFSET + 4] = len(data).to_bytes(4, "little")
        self._mm[self.PAYLOAD_OFFSET : self.PAYLOAD_OFFSET + len(data)] = data
        self._set_status(self.STATUS_WRITTEN)

        if not self._wait_for_status(self.STATUS_READ, timeout):
            raise TimeoutError("send: timeout waiting for peer read")

        self._set_status(self.STATUS_IDLE)

    def recv(self, timeout: Optional[float] = None) -> bytes:
        import time

        start = time.time()
        while True:
            if self._read_status() == self.STATUS_WRITTEN:
                length = self._read_len()
                payload = self._read_payload(length)
                self._set_status(self.STATUS_READ)
                return payload
            if timeout is not None and (time.time() - start) >= timeout:
                raise TimeoutError("recv timeout")
            time.sleep(self.spin_delay)

    def close(self):
        try:
            if hasattr(self, "_mm") and self._mm:
                self._mm.close()
        finally:
            if hasattr(self, "_f") and self._f:
                self._f.close()


class DuplexSpinChannel:
    """
    方案通过mmap（内存映射文件）实现跨进程共享内存，并结合状态位机制实现同步，构建了单向 / 全双工通信通道。
    核心是将文件映射到进程地址空间，通过读写共享内存区域完成数据传输，无需依赖进程间信号量，仅通过状态位轮询实现同步。


    双向通道：对端的两条方向各自一份共享文件
    out_path: 发送方向的共享文件
    in_path:  接收方向的共享文件
    通过对这两份共享内存的读写实现两端的全双工通信
    对上层的数据类型目前采用 msgpack 序列化，send/recv 支持任意 Python 对象。
    """

    def __init__(
        self, out_path: str, in_path: str, size: int = 1024 * 1024, create: bool = True, spin_delay: float = 0.001
    ):

        self._out = SharedMemoryChannel(path=out_path, size=size, role="writer", create=create, spin_delay=spin_delay)
        self._in = SharedMemoryChannel(path=in_path, size=size, role="reader", create=create, spin_delay=spin_delay)

    def _reopen(self):
        # 重新绑定两个方向的通道
        self._out.close()
        self._in.close()
        # 尝试以相同参数重新打开
        self._out = SharedMemoryChannel(path=self._out.path, size=self._out.size, role="writer", create=True)
        self._in = SharedMemoryChannel(path=self._in.path, size=self._in.size, role="reader", create=True)

    def reconnect(self):
        self._reopen()

    def send(self, data, timeout: Optional[float] = None, binary=False) -> None:
        if binary:
            payload = data
        else:
            payload = msgpack.packb(data, use_bin_type=True, strict_types=True)
        self._out.send(payload, timeout=timeout)

    def recv(self, timeout: Optional[float] = None, binary=False):
        payload = self._in.recv(timeout=timeout)
        if binary:
            return payload
        else:
            return msgpack.unpackb(payload, raw=False)

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
