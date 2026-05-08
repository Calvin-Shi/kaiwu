#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import socket
import struct


class RpcUtil(object):
    def __init__(self, host, port, logger):
        self.host = host
        self.port = port
        self.logger = logger

    def connect(self):
        """尝试连接到服务器"""
        max_retries = 600  # 最大重连尝试次数
        retry_count = 0  # 重连计数器

        while retry_count < max_retries:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 1MB 发送缓冲区
                self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
                # 1MB 接收缓冲区
                self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                # 禁用 Nagle 算法
                self.client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # 快速确认
                self.client_socket.setsockopt(socket.SOL_TCP, socket.TCP_QUICKACK, 1)
                self.client_socket.connect((self.host, self.port))

                self.logger.info(f"Connected to server, host is {self.host}, port is {self.port}")
                # 连接成功，跳出循环
                return True
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Failed to connect to server (attempt {retry_count}/{max_retries}): {e}")
                time.sleep(1)  # 等待1秒后重试连接

        if retry_count == max_retries:
            self.logger.error(
                f"Failed to connect to server after maximum retries, host is {self.host}, port is {self.port}"
            )
            return False

    def reconnect(self):
        """尝试重新连接到服务器"""
        try:
            self.client_socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        finally:
            self.client_socket.close()
            del self.client_socket

        # 再次重新链接到目的端
        self.connect()

    def send_all(self, data):
        """
        数据发送时, 如果异常就返回
        """
        try:
            # 使用内存视图避免拷贝
            with memoryview(data) as mv:
                # 合并包头与数据
                header = struct.pack("!I", len(data))
                self.client_socket.sendall(header + mv)

            return True
        except (BrokenPipeError, ConnectionResetError, TimeoutError) as e:
            self.logger.error(f"send_all failed, error message: {str(e)}")
            # 在异常后自动重连, 并且返回False, 由使用者调用处理
            self.reconnect()
            return False

    def recv_with_length(self):
        try:
            # 首先接收4字节的数据长度
            data_length = self.client_socket.recv(4)
            if not data_length:
                # 此时返回数据为None, 不是抛出来异常, 否则调用者会退出
                self.logger.error("self.client_socket.recv return 0, resetting connection")
                self.reconnect()
                return None

            # 将数据长度解包成整数
            length = struct.unpack("!I", data_length)[0]
            # 根据数据长度接收数据
            data = self.recv_all(length)
            return data
        except (socket.timeout, ConnectionResetError, BrokenPipeError) as e:
            self.logger.error("recv_with_length timeout, resetting connection")
            self.reconnect()
            return None

    def recv_all(self, length):
        # 预分配固定大小缓冲区
        buf = bytearray(length)
        view = memoryview(buf)
        bytes_recv = 0
        while bytes_recv < length:
            try:
                n = self.client_socket.recv_into(view[bytes_recv:], length - bytes_recv)
                if n == 0:
                    # 这里需要抛出来ConnectionResetError, 让recv_with_length捕捉到
                    raise ConnectionResetError("socket connection broken")
                bytes_recv += n
            except socket.timeout:
                self.logger.error("recv_all timeout, so raise exception")
                raise

        return bytes(buf)
