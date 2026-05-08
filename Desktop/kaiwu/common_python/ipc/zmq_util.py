#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import queue
import socket
import threading
import time

try:
    import msgpack
    import msgpack_numpy

    msgpack_numpy.patch()
except ImportError:
    import msgpack

    msgpack_numpy = None

# need pip install pyzmq
import zmq


def pick_unused_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    addr, port = s.getsockname()
    s.close()
    return port


"""
采用配置类方式, 支持多个配置项

使用实例:
config = ZmqConfig(
    zmq_io_threads_server=1,
    zmq_io_threads_client=1,
    tcp_keep_alive=1,
    tcp_keep_alive_idle=1,
    tcp_keep_alive_intvl=1,
    tcp_keep_alive_cnt=1,
    sock_buff_size=1,
    backlog_size=1,
    tcp_immediate=1,
    zmq_ops_sendhwm=1,
    zmq_ops_recvhwm=1
)

使用方式zmq_server = ZmqServer(config=config, logger=logger)

"""


class ZmqConfig:
    def __init__(
        self,
        zmq_io_threads_server=2,  # 默认值：2
        zmq_io_threads_client=1,  # 默认值：1
        tcp_keep_alive=1,  # 默认值：1
        tcp_keep_alive_idle=60,  # 默认值：60
        tcp_keep_alive_intvl=1,  # 默认值：1
        tcp_keep_alive_cnt=3,  # 默认值：3
        sock_buff_size=31457280,  # 默认值：31457280
        backlog_size=1024,  # 默认值：1024
        tcp_immediate=True,  # 默认值：True
        zmq_ops_sendhwm=30720,  # 默认值：30720
        zmq_ops_recvhwm=30720,  # 默认值：30720
    ):
        self.zmq_io_threads_server = zmq_io_threads_server
        self.zmq_io_threads_client = zmq_io_threads_client
        self.tcp_keep_alive = tcp_keep_alive
        self.tcp_keep_alive_idle = tcp_keep_alive_idle
        self.tcp_keep_alive_intvl = tcp_keep_alive_intvl
        self.tcp_keep_alive_cnt = tcp_keep_alive_cnt
        self.sock_buff_size = sock_buff_size
        self.backlog_size = backlog_size
        self.tcp_immediate = tcp_immediate
        self.zmq_ops_sendhwm = zmq_ops_sendhwm
        self.zmq_ops_recvhwm = zmq_ops_recvhwm


"""
zmq_server:
    svr = ZmqServer('127.0.0.1', 9999)
    while True:
        client_id, data = svr.recv()
        print("server receive data: " + str(data))

        svr.send(client_id, data)
        print("server send data: " + str(data))

zmq_client:
    svr = ZmqClient('client-id', '127.0.0.1', 9999)
    while True:
        data = "hello world"
        svr.send(data)
        print("client send data: " + str(data))

        data = svr.recv()
        print("client receive data: " + str(data))

建议是先启动zmq_server, 再启动zmq_client

注意下面情况：
1. 多进程使用时, 在init函数里调用init, 在run函数里调用bind
2. 单进程使用时, 连续调用init和bind即可
3. 如果需要支持请求可以从zmq_client发起, 也可以从zmq_server发起, 那么需要在初始化ZmqServer和ZmqClient传入duplex参数即可, 其余代码不用变
参考:
zmq_svr = ZmqServer('127.0.0.1', 9999, True)
zmq_client = ZmqClient('client-id', '127.0.0.1', 9999, True)


如果不按照上述方法调用, zmq在多进程环境里调用会出现收发包异常的情况
"""


class ZmqServer:
    def __init__(self, ip, port, config, duplex=False, pull_mode=False):
        self._context = zmq.Context()

        self.tcp_keep_alive = config.tcp_keep_alive
        self.tcp_keep_alive_idle = config.tcp_keep_alive_idle
        self.tcp_keep_alive_intvl = config.tcp_keep_alive_intvl
        self.tcp_keep_alive_cnt = config.tcp_keep_alive_cnt
        self.sock_buff_size = config.sock_buff_size
        self.backlog_size = config.backlog_size
        self.tcp_immediate = config.tcp_immediate
        self.zmq_ops_sendhwm = config.zmq_ops_sendhwm
        self.zmq_ops_recvhwm = config.zmq_ops_recvhwm
        self.zmq_io_threads_server = config.zmq_io_threads_server
        self.zmq_io_threads_client = config.zmq_io_threads_client

        """
        推荐的值取决于应用程序的需求和机器的硬件资源
        """
        self._context.set(zmq.IO_THREADS, self.zmq_io_threads_server)
        self._lock = threading.Lock()

        self.ip = ip
        self.port = port

        """
        False, 代表请求只能从client开始, 不能从server开始
        True, 代表请求可以从client和server开始
        """
        self.duplex = duplex

        """
        pull_mode=True时采用zmq.PULL, 只接收不回包, 适用于高吞吐量场景(如learner接收aisrv样本)
        注意: PULL模式下recv只返回data(没有client_id), send不可用
        """
        self.pull_mode = pull_mode

    def bind(self):
        # zmq.PUB, zmq.ROUTER, zmq.SUB, zmq.REQ, zmq.DEALER, zmq.PULL
        if self.pull_mode:
            self._socket = self._context.socket(zmq.PULL)
        else:
            self._socket = self._context.socket(zmq.ROUTER)
        self._socket.setsockopt(zmq.LINGER, 0)

        # 设置下网络参数
        self._socket.setsockopt(zmq.TCP_KEEPALIVE, self.tcp_keep_alive)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, self.tcp_keep_alive_idle)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, self.tcp_keep_alive_intvl)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, self.tcp_keep_alive_cnt)
        self._socket.setsockopt(zmq.SNDBUF, self.sock_buff_size)
        self._socket.setsockopt(zmq.RCVBUF, self.sock_buff_size)
        self._socket.setsockopt(zmq.BACKLOG, self.backlog_size)
        self._socket.setsockopt(zmq.IMMEDIATE, self.tcp_immediate)

        """
        zmq的发送和接收缓冲区大小对性能影响很大。如果缓冲区大小太小, 会导致消息堆积和阻塞,从而降低整体性能
        如果缓冲区大小太大, 会导致内存占用过多,从而影响系统的稳定性和可靠性
        目前KaiwuDRL使用到zmq的场景有:
        1. aisrv <--> actor, 单个包比较小, 设置发送接收缓冲区10MB合理
        2. aisrv <--> learner, 单个包比较大, 设置发送接收缓冲区30MB合理
        综合1和2的情况, 故设置发送接收缓冲区30MB合理
        """

        self._socket.setsockopt(zmq.SNDHWM, self.zmq_ops_sendhwm)
        self._socket.setsockopt(zmq.RCVHWM, self.zmq_ops_recvhwm)

        # self._socket.set_hwm(self.zmq_ops_hwm)
        self._socket.bind("tcp://" + self.ip + ":" + str(self.port))

    def readable(self):
        return self._socket.poll(0, flags=zmq.POLLIN) == zmq.POLLIN

    def recv_nowait(self):
        return self.recv(block=False)

    def recv(self, block=True, timeout=-1, binary=False):
        if block and timeout == -1:
            with self._lock:
                if self.pull_mode:
                    # PULL模式: 直接recv, 没有client_id
                    data = self._socket.recv()
                elif not self.duplex:
                    client_id, _, data = self._socket.recv_multipart()
                else:
                    [client_id, data] = self._socket.recv_multipart()
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Empty
            else:
                if not self._lock.acquire(False):
                    raise queue.Empty
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLIN):
                        raise queue.Empty
                elif not self.readable():
                    raise queue.Empty
                if self.pull_mode:
                    data = self._socket.recv()
                elif not self.duplex:
                    client_id, _, data = self._socket.recv_multipart()
                else:
                    [client_id, data] = self._socket.recv_multipart()
            finally:
                self._lock.release()

        if not binary:
            data = msgpack.unpackb(data, raw=False)

        # PULL模式只返回data, ROUTER模式返回(client_id, data)
        if self.pull_mode:
            return data

        client_id = str(client_id, "utf-8")
        return client_id, data

    def writable(self):
        return self._socket.poll(0, flags=zmq.POLLOUT) == zmq.POLLOUT

    """
    获取本地缓存的消息大小
    """

    def get_cache_message_count(self):
        return self._socket.getsockopt(zmq.RCVBUF)

    def send_nowait(self, client_id, data):
        self.send(client_id, data, block=False)

    def send(self, client_id, data, block=True, timeout=-1, binary=False):
        client_id = bytes(client_id, "utf-8")
        if not binary:
            data = msgpack.packb(data, use_bin_type=True)

        if block and timeout == -1:
            with self._lock:
                self._socket.send_multipart([client_id, b"", data])
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Full
            else:
                if not self._lock.acquire(False):
                    raise queue.Full
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLOUT):
                        raise queue.Full
                elif not self.writable():
                    raise queue.Full
                self._socket.send_multipart([client_id, b"", data])
            finally:
                self._lock.release()


"""
注意下面情况：
1. 多进程使用时, 在init函数里调用init, 在run函数里调用connect
2. 单进程使用时, 连续调用init和connect即可

如果不按照上述方法调用, zmq在多进程环境里调用会出现收发包异常的情况
"""


class ZmqClient:
    def __init__(self, client_id, ip, port, config, duplex=False, push_mode=False):
        self._context = zmq.Context()

        self.tcp_keep_alive = config.tcp_keep_alive
        self.tcp_keep_alive_idle = config.tcp_keep_alive_idle
        self.tcp_keep_alive_intvl = config.tcp_keep_alive_intvl
        self.tcp_keep_alive_cnt = config.tcp_keep_alive_cnt
        self.sock_buff_size = config.sock_buff_size
        self.backlog_size = config.backlog_size
        self.tcp_immediate = config.tcp_immediate
        self.zmq_ops_sendhwm = config.zmq_ops_sendhwm
        self.zmq_ops_recvhwm = config.zmq_ops_recvhwm
        self.zmq_io_threads_server = config.zmq_io_threads_server
        self.zmq_io_threads_client = config.zmq_io_threads_client

        self._context.set(zmq.IO_THREADS, self.zmq_io_threads_client)

        self._lock = threading.Lock()
        self.client_id = client_id
        self.ip = ip
        self.port = port

        """
        False, 代表请求只能从client开始, 不能从server开始
        True, 代表请求可以从client和server开始
        """
        self.duple = duplex

        """
        push_mode=True时采用zmq.PUSH, 只发送不接收回包, 适用于高吞吐量场景(如aisrv --> learner样本发送)
        """
        self.push_mode = push_mode

    def connect(self):
        """

        zmq支持的默认: zmq.PUB, zmq.ROUTER, zmq.SUB, zmq.REQ, zmq.DEALER
        C++/python版本zmq: DEALER/ROUTER
        push_mode=True时: zmq.PUSH, 只发送不接收
        """
        if self.push_mode:
            zmq_type = zmq.PUSH
        elif self.duple:
            zmq_type = zmq.DEALER
        else:
            zmq_type = zmq.REQ

        self._socket = self._context.socket(zmq_type)
        self._socket.setsockopt(zmq.LINGER, 0)

        # 设置下网络参数
        self._socket.setsockopt(zmq.SNDBUF, self.sock_buff_size)
        self._socket.setsockopt(zmq.RCVBUF, self.sock_buff_size)

        # 增加重连机制
        self._socket.setsockopt(zmq.TCP_KEEPALIVE, self.tcp_keep_alive)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, self.tcp_keep_alive_idle)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, self.tcp_keep_alive_intvl)
        self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, self.tcp_keep_alive_cnt)
        self._socket.setsockopt(zmq.SNDHWM, self.zmq_ops_sendhwm)
        self._socket.setsockopt(zmq.RCVHWM, self.zmq_ops_recvhwm)
        self._socket.setsockopt(zmq.IMMEDIATE, self.tcp_immediate)

        # self._socket.set_hwm(self.zmq_ops_hwm)
        self._socket.setsockopt(zmq.IDENTITY, bytes(self.client_id, "utf-8"))
        self._socket.connect("tcp://" + self.ip + ":" + str(self.port))

    def readable(self):
        return self._socket.poll(0, flags=zmq.POLLIN) == zmq.POLLIN

    def recv_nowait(self):
        return self.recv(block=False)

    def recv(self, block=True, timeout=-1, binary=False):
        if block and timeout == -1:
            with self._lock:
                data = self._socket.recv()
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Empty
            else:
                if not self._lock.acquire(False):
                    raise queue.Empty

            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLIN):
                        raise queue.Empty
                elif not self.readable():
                    raise queue.Empty
                data = self._socket.recv()
            finally:
                self._lock.release()

        if not binary:
            data = msgpack.unpackb(data, raw=False)
        return data

    def writable(self):
        return self._socket.poll(0, flags=zmq.POLLOUT) == zmq.POLLOUT

    def send_nowait(self, data):
        self.send(data, block=False)

    def send(self, data, block=True, timeout=-1, binary=False):
        if not binary:
            data = msgpack.packb(data, use_bin_type=True)

        if block and timeout == -1:
            with self._lock:
                self._socket.send(data)
        else:
            if block:
                deadline = time.monotonic() + timeout

            if block:
                if not self._lock.acquire(block, timeout):
                    raise queue.Full
            else:
                if not self._lock.acquire(False):
                    raise queue.Full
            try:
                if block:
                    timeout = deadline - time.monotonic()
                    if not self._socket.poll(timeout * 1000, flags=zmq.POLLOUT):
                        raise queue.Full
                elif not self.writable():
                    raise queue.Full
                self._socket.send(data)
            finally:
                self._lock.release()


"""
aisrv <--> actror上通信方法
1. aisrv --> actor, aisrv采用ZmqOpsClient, actor上采用ZMQPullSocket(类似Server)
2. actor --> aisrv, aisrv采用ZmqClient, actor上采用ZmqServer
"""


class ZmqOpsClient:
    def __init__(self, client_id, ip, port, config):
        self._context = zmq.Context()
        self._lock = threading.Lock()
        self.client_id = client_id
        self.ip = ip
        self.port = port

        self.tcp_keep_alive = config.tcp_keep_alive
        self.tcp_keep_alive_idle = config.tcp_keep_alive_idle
        self.tcp_keep_alive_intvl = config.tcp_keep_alive_intvl
        self.tcp_keep_alive_cnt = config.tcp_keep_alive_cnt
        self.sock_buff_size = config.sock_buff_size
        self.backlog_size = config.backlog_size
        self.tcp_immediate = config.tcp_immediate
        self.zmq_ops_sendhwm = config.zmq_ops_sendhwm
        self.zmq_ops_recvhwm = config.zmq_ops_recvhwm
        self.zmq_io_threads_server = config.zmq_io_threads_server
        self.zmq_io_threads_client = config.zmq_io_threads_client

    def connect(self):
        self._socket = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, self.zmq_ops_sendhwm)
        self._socket.setsockopt(zmq.RCVHWM, self.zmq_ops_recvhwm)

        # self._socket.set_hwm(self.zmq_ops_hwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IDENTITY, bytes(self.client_id, "utf-8"))
        self._socket.connect("tcp://" + self.ip + ":" + str(self.port))

    def send(self, data):
        self._socket.send(data, copy=False)


"""
zmq Poller
"""


class ZmqPoller:
    def __init__(self) -> None:
        self.poller = zmq.Poller()

    def get_poller(self):
        return self.poller

    # 注册读事务
    def register_read(self, socket):
        self.poller.register(socket, zmq.POLLIN)

    # 注册写事务
    def register_write(self, socket):
        self.poller.register(socket, zmq.POLLOUT)

    # 消息可读的标志位
    def get_zmq_pollin_state(self):
        return zmq.POLLIN

    # 消息可写的标志位
    def get_zmq_pollout_state(self):
        return zmq.POLLOUT
