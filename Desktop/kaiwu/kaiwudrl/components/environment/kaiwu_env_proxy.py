#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import msgpack
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from common_python.ipc.zmq_util import ZmqClient, ZmqConfig
from common_python.ipc.duplex_posix_channel import DuplexPosixChannel
from kaiwudrl.common.utils.common_func import (
    set_schedule_event,
    compress_data,
    get_uuid,
    is_valid_ip,
    is_valid_port,
)
from kaiwudrl.components.environment.env_wrapper import process_error_env_obs


class KaiwuEnvProxy:
    """
    A class to manage the environment communication via ZMQ/SharedMemory for Kaiwu.
    """

    def __init__(self, logger, monitor_proxy) -> None:

        self.logger = logger
        self.monitor_proxy = monitor_proxy
        self.ip = None
        self.port = None
        self.need_connect = False

        if CONFIG.aisrv_env_ipc_method == KaiwuDRLDefine.AISRV_ENV_IPC_METHOD_SHARED_MEMORY:
            self.duplex_client = None
        else:
            # zmq通信client
            self.zmq_client = None
            self.zmq_client_id = None
            self.zmq_config = None

        super(KaiwuEnvProxy, self).__init__()

    def init(self, ip, port):
        if not ip or not port:
            self.logger.error(f"ip: {ip} or port: {port} is None, please check")
            return False

        if not is_valid_ip(ip) or not is_valid_port(port):
            self.logger.error(f"ip: {ip} or port: {port} is not valid, please check")
            return False

        self.ip = ip
        self.port = port

        if CONFIG.aisrv_env_ipc_method == KaiwuDRLDefine.AISRV_ENV_IPC_METHOD_SHARED_MEMORY:
            out_path = f"aisrv_{self.ip}_{self.port}"
            in_path = f"{self.ip}_{self.port}_aisrv"
            self.duplex_client = DuplexPosixChannel(
                out_path=out_path, in_path=in_path, size=CONFIG.shared_memory_max_size, create=True
            )
            self.logger.info(
                f"kaiwu_env_proxy use shared_memory, DuplexPosixChannel aisrv <--> env, start at out:{out_path}, in:{in_path} success"
            )
        else:
            self.client_id = get_uuid()
            self.zmq_config = ZmqConfig(
                zmq_io_threads_server=CONFIG.zmq_io_threads_server,
                zmq_io_threads_client=CONFIG.zmq_io_threads_client,
                tcp_keep_alive=CONFIG.tcp_keep_alive,
                tcp_keep_alive_idle=CONFIG.tcp_keep_alive_idle,
                tcp_keep_alive_intvl=CONFIG.tcp_keep_alive_intvl,
                tcp_keep_alive_cnt=CONFIG.tcp_keep_alive_cnt,
                sock_buff_size=CONFIG.sock_buff_size,
                backlog_size=CONFIG.backlog_size,
                tcp_immediate=CONFIG.tcp_immediate,
                zmq_ops_sendhwm=CONFIG.zmq_ops_sendhwm,
                zmq_ops_recvhwm=CONFIG.zmq_ops_recvhwm,
            )

            self.zmq_client = ZmqClient(
                str(self.client_id),
                self.ip,
                str(self.port),
                self.zmq_config,
            )

            self.zmq_client.connect()
            self.logger.info(
                f"kaiwu_env_proxy use zmq, connect to {self.ip}, port is {self.port}, client_id is {self.client_id}",
            )

        return True

    def reset(self, usr_conf):
        """
        处理reset的请求和响应
        result_code < 0 会终止训练, result_code > 0 会重试继续训练
        """
        if usr_conf is None:
            self.logger.error(f"kaiwu_env_proxy usr_conf is None, please check")
            return process_error_env_obs(result_code=-1, result_message=f"usr_conf is None, please check")

        # 处理重连
        if self.need_connect:
            self.reconnect()

        b_usr_conf = msgpack.dumps(usr_conf)
        message = {"message_type": "reset", "message_data": b_usr_conf}

        try:
            b_result = self.send_and_receive(timeout=60, message=message)
            if b_result is None:
                return process_error_env_obs(result_code=1, result_message=f"send_and_receive failed, please check")

            env_info = msgpack.loads(b_result)

        except Exception as e:
            self.logger.error(f"kaiwu_env_proxy failed to get env_info during reset, error: {str(e)}")
            self.need_connect = True
            return process_error_env_obs(
                result_code=1, result_message=f"failed to get env_info during reset, error: {str(e)}"
            )

        return env_info["obs"]

    def step(self, action_data):
        """
        处理step的请求和响应
        """
        if action_data is None:
            self.logger.error(f"kaiwu_env_proxy action_data is None, please check")
            return None, process_error_env_obs(result_code=-1, result_message=f"action_data is None, please check")

        # 处理重连
        if self.need_connect:
            self.reconnect()

        b_action = msgpack.dumps(action_data)
        message = {"message_type": "step", "message_data": b_action}
        try:
            b_result = self.send_and_receive(timeout=60, message=message)
            if b_result is None:
                return None, process_error_env_obs(
                    result_code=1, result_message=f"send_and_receive failed, please check"
                )

            env_info = msgpack.loads(b_result)
            return env_info["reward"], env_info["obs"]

        except Exception as e:
            self.logger.error(f"kaiwu_env_proxy failed to get env_info during step, error: {str(e)}")
            self.need_connect = True
            return None, process_error_env_obs(
                result_code=1, result_message=f"failed to get env_info during step, error: {str(e)}"
            )

    def client_send(self, data, timeout, binary):

        if CONFIG.aisrv_env_ipc_method == KaiwuDRLDefine.AISRV_ENV_IPC_METHOD_SHARED_MEMORY:
            self.duplex_client.send(data, binary=binary)
        else:
            self.zmq_client.send(data, timeout=timeout, binary=binary)

    def client_recv(self, block, timeout, binary):
        if CONFIG.aisrv_env_ipc_method == KaiwuDRLDefine.AISRV_ENV_IPC_METHOD_SHARED_MEMORY:
            return self.duplex_client.recv(timeout=timeout, binary=binary)
        else:
            return self.zmq_client.recv(block=block, timeout=timeout, binary=binary)

    def reconnect(self):
        """
        处理重连, 注意共享内存不需要重新初始化
        """
        if CONFIG.aisrv_env_ipc_method == KaiwuDRLDefine.AISRV_ENV_IPC_METHOD_SHARED_MEMORY:
            pass
        else:
            self.client_id = get_uuid()
            self.zmq_client = ZmqClient(
                str(self.client_id),
                self.ip,
                str(self.port),
                self.zmq_config,
            )

            self.zmq_client.connect()
            self.logger.info(
                f"kaiwu_env_proxy use zmq, reconnect to {self.ip}, port is {self.port}, client_id is {self.client_id}",
            )

        # 复原标志位
        self.need_connect = False

    def send_and_receive(self, timeout, message):
        """
        reset和step都需要采用同步等待方法
        """
        if not message:
            self.logger.info("kaiwu_env_proxy message is None, please check")
            return None
        try:
            self.client_send(message, timeout=timeout, binary=False)
            while True:
                result = self.client_recv(block=True, timeout=timeout, binary=True)
                if result:
                    break
            return result
        except Exception as e:
            self.need_connect = True
            self.logger.info(f"kaiwu_env_proxy communication send_and_receive failed, error_message: {str(e)}")
            return None

    def close(self):
        pass
