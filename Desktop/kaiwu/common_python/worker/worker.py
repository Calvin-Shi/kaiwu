#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
Worker通用基类, 框架标准化2.0引入
"""
import os
import abc
import time
import socket
import hashlib
import multiprocessing
from datetime import datetime
from dataclasses import dataclass
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from common_python.monitor.monitor_proxy_process import MonitorProxy
from common_python.monitor.monitor_manager import get_monitor_proxy
from common_python.alloc.alloc_proxy import AllocProxy
from common_python.config.config_control import CONFIG
from typing import Optional
from common_python.utils.common_func import get_interface_ip


"""
# 使用示例
worker_config = WorkerConfig(
    worker_name="AiSrvHandle",
    father_pid=os.getpid(),
    use_logger=True,
    use_default_monitor=True,
)

worker = Worker(worker_config)
"""

"""
以上设计可以最大化增强通用性与扩展性
基于以上部分，新建子类时可以定制配置如下：

# 新的配置类，继承自基类，新增 ip 字段
@dataclass
class NewWorkerConfig(WorkerConfig):
    ip: str = "localhost" # 新增字段

# 新的 Worker 子类，配合 NewWorkerConfig 使用
class NewWorker(Worker):
    def __init__(self, worker_config: NewWorkerConfig):
        # 先调用基类初始化，传入 NewWorkerConfig
        super().__init__(worker_config)
        # 新增字段在子类中初始化
        self.ip = worker_config.ip

cfg = NewWorkerConfig(
    worker_name="AiSrvHandle",
    father_pid=os.getpid(),
    use_logger=True,
    use_default_monitor=True,
    use_default_alloc=True,
    ip="192.168.0.10"
)

nw = NewWorker(cfg)
nw.start()
"""


@dataclass
class WorkerConfig:
    """这个类专用于传输配置项，并且带有默认值，防止用户遗漏导致程序异常"""

    father_pid: int = 0
    master_adds: str = "127.0.0.1"
    master_port: int = 29500
    network_card: str = "eth1"
    use_logger: bool = True
    use_default_monitor: bool = True  # 使用默认监控（自动使用共享 MonitorProxy）| Use default monitor (auto shared MonitorProxy)
    use_default_alloc: bool = True
    worker_rank: int = 0
    worker_name: str = "default_worker_name"
    worker_self_port: int = 8080
    world_size: int = 1


class Worker(abc.ABC, multiprocessing.Process):
    def __init__(self, worker_config: WorkerConfig) -> None:
        super().__init__()

        self.worker_status = "pending"
        self.worker_name = worker_config.worker_name
        self.worker_start_time = time.time()
        self.worker_pid = -1
        self.worker_self_ip = get_interface_ip(worker_config.network_card)
        self.worker_self_port = worker_config.worker_self_port
        self.father_pid = worker_config.father_pid

        self.worker_config = worker_config

        # 通用日志器, 默认的日志格式, 子类格式更加复杂的话可以在调用基类初始化方法后, 单独执行set_logger_format覆盖
        self.logger: Optional[KaiwuLogger] = None

        # 通用监控上报
        self.monitor_proxy: Optional[MonitorProxy] = None

        # 通用服务发现功能
        self.alloc_proxy: Optional[AllocProxy] = None

        # TODO - 通信模块待引入
        self.communication_tool = None

        # 进程run函数节律控制
        self.process_run_count = 0

        # 终止时可以用以下两种方式之一处理，建议使用第一种_stop_event
        self._stop_event = multiprocessing.Event()
        self.worker_stop_flag = False

        self.set_parameters_from_env_variables(worker_config)

    def set_parameters_from_env_variables(self, worker_config) -> None:
        self.worker_rank = os.environ.get("RANK", worker_config.worker_rank)
        self.world_size = os.environ.get("WORLD_SIZE", worker_config.world_size)
        self.master_adds = os.environ.get("MASTER_ADDR", worker_config.master_adds)
        self.master_port = os.environ.get("MASTER_PORT", worker_config.master_port)

    def set_logger(self, income_logger, override_existing: bool = False) -> None:
        if income_logger is None:
            return

        if self.logger is None or override_existing:
            self.logger = income_logger

    def set_monitor(self, income_monitor, override_existing: bool = False) -> None:
        if income_monitor is None:
            return

        if self.monitor_proxy is None or override_existing:
            self.monitor_proxy = income_monitor

    def _cleanup_resources(self) -> None:
        """在退出前清理资源的钩子，子类可以覆盖，如关闭网络连接、数据库连接等"""
        self.logger.info(f"this is _cleanup_resources")

    def worker_start_up(self) -> bool:
        self.worker_pid = os.getpid()
        """子类必须在run函数一开始调用"""
        if self.worker_config.use_logger and self.logger is None:
            self.logger = KaiwuLogger(CONFIG.svr_name)
            params = {
                "compression": CONFIG.compression,
                "encoding": CONFIG.encoding,
                "rotation": CONFIG.rotation,
                "level": CONFIG.level,
                "serialize": CONFIG.serialize,
                "retention": CONFIG.retention,
                "max_single_message_len": CONFIG.max_single_message_len,
                "max_calls_log_per_min": CONFIG.max_calls_log_per_min,
            }
            # 使用hostname的hash后8位区分不同容器，不依赖hostname格式
            container_id = hashlib.md5(socket.gethostname().encode()).hexdigest()[:8]
            self.logger.set_logger_format(
                f"{CONFIG.log_dir}/{CONFIG.svr_name}/{self.worker_name}_container{container_id}_pid{self.worker_pid}_log_{datetime.now().strftime('%Y-%m-%d-%H')}.log",
                self.worker_name,
                params,
            )

        # MonitorProxy 初始化 | MonitorProxy initialization
        if CONFIG.use_prometheus and self.worker_config.use_default_monitor and self.monitor_proxy is None:
            # 使用全局共享的 MonitorProxy（推荐，节省资源）| Use global shared MonitorProxy (recommended, saves resources)
            self.monitor_proxy = get_monitor_proxy()
            if self.logger:
                monitor_pid = self.monitor_proxy.pid if hasattr(self.monitor_proxy, "pid") else "unknown"
                self.logger.info(
                    f"{self.worker_name} using shared MonitorProxy (PID: {monitor_pid}) {g_not_server_label}"
                )

        if CONFIG.use_alloc and self.worker_config.use_default_alloc and self.alloc_proxy is None:
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()
        return True

    @abc.abstractmethod
    def before_run(self) -> bool:
        """子类必须实现：在 run 循环前执行的初始化逻辑，返回 True 表示成功"""
        self.worker_start_up()

        return True

    def after_run(self) -> bool:
        """子类选择实现"""
        ...

    @abc.abstractmethod
    def run_once(self) -> bool:
        """子类必须实现：run 循环中单次执行的核心逻辑，返回 True 表示成功"""
        ...

    def run(self) -> None:

        if not self.before_run():
            self.logger.error(f"{self.worker_name} before_run failed, so return {g_not_server_label}")
            return

        try:
            while not self._stop_event.is_set():
                try:
                    self.run_once()

                    self.process_run_count += 1
                    if self.process_run_count % CONFIG.idle_sleep_count == 0:
                        time.sleep(CONFIG.idle_sleep_second)

                        self.process_run_count = 0

                except Exception as e:
                    self.logger.exception(f"{self.worker_name} run error: {str(e)}, so return {g_not_server_label}")

        finally:
            self._cleanup_resources()
            if not self.after_run():
                self.logger.error(f"{self.worker_name} after_run failed, so return {g_not_server_label}")
                return

    def stop(self, timeout: Optional[float] = None) -> None:
        """
        外部通过该方法请求停止当前 Worker。
        会设置跨进程的停止信号，并等待子进程退出。
        参数：
          timeout: 最长等待时间（秒），为 None 时无限等待
        """
        # 设置信号：跨进程可见
        self.worker_stop_flag = True
        self._stop_event.set()

        # 等待子进程自然结束
        if self.is_alive():
            self.join(timeout)

        # 若仍未退出，强制终止
        if self.is_alive():
            try:
                self.terminate()
            finally:
                self.join()
