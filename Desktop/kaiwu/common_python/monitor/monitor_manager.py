#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
全局 MonitorProxy 管理器
支持多个 Worker 共享同一个 MonitorProxy 进程，节省资源
"""

import multiprocessing
import atexit
import os
from typing import Optional
from common_python.monitor.monitor_proxy_process import MonitorProxy
from common_python.logging.kaiwu_logger import KaiwuLogger
from common_python.utils.multiprocess_singleton import MultiprocessSingleton

# ============================================================================
# 跨进程共享资源（模块级别，使用 Value 和共享内存）
# 支持 spawn 模式的独立进程
# ============================================================================
# 使用共享内存标记是否已创建 MonitorProxy
_monitor_created = multiprocessing.Value("i", 0)  # 0=未创建, 1=已创建
_monitor_pid = multiprocessing.Value("i", -1)  # MonitorProxy 的 PID
_external_monitor = multiprocessing.Value("i", 0)  # 0=内部创建, 1=外部注入

# 使用进程锁保护并发访问
_process_lock = multiprocessing.Lock()

# 延迟创建 Manager（在需要时才创建）
_manager = None
_shared_queue = None


def _get_shared_queue():
    """延迟创建共享队列（避免 fork 后的连接问题）"""
    global _manager, _shared_queue
    if _shared_queue is None:
        _manager = multiprocessing.Manager()
        _shared_queue = _manager.Queue()
    return _shared_queue


class GlobalMonitorManager(MultiprocessSingleton[MonitorProxy]):
    """
    全局 MonitorProxy 管理器 - 继承 MultiprocessSingleton 实现跨进程单例

    使用 MultiprocessSingleton 基类 + 共享内存实现跨进程共享，适用于：
    - fork 模式：子进程继承父进程资源（通过 MultiprocessSingleton._resource）
    - spawn 模式：独立启动的新进程（通过 _monitor_created 等共享内存标记）

    功能特性：
    1. 跨进程单例：整个进程族共享同一个 MonitorProxy 进程
    2. 自动启动：第一个调用的进程创建和启动 MonitorProxy
    3. 进程复用：后续进程检测到已创建则复用共享队列
    4. 资源节省：N 个 Worker 只需 1 个 MonitorProxy 进程（而非 N 个）
    5. 线程安全：使用跨进程锁保护并发访问

    使用示例：
        # 任意进程（主进程或子进程）
        monitor = get_monitor_proxy()
        monitor.put_data({"metric": 100})

        # 多个独立进程示例（如 actor 的 predictor）
        for i in range(CONFIG.actor_predict_process_num):
            predictor = Predictor()
            predictor.start()  # ✅ 只有第一个会创建 MonitorProxy
    """

    def _create_resource(self, file_path=None, section=None) -> MonitorProxy:
        """
        创建 MonitorProxy 实例（实现 MultiprocessSingleton 抽象方法）

        Args:
            file_path: 配置文件路径 | Config file path
            section: 配置文件节 | Config section

        Returns:
            MonitorProxy 实例
        """
        return MonitorProxy(file_path, section)

    def _start_resource(self, resource: MonitorProxy):
        """
        启动 MonitorProxy 进程（覆盖父类方法）

        Args:
            resource: MonitorProxy 实例
        """
        # 使用共享队列
        resource.msg_queue = _get_shared_queue()
        resource.start()

        # 保存 PID 到跨进程共享变量（用于 spawn 模式检测）
        if resource.pid:
            _monitor_pid.value = resource.pid
            _monitor_created.value = 1

    def _cleanup_resource(self, resource: MonitorProxy):
        """
        清理 MonitorProxy 资源（覆盖父类方法）

        清理策略：
        1. 设置退出标志
        2. 等待 5 秒让进程正常退出
        3. 超时则强制终止

        Args:
            resource: MonitorProxy 实例
        """
        if resource.is_alive():
            try:
                resource.exit_flag.value = True
                resource.join(timeout=5)

                if resource.is_alive():
                    resource.terminate()
                    resource.join(timeout=1)
            except Exception:
                pass  # 忽略清理过程中的异常

        # 清理共享内存标记
        _monitor_created.value = 0
        _monitor_pid.value = -1
        _external_monitor.value = 0

    def get_or_create_monitor(self, file_path=None, section=None) -> MonitorProxy:
        """
        获取或创建全局 MonitorProxy 实例（真正的跨进程共享单例）

        结合 MultiprocessSingleton 和共享内存实现跨进程单例：
        - fork 模式：通过 MultiprocessSingleton._resource 继承
        - spawn 模式：通过 _monitor_created 等共享内存标记检测

        特性：
        - 第一个调用的进程创建 MonitorProxy 并启动
        - 后续进程检测到已创建则复用共享队列
        - 适用于 fork 和 spawn 两种进程启动模式

        Args:
            file_path: 配置文件路径 | Config file path
            section: 配置文件节 | Config section

        Returns:
            MonitorProxy 实例 | MonitorProxy instance
        """
        with _process_lock:
            # 检查是否已经通过 set_external_monitor 设置了外部 MonitorProxy
            if _external_monitor.value == 1:
                current_pid = _monitor_pid.value

                # 验证外部 MonitorProxy 进程是否存在
                try:
                    os.kill(current_pid, 0)
                except (OSError, TypeError):
                    # 外部 MonitorProxy 进程不存在，清理状态
                    _monitor_created.value = 0
                    _external_monitor.value = 0
                    _monitor_pid.value = -1
                    return self.get_or_create_monitor(file_path, section)

                # 创建本地 MonitorProxy 对象（复用共享队列）
                monitor = MonitorProxy(file_path, section)
                monitor.msg_queue = _get_shared_queue()
                return monitor

            # fork 模式：优先使用父类的 _resource（会被 fork 继承）
            if self._resource is not None:
                return self._resource

            # spawn 模式：检查共享内存标记
            if _monitor_created.value == 1:
                current_pid = _monitor_pid.value

                # 验证 MonitorProxy 进程是否存在
                try:
                    os.kill(current_pid, 0)
                except (OSError, TypeError):
                    # MonitorProxy 进程不存在，清理状态并重新创建
                    _monitor_created.value = 0
                    _monitor_pid.value = -1
                    return self.get_or_create_monitor(file_path, section)

                # 创建本地 MonitorProxy 对象（复用共享队列）
                monitor = MonitorProxy(file_path, section)
                monitor.msg_queue = _get_shared_queue()
                return monitor

            # 都没有：调用父类方法创建新的 MonitorProxy
            return self.get_or_create(file_path, section)

    def set_external_monitor(self, monitor: MonitorProxy):
        """
        设置外部创建的 MonitorProxy（由主进程管理）

        使用场景：主进程显式创建 MonitorProxy，多个 Worker 共享
        委托给父类的 set_external_resource 方法，同时更新共享内存标记

        Args:
            monitor: 外部创建的 MonitorProxy 实例

        Example:
            # 主进程
            global_monitor = MonitorProxy()
            global_monitor.start()
            set_monitor_proxy(global_monitor)

            # Worker 会自动使用这个外部 MonitorProxy
        """
        with _process_lock:
            if not monitor.is_alive():
                raise ValueError("External monitor must be started before setting")

            # 调用父类方法设置外部资源
            super().set_external_resource(monitor)

            # 更新共享内存标记（支持 spawn 模式）
            global _shared_queue
            _shared_queue = monitor.msg_queue
            _monitor_created.value = 1
            _monitor_pid.value = monitor.pid
            _external_monitor.value = 1

    @classmethod
    def reset(cls):
        """
        重置管理器（主要用于测试）

        警告：生产环境不要调用此方法！
        """
        # 清理共享内存标记
        _monitor_created.value = 0
        _monitor_pid.value = -1
        _external_monitor.value = 0

        # 调用父类重置
        super().reset()


# ============================================================================
# 便捷函数（推荐使用）
# ============================================================================


def get_monitor_proxy(file_path=None, section=None) -> MonitorProxy:
    """
    获取全局共享的 MonitorProxy - 真正的跨进程单例（推荐使用）

    特点：
    - 第一个调用的进程创建和启动 MonitorProxy
    - 所有进程（包括独立启动的子进程）共享同一个 MonitorProxy
    - 适用于 fork 和 spawn 两种进程启动模式
    - 自动资源管理

    Args:
        file_path: 配置文件路径 | Config file path
        section: 配置文件节 | Config section

    Returns:
        MonitorProxy 实例 | MonitorProxy instance

    Example:
        # 在任意进程中使用（自动共享）
        self.monitor_proxy = get_monitor_proxy()
        self.monitor_proxy.put_data(monitor_data)

        # 多个独立进程示例（如 actor 的 predictor）
        for i in range(CONFIG.actor_predict_process_num):
            predictor = Predictor()
            predictor.start()  # 所有 predictor 共享同一个 MonitorProxy
    """
    return GlobalMonitorManager().get_or_create_monitor(file_path, section)


def set_monitor_proxy(monitor: MonitorProxy):
    """
    设置外部创建的 MonitorProxy

    使用场景：主进程显式管理 MonitorProxy 生命周期

    Args:
        monitor: 外部创建的 MonitorProxy 实例

    Example:
        # 主进程
        global_monitor = MonitorProxy()
        global_monitor.start()
        set_monitor_proxy(global_monitor)

        # 创建 Workers（会自动使用外部 MonitorProxy）
        workers = [MyWorker(...) for i in range(3)]
        for w in workers:
            w.start()
    """
    GlobalMonitorManager().set_external_monitor(monitor)
