#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
跨进程单例模式 (Multiprocess Singleton)

与 threading 单例的区别：
1. 使用 multiprocessing.Lock 而非 threading.Lock
2. 真正的跨进程共享：fork 后子进程共享父进程的单例实例
3. 支持管理 multiprocessing.Process 子进程对象
4. 适用于多进程架构（如多个 Worker 进程共享资源）

使用场景：
- 多个 Worker 进程共享同一个监控代理
- 多个进程共享同一个数据库连接池管理器
- 多个进程共享同一个消息队列管理器
"""

import multiprocessing
import atexit
import os
from typing import Optional, TypeVar, Generic, Type
from abc import ABC, abstractmethod


T = TypeVar("T")


class MultiprocessSingleton(ABC, Generic[T]):
    """
    跨进程单例基类 - 真正的跨进程共享单例

    功能特性：
    1. 跨进程单例：整个应用（主进程+所有子进程）共享同一个资源实例
    2. 自动启动：主进程首次调用时创建和启动资源
    3. 进程复用：子进程 fork 后直接复用主进程的资源，不再重新创建
    4. 资源节省：N 个 Worker 只需 1 个资源实例（而非 N 个）
    5. 线程安全：使用锁保护并发访问
    6. 生命周期管理：支持 atexit 自动清理

    使用方式：
        1. 继承 MultiprocessSingleton
        2. 实现 _create_resource() 方法
        3. 可选：实现 _start_resource() 和 _cleanup_resource() 方法

    Example:
        class MonitorManager(MultiprocessSingleton[MonitorProxy]):
            def _create_resource(self, *args, **kwargs) -> MonitorProxy:
                return MonitorProxy(*args, **kwargs)

            def _start_resource(self, resource: MonitorProxy):
                resource.start()

            def _cleanup_resource(self, resource: MonitorProxy):
                resource.exit_flag.value = True
                resource.join(timeout=5)

        # 使用
        manager = MonitorManager()
        monitor = manager.get_or_create(*args)  # 所有进程共享同一个 MonitorProxy
    """

    _instance: Optional["MultiprocessSingleton"] = None
    _lock = multiprocessing.Lock()
    _resource: Optional[T] = None
    _started: bool = False
    _owned_by_manager: bool = False  # 标记是否由管理器创建（决定是否清理）
    _creator_pid: Optional[int] = None  # 记录创建者的 PID

    def __new__(cls):
        """单例模式实现 - 确保整个应用只有一个管理器实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @abstractmethod
    def _create_resource(self, *args, **kwargs) -> T:
        """
        创建资源实例（子类必须实现）

        Args:
            *args, **kwargs: 传递给资源构造函数的参数

        Returns:
            资源实例
        """
        pass

    def _start_resource(self, resource: T):
        """
        启动资源（子类可选实现）

        默认行为：如果资源有 start() 方法，则调用

        Args:
            resource: 资源实例
        """
        if hasattr(resource, "start") and callable(getattr(resource, "start")):
            resource.start()

    def _cleanup_resource(self, resource: T):
        """
        清理资源（子类可选实现）

        默认行为：
        1. 如果资源有 exit_flag 属性，设置为 True
        2. 如果资源有 join() 方法，等待退出
        3. 如果资源有 terminate() 方法，强制终止

        Args:
            resource: 资源实例
        """
        try:
            # 尝试优雅退出
            if hasattr(resource, "exit_flag"):
                resource.exit_flag.value = True

            if hasattr(resource, "join") and callable(getattr(resource, "join")):
                resource.join(timeout=5)

            # 如果还活着，强制终止
            if hasattr(resource, "is_alive") and callable(getattr(resource, "is_alive")):
                if resource.is_alive():
                    if hasattr(resource, "terminate") and callable(getattr(resource, "terminate")):
                        resource.terminate()
                        if hasattr(resource, "join"):
                            resource.join(timeout=1)
        except Exception:
            pass  # 忽略清理过程中的异常

    def set_external_resource(self, resource: T):
        """
        设置外部创建的资源（由主进程管理）

        使用场景：主进程显式创建资源，多个 Worker 共享

        Args:
            resource: 外部创建的资源实例

        Example:
            # 主进程
            external_resource = create_resource()
            manager.set_external_resource(external_resource)

            # Worker 会自动使用这个外部资源
        """
        with self._lock:
            if self._resource is not None and self._owned_by_manager:
                # 如果已经有自己创建的，先清理
                self._cleanup()

            self._resource = resource
            self._owned_by_manager = False  # 不由管理器管理生命周期
            self._creator_pid = os.getpid()

            # 确保已启动
            if hasattr(resource, "is_alive") and callable(getattr(resource, "is_alive")):
                if not resource.is_alive():
                    self._start_resource(resource)
            self._started = True

    def get_or_create(self, *args, **kwargs) -> T:
        """
        获取或创建资源实例（真正的跨进程共享单例）

        特性：
        - 首次调用时创建资源并启动（通常在主进程）
        - 所有子进程共享同一个资源实例
        - fork 后子进程直接复用主进程的资源，不再重新创建

        Args:
            *args, **kwargs: 传递给 _create_resource() 的参数

        Returns:
            资源实例
        """
        current_pid = os.getpid()

        # ✅ 关键特性：直接复用主进程的资源
        # 如果 _resource 已存在，直接返回（不再检查 PID 或 is_alive）
        if self._resource is not None:
            # ✅ 跨进程共享：直接返回，不做任何检查
            # 资源由主进程管理，子进程只需要使用即可
            return self._resource

        # 创建资源（如果不存在）
        if self._resource is None:
            with self._lock:
                if self._resource is None:
                    self._resource = self._create_resource(*args, **kwargs)
                    self._owned_by_manager = True  # 由管理器创建，负责清理
                    self._creator_pid = current_pid

                    # 注册退出清理（只在创建进程中注册）
                    atexit.register(self._cleanup)

        # 启动资源（如果未启动）
        if not self._started and self._resource is not None:
            with self._lock:
                if not self._started:
                    self._start_resource(self._resource)
                    self._started = True

        return self._resource

    def _cleanup(self):
        """
        清理资源（仅清理自己创建的资源）
        """
        if self._owned_by_manager and self._resource is not None:
            self._cleanup_resource(self._resource)

    @classmethod
    def reset(cls):
        """
        重置管理器（主要用于测试）

        警告：生产环境不要调用此方法！
        """
        if cls._instance is not None:
            cls._instance._cleanup()
            cls._instance = None
            cls._resource = None
            cls._started = False
            cls._owned_by_manager = False
            cls._creator_pid = None

    def get_resource(self) -> Optional[T]:
        """
        获取当前资源实例（不创建）

        Returns:
            资源实例，如果不存在则返回 None
        """
        return self._resource


# ============================================================================
# 便捷装饰器（可选）
# ============================================================================


def multiprocess_singleton(cls: Type[T]) -> Type[T]:
    """
    跨进程单例装饰器（简化版）

    注意：此装饰器提供基本的跨进程单例功能，但不支持生命周期管理。
    如果需要完整的生命周期管理，请使用 MultiprocessSingleton 基类。

    Example:
        @multiprocess_singleton
        class MyResource:
            def __init__(self, value):
                self.value = value

        # 使用
        resource1 = MyResource(100)  # 创建单例
        resource2 = MyResource(200)  # 返回同一个实例，参数被忽略
        assert resource1 is resource2  # True
    """
    _instance_lock = multiprocessing.Lock()
    _instances = {}

    def wrapper(*args, **kwargs):
        if cls not in _instances:
            with _instance_lock:
                if cls not in _instances:
                    _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return wrapper
