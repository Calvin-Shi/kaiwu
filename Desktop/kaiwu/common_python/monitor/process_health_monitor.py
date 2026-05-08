#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

通用进程和线程健康监控模块
用于监控各子进程(actor_proxy, learner_proxy, workflow等)和线程(kaiwu_rl_helper, actor_receiver等)的健康状态
支持跨容器的进程监控

适用场景:
- aisrv监控其子进程(workflow, actor_proxy, learner_proxy)和线程(kaiwu_rl_helper)
- actor监控其子进程和线程(actor_receiver, actor_sender)
- learner监控其子进程和线程
- 任何需要监控子进程/线程健康状态的场景
"""

import time
import threading
import psutil
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Union
from common_python.logging.kaiwu_logger import KaiwuLogger


class ProcessExitReason(Enum):
    """进程/线程退出原因枚举"""

    UNKNOWN = "unknown"
    OOM_KILLED = "oom_killed"  # 被OOM killer杀死(仅进程)
    NORMAL_EXIT = "normal_exit"  # 正常退出
    SIGNAL_KILLED = "signal_killed"  # 被信号杀死(仅进程)
    STILL_ALIVE = "still_alive"  # 进程/线程仍在运行


@dataclass
class ProcessInfo:
    """进程信息"""

    pid: int
    name: str  # 进程名称,如"actor_proxy_0", "learner_proxy_1", "workflow_2"
    process_type: str  # 进程类型,如"actor_proxy", "learner_proxy", "workflow"
    start_time: float  # 进程启动时间戳
    last_check_time: float = 0  # 上次检查时间
    is_alive: bool = True  # 是否存活
    exit_reason: ProcessExitReason = ProcessExitReason.STILL_ALIVE  # 退出原因
    exit_code: Optional[int] = None  # 退出码
    container_id: Optional[str] = None  # 容器ID(如果在容器中)


@dataclass
class ThreadInfo:
    """线程信息"""

    thread: threading.Thread  # 线程对象
    name: str  # 线程名称,如"kaiwu_rl_helper_0", "actor_receiver", "actor_sender"
    thread_type: str  # 线程类型,如"kaiwu_rl_helper", "actor_receiver", "actor_sender"
    start_time: float  # 线程启动时间戳
    last_check_time: float = 0  # 上次检查时间
    is_alive: bool = True  # 是否存活
    exit_reason: ProcessExitReason = ProcessExitReason.STILL_ALIVE  # 退出原因


class ProcessHealthMonitor:
    """
    通用进程和线程健康监控器

    功能:
    1. 注册需要监控的进程和线程
    2. 周期性检查进程/线程健康状态(调用check_once)
    3. 检测OOM等异常退出情况(进程)
    4. 检测线程死锁或异常退出
    5. 触发告警回调

    使用示例:

    # 在aisrv中使用
    monitor = ProcessHealthMonitor(logger=self.logger)
    monitor.register_alert_callback(self.on_process_exit_alert)

    # 注册子进程
    workflow_process.start()
    monitor.register_process(
        pid=workflow_process.pid,
        name="workflow_0",
        process_type="workflow"
    )

    # 注册线程
    helper_thread = KaiWuRLStandardHelper(simu_ctx)
    helper_thread.start()
    monitor.register_thread(
        thread=helper_thread,
        name="kaiwu_rl_helper_0",
        thread_type="kaiwu_rl_helper"
    )

    # 在主循环中定期检查
    def run_once(self):
        monitor.check_once()

    # 在actor中使用
    monitor = ProcessHealthMonitor(logger=logger)

    # 注册线程
    receiver_thread = threading.Thread(target=self.actor_receive_msg)
    receiver_thread.start()
    monitor.register_thread(
        thread=receiver_thread,
        name="actor_receiver",
        thread_type="actor_receiver"
    )
    """

    def __init__(self, logger: Optional[KaiwuLogger] = None, check_interval: int = 10, log_tag: str = ""):
        """
        初始化监控器

        Args:
            logger: 日志对象,如果不提供则创建新的KaiwuLogger
            check_interval: 检查间隔(秒),建议10-30秒
            log_tag: 日志标签,用于日志filter匹配(如"ai_server"),为空则不加前缀
        """
        self.logger = logger if logger else KaiwuLogger()
        self.check_interval = check_interval
        self.log_tag = log_tag

        # 进程信息字典: {pid: ProcessInfo}
        self.processes: Dict[int, ProcessInfo] = {}

        # 线程信息字典: {thread_id: ThreadInfo}
        self.threads: Dict[int, ThreadInfo] = {}

        # 告警回调列表: [(callback_func, monitor_types)]
        # callback_func(info: Union[ProcessInfo, ThreadInfo]) -> None
        self.alert_callbacks: List[tuple] = []

        # 统计信息
        self.stats = {
            "total_process_registered": 0,
            "total_thread_registered": 0,
            "total_oom_killed": 0,
            "total_abnormal_exit": 0,
        }

    def _tag_msg(self, msg: str) -> str:
        """为日志消息添加log_tag前缀,用于匹配logger的filter"""
        return f"{self.log_tag} {msg}" if self.log_tag else msg

    def register_process(
        self,
        pid: int,
        name: str,
        process_type: str,
        container_id: Optional[str] = None,
    ):
        """
        注册需要监控的进程

        Args:
            pid: 进程ID
            name: 进程名称(建议格式: "{process_type}_{index}")
            process_type: 进程类型(如"actor_proxy", "learner_proxy", "workflow", "kaiwu_rl_helper")
            container_id: 容器ID(可选)

        示例:
            monitor.register_process(
                pid=worker.pid,
                name="actor_proxy_0",
                process_type="actor_proxy"
            )
        """
        if pid in self.processes:
            self.logger.warning(self._tag_msg(f"ProcessHealthMonitor process {name}(pid={pid}) already registered"))
            return

        process_info = ProcessInfo(
            pid=pid,
            name=name,
            process_type=process_type,
            start_time=time.time(),
            last_check_time=time.time(),
            container_id=container_id,
        )

        self.processes[pid] = process_info
        self.stats["total_process_registered"] += 1

        self.logger.info(
            self._tag_msg(f"ProcessHealthMonitor registered process: {name}(pid={pid}, type={process_type})")
        )

    def register_thread(
        self,
        thread: threading.Thread,
        name: str,
        thread_type: str,
        auto_start: bool = False,
    ):
        """
        注册需要监控的线程

        Args:
            thread: 线程对象(threading.Thread实例)
            name: 线程名称(建议格式: "{thread_type}_{index}")
            thread_type: 线程类型(如"kaiwu_rl_helper", "actor_receiver", "actor_sender")
            auto_start: 如果线程未启动,是否自动启动(默认False)

        示例:
            # 方式1: 先启动再注册
            helper = KaiWuRLStandardHelper(simu_ctx)
            helper.start()
            monitor.register_thread(
                thread=helper,
                name="kaiwu_rl_helper_0",
                thread_type="kaiwu_rl_helper"
            )

            # 方式2: 注册时自动启动
            receiver = threading.Thread(target=self.receive_msg)
            monitor.register_thread(
                thread=receiver,
                name="actor_receiver",
                thread_type="actor_receiver",
                auto_start=True,
            )
        """
        # 如果线程未启动且auto_start=True,则启动线程
        if not thread.is_alive() and auto_start:
            thread.start()

        # 获取线程ID
        thread_id = thread.ident
        if thread_id is None:
            # 线程未启动
            self.logger.warning(self._tag_msg(f"ProcessHealthMonitor thread {name} not started, cannot register"))
            return

        if thread_id in self.threads:
            self.logger.warning(
                self._tag_msg(f"ProcessHealthMonitor thread {name}(tid={thread_id}) already registered")
            )
            return

        thread_info = ThreadInfo(
            thread=thread,
            name=name,
            thread_type=thread_type,
            start_time=time.time(),
            last_check_time=time.time(),
        )

        self.threads[thread_id] = thread_info
        self.stats["total_thread_registered"] += 1

        self.logger.info(
            self._tag_msg(f"ProcessHealthMonitor registered thread: {name}(tid={thread_id}, type={thread_type})"),
        )

    def unregister_process(self, pid: int):
        """
        取消注册进程(正常退出时调用)

        Args:
            pid: 进程ID

        注意:
            - 正常退出时应该调用此方法,避免误报
            - 异常退出会被自动检测,不需要手动调用
        """
        if pid in self.processes:
            process_info = self.processes.pop(pid)
            self.logger.info(
                self._tag_msg(f"ProcessHealthMonitor unregistered process: {process_info.name}(pid={pid})")
            )

    def unregister_thread(self, thread_id: int):
        """
        取消注册线程(正常退出时调用)

        Args:
            thread_id: 线程ID(thread.ident)

        注意:
            - 正常退出时应该调用此方法,避免误报
            - 异常退出会被自动检测,不需要手动调用
        """
        if thread_id in self.threads:
            thread_info = self.threads.pop(thread_id)
            self.logger.info(
                self._tag_msg(f"ProcessHealthMonitor unregistered thread: {thread_info.name}(tid={thread_id})")
            )

    def register_alert_callback(self, callback: Callable, monitor_types: Optional[List[str]] = None):
        """
        注册告警回调函数

        Args:
            callback: 回调函数 callback(info: Union[ProcessInfo, ThreadInfo]) -> None
            monitor_types: 监控的进程/线程类型列表,None表示监控所有类型

        示例:
            def on_exit_alert(info):
                if isinstance(info, ProcessInfo):
                    if info.exit_reason == ProcessExitReason.OOM_KILLED:
                        send_alert(f"Process {info.name} OOMed!")
                elif isinstance(info, ThreadInfo):
                    send_alert(f"Thread {info.name} died!")

            monitor.register_alert_callback(
                callback=on_exit_alert,
                monitor_types=["learner_proxy", "actor_proxy", "kaiwu_rl_helper"]
            )
        """
        self.alert_callbacks.append((callback, monitor_types))
        self.logger.info(
            self._tag_msg(f"ProcessHealthMonitor registered alert callback for types: {monitor_types or 'ALL'}")
        )

    def is_process_alive(self, pid: int) -> bool:
        """
        判断进程是否存活

        Args:
            pid: 进程ID

        Returns:
            bool: True表示存活,False表示已退出

        注意:
            - 这是一个轻量级的检查,不会触发告警
            - 如果需要自动告警,请使用check_once()
        """
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
        except Exception as e:
            self.logger.warning(self._tag_msg(f"ProcessHealthMonitor failed to check if process {pid} is alive: {e}"))
            return False

    def check_process_exit_reason(self, pid: int, exit_code: Optional[int] = None) -> ProcessExitReason:
        """
        检查进程退出原因

        判断逻辑:
        1. 根据退出码判断: 正数为正常退出, 负数为被信号杀死
        2. 只有被SIGKILL(-9)杀死时, 才进一步检查是否为OOM killer所为
        3. OOM验证: 检查dmesg日志中是否有该pid的OOM记录

        Args:
            pid: 进程ID
            exit_code: 进程退出码(负数表示被信号杀死, 如-9表示SIGKILL)

        Returns:
            ProcessExitReason: 退出原因
        """
        try:
            if exit_code is not None:
                if exit_code == 0:
                    return ProcessExitReason.NORMAL_EXIT
                elif exit_code > 0:
                    # 正数退出码表示进程自己退出(非零表示异常)
                    return ProcessExitReason.UNKNOWN
                else:
                    # 负数退出码表示被信号杀死, 信号值为 -exit_code
                    signal_num = -exit_code
                    if signal_num == 9:
                        # SIGKILL: 可能是OOM killer, 也可能是手动kill -9
                        # 需要进一步检查dmesg确认是否为OOM
                        if self._check_oom_by_dmesg(pid):
                            return ProcessExitReason.OOM_KILLED
                        else:
                            return ProcessExitReason.SIGNAL_KILLED
                    else:
                        # 其他信号(如SIGTERM=15, SIGINT=2等)
                        return ProcessExitReason.SIGNAL_KILLED
            else:
                # 没有退出码, 无法准确判断
                return ProcessExitReason.UNKNOWN

        except Exception as e:
            self.logger.warning(self._tag_msg(f"ProcessHealthMonitor failed to check exit reason for pid {pid}: {e}"))

        return ProcessExitReason.UNKNOWN

    def _check_oom_by_dmesg(self, pid: int) -> bool:
        """
        通过dmesg日志检查进程是否被OOM killer杀死

        Args:
            pid: 进程ID

        Returns:
            bool: 是否被OOM killer杀死
        """
        try:
            import subprocess

            result = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout:
                # 只检查最近500行, 查找该pid的OOM记录
                for line in result.stdout.split("\n")[-500:]:
                    if f"pid {pid}" in line and ("Killed process" in line or "Out of memory" in line):
                        return True
        except (FileNotFoundError, PermissionError, Exception):
            pass

        return False

    def check_once(self):
        """
        执行一次健康检查

        检查所有注册的进程和线程:
        1. 检查进程是否存活
        2. 检查线程是否存活
        3. 如果进程退出,判断退出原因
        4. 如果线程退出,记录退出信息
        5. 触发告警回调

        注意:
            - 应该在主循环中定期调用此方法
            - 检查间隔由check_interval控制
            - 建议每次loop都调用,内部会根据check_interval判断是否真正检查

        示例:
            def run_once(self):
                # 其他业务逻辑...
                self.process_health_monitor.check_once()
        """
        current_time = time.time()

        # 检查进程
        self._check_processes(current_time)

        # 检查线程
        self._check_threads(current_time)

    def _check_processes(self, current_time: float):
        """检查所有进程的健康状态"""
        dead_pids = []

        for pid, process_info in self.processes.items():
            # 检查是否到达检查间隔
            if current_time - process_info.last_check_time < self.check_interval:
                continue

            process_info.last_check_time = current_time

            # 检查进程是否存活
            try:
                p = psutil.Process(pid)
                is_running = p.is_running() and p.status() != psutil.STATUS_ZOMBIE

                if not is_running:
                    # 进程已退出
                    process_info.is_alive = False

                    # 先获取退出码
                    try:
                        process_info.exit_code = p.wait(timeout=0.1)
                    except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                        process_info.exit_code = None

                    # 根据退出码判断退出原因
                    exit_reason = self.check_process_exit_reason(pid, process_info.exit_code)
                    process_info.exit_reason = exit_reason

                    # 更新统计信息
                    if exit_reason == ProcessExitReason.OOM_KILLED:
                        self.stats["total_oom_killed"] += 1
                    else:
                        self.stats["total_abnormal_exit"] += 1

                    # 打印告警日志
                    self._log_process_exit(process_info)

                    # 触发告警回调
                    self._trigger_alert_callbacks(process_info)

                    dead_pids.append(pid)

            except psutil.NoSuchProcess:
                # 进程不存在, 无法获取退出码
                process_info.is_alive = False
                process_info.exit_code = None
                process_info.exit_reason = self.check_process_exit_reason(pid, None)

                if process_info.exit_reason == ProcessExitReason.OOM_KILLED:
                    self.stats["total_oom_killed"] += 1
                else:
                    self.stats["total_abnormal_exit"] += 1

                self._log_process_exit(process_info)
                self._trigger_alert_callbacks(process_info)
                dead_pids.append(pid)

            except Exception as e:
                self.logger.error(
                    self._tag_msg(f"ProcessHealthMonitor failed to check process {process_info.name}(pid={pid}): {e}"),
                )

        # 清理已退出的进程
        for pid in dead_pids:
            del self.processes[pid]

    def _check_threads(self, current_time: float):
        """检查所有线程的健康状态"""
        dead_thread_ids = []

        for thread_id, thread_info in self.threads.items():
            # 检查是否到达检查间隔
            if current_time - thread_info.last_check_time < self.check_interval:
                continue

            thread_info.last_check_time = current_time

            # 检查线程是否存活
            try:
                is_alive = thread_info.thread.is_alive()

                if not is_alive:
                    # 线程已退出
                    thread_info.is_alive = False
                    thread_info.exit_reason = ProcessExitReason.UNKNOWN

                    # 更新统计信息
                    self.stats["total_abnormal_exit"] += 1

                    # 打印告警日志
                    self._log_thread_exit(thread_info)

                    # 触发告警回调
                    self._trigger_alert_callbacks(thread_info)

                    dead_thread_ids.append(thread_id)

            except Exception as e:
                self.logger.error(
                    self._tag_msg(
                        f"ProcessHealthMonitor failed to check thread {thread_info.name}(tid={thread_id}): {e}"
                    ),
                )

        # 清理已退出的线程
        for thread_id in dead_thread_ids:
            del self.threads[thread_id]

    def _log_process_exit(self, process_info: ProcessInfo):
        """打印进程退出日志"""
        runtime = time.time() - process_info.start_time

        tag_prefix = f"{self.log_tag} " if self.log_tag else ""
        log_msg = (
            f"{tag_prefix}🔴 ProcessHealthMonitor detected process exit: "
            f"name={process_info.name}, "
            f"pid={process_info.pid}, "
            f"type={process_info.process_type}, "
            f"exit_reason={process_info.exit_reason.value}, "
            f"exit_code={process_info.exit_code}, "
            f"runtime={runtime:.1f}s"
        )

        if process_info.container_id:
            log_msg += f", container_id={process_info.container_id}"

        # OOM使用ERROR级别,其他使用WARNING级别
        if process_info.exit_reason == ProcessExitReason.OOM_KILLED:
            self.logger.error(log_msg)
        else:
            self.logger.warning(log_msg)

    def _log_thread_exit(self, thread_info: ThreadInfo):
        """打印线程退出日志"""
        runtime = time.time() - thread_info.start_time

        tag_prefix = f"{self.log_tag} " if self.log_tag else ""
        log_msg = (
            f"{tag_prefix}🔴 ProcessHealthMonitor detected thread exit: "
            f"name={thread_info.name}, "
            f"type={thread_info.thread_type}, "
            f"exit_reason={thread_info.exit_reason.value}, "
            f"runtime={runtime:.1f}s"
        )

        self.logger.warning(log_msg)

    def _trigger_alert_callbacks(self, info: Union[ProcessInfo, ThreadInfo]):
        """触发告警回调"""
        # 获取类型字段
        if isinstance(info, ProcessInfo):
            info_type = info.process_type
        else:  # ThreadInfo
            info_type = info.thread_type

        for callback, monitor_types in self.alert_callbacks:
            # 检查类型是否匹配
            if monitor_types is None or info_type in monitor_types:
                try:
                    callback(info)
                except Exception as e:
                    self.logger.error(self._tag_msg(f"ProcessHealthMonitor alert callback failed: {e}"))

    def get_stats(self) -> Dict:
        """
        获取统计信息

        Returns:
            Dict: {
                "total_process_registered": 总注册进程数,
                "total_thread_registered": 总注册线程数,
                "total_oom_killed": OOM杀死数,
                "total_abnormal_exit": 异常退出数,
                "current_monitored_processes": 当前监控进程数,
                "current_monitored_threads": 当前监控线程数
            }
        """
        return {
            **self.stats,
            "current_monitored_processes": len(self.processes),
            "current_monitored_threads": len(self.threads),
        }

    def get_alive_processes(self, process_type: Optional[str] = None) -> List[ProcessInfo]:
        """
        获取存活的进程列表

        Args:
            process_type: 进程类型过滤,None表示所有类型

        Returns:
            存活的进程信息列表

        示例:
            # 获取所有存活的learner_proxy进程
            learner_processes = monitor.get_alive_processes("learner_proxy")
            for p in learner_processes:
                print(f"Learner {p.name} is running (pid={p.pid})")
        """
        result = []
        for process_info in self.processes.values():
            if process_info.is_alive:
                if process_type is None or process_info.process_type == process_type:
                    result.append(process_info)
        return result

    def get_alive_threads(self, thread_type: Optional[str] = None) -> List[ThreadInfo]:
        """
        获取存活的线程列表

        Args:
            thread_type: 线程类型过滤,None表示所有类型

        Returns:
            存活的线程信息列表

        示例:
            # 获取所有存活的kaiwu_rl_helper线程
            helper_threads = monitor.get_alive_threads("kaiwu_rl_helper")
            for t in helper_threads:
                print(f"Helper {t.name} is running")
        """
        result = []
        for thread_info in self.threads.values():
            if thread_info.is_alive:
                if thread_type is None or thread_info.thread_type == thread_type:
                    result.append(thread_info)
        return result
