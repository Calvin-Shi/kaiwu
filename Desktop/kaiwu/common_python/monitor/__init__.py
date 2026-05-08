#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.monitor.monitor_proxy_process import MonitorProxy
from common_python.monitor.monitor_manager import (
    GlobalMonitorManager,
    get_monitor_proxy,
    set_monitor_proxy,
)
from common_python.monitor.process_health_monitor import (
    ProcessHealthMonitor,
    ProcessInfo,
    ThreadInfo,
    ProcessExitReason,
)

__all__ = [
    "MonitorProxy",  # 进程版本（默认）| Process version (default)
    "GlobalMonitorManager",
    "get_monitor_proxy",
    "set_monitor_proxy",
    "ProcessHealthMonitor",  # 进程和线程健康监控器
    "ProcessInfo",  # 进程信息
    "ThreadInfo",  # 线程信息
    "ProcessExitReason",  # 进程/线程退出原因枚举
]
