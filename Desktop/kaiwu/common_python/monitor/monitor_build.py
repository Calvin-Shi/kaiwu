#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.config.config_control import CONFIG
from common_python.monitor.monitor_proxy_process import MonitorProxy

"""
主要是考虑到有多个进程调用, 但是只是需要初始化monitor_proxy一次
"""


class MonitorBuilder:
    def __init__(self) -> None:
        self.monitor_proxy = None

    def build(self):
        pass
