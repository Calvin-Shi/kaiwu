#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
单例模式
"""


import threading

# Singleton
class Singleton(object):
    _instance_lock = threading.Lock()

    def __init__(self, cls):
        self._cls = cls
        self._instance = {}

    def __call__(self, *args, **kwargs):
        if self._cls not in self._instance:
            with Singleton._instance_lock:
                if self._cls not in self._instance:
                    # 创建实例时传递参数
                    self._instance[self._cls] = self._cls(*args, **kwargs)
        return self._instance[self._cls]
