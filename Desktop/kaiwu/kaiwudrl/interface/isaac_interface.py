#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from abc import ABC, abstractmethod


class RobotInterface(ABC):
    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def close(self, *args, **kwargs):
        pass
