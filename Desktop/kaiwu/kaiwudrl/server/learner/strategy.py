#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from abc import ABC, abstractmethod
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class ITrainerStrategy(ABC):
    """
    预测策略接口
    """

    @abstractmethod
    def before_run(self):
        """
        运行前的策略特定初始化
        """
        pass

    @abstractmethod
    def process_policy_specific(self, model_file_id):
        """
        策略特定的处理逻辑
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        清理时的策略特定操作
        """
        pass

    @abstractmethod
    def periodic_operation(self):
        """
        定时的策略特定操作
        """
        pass

    @abstractmethod
    def train_condition(self, current_size):
        """
        是否满足训练条件
        """
        pass

    @abstractmethod
    def train_stat(self):
        """
        策略特定的统计信息
        """
        pass

    @abstractmethod
    def strategy_name(self):
        """
        策略特定的名字
        """
        pass


def create_strategy(trainer):
    """
    根据配置确定on_policy/off_policy策略
    """
    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
        from kaiwudrl.server.learner.on_policy_strategy import OnPolicyStrategy

        return OnPolicyStrategy(trainer)
    else:
        from kaiwudrl.server.learner.off_policy_strategy import OffPolicyStrategy

        return OffPolicyStrategy(trainer)
