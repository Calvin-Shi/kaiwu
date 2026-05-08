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


class IPredictorStrategy(ABC):
    """
    预测策略接口, 由于存在下面三种情况来调用, 故提取到common目录下
    1. 大规模场景, 即训练, 利用, 加载模型都在actor侧
    2. 中规模场景, 即训练, 利用, 加载模型都在aisrv的预测进程predictor_local侧
    3. 小规模场景, 即训练, 利用, 加载模型都在aisrv的workflow进程侧
    """

    @abstractmethod
    def before_run(self):
        """
        运行前的策略特定初始化
        """
        pass

    @abstractmethod
    def process_policy_specific(self):
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
    def predict_stat(self):
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


def create_strategy(predictor):
    """
    根据配置确定on_policy/off_policy策略
    """
    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
        from kaiwudrl.server.common.on_policy_strategy import OnPolicyStrategy

        return OnPolicyStrategy(predictor)
    else:
        from kaiwudrl.server.common.off_policy_strategy import OffPolicyStrategy

        return OffPolicyStrategy(predictor)
