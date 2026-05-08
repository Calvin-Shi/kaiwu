#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.server.common.strategy import IPredictorStrategy
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class OffPolicyStrategy(IPredictorStrategy):
    """
    OffPolicy的策略实现
    """

    def __init__(self, predictor):

        # 传递了predictor对象, 则该类里可以复用, 而不是每次调用都传predictor对象
        self.predictor = predictor

    def before_run(self, context):
        """
        OffPolicy运行前的初始化
        """
        self.predictor.logger.info("OffPolicyStrategy before_run")

    def process_policy_specific(self):
        """
        OffPolicy不需要特殊处理
        """
        pass

    def cleanup(self):
        """
        OffPolicy清理操作
        """
        pass

    def predict_stat(self):
        """
        OffPolicy特定的统计信息
        """
        monitor_data = {
            KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_COST_MS: self.predictor.predict_common_object.get_actor_load_last_model_error_cnt(),
            KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_SUCC_CNT: self.predictor.predict_common_object.get_actor_load_last_model_succ_cnt(),
            KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_ERROR_CNT: self.predictor.predict_common_object.get_actor_load_last_model_error_cnt(),
        }

        if CONFIG.use_prometheus:
            self.predictor.monitor_proxy.put_data({self.predictor.current_pid: monitor_data})

    def strategy_name(self):
        """
        策略特定的名字
        """
        return "off_policy"
