#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.server.learner.strategy import ITrainerStrategy
from kaiwudrl.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper


class OffPolicyStrategy(ITrainerStrategy):
    """
    OffPolicy的策略实现
    """

    def __init__(self, trainer):
        self.current_sync_model_version_from_learner = -1

        # 传递了trainer对象, 则该类里可以复用, 而不是每次调用都传trainer对象
        self.trainer = trainer

    def before_run(self):
        """
        OffPolicy运行前的初始化
        """
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            self.trainer.model_file_sync_wrapper = ModelFileSyncWrapper()
            self.trainer.model_file_sync_wrapper.init()

            # 只有在训练时才进行
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                # off-policy场景下, 训练进程主动推送下第一个model文件到modelpool, 预加载和非预加载情况下都适合
                push_model_success = (
                    self.trainer.model_file_sync_wrapper.ckpt_sync_warper.push_checkpoint_to_model_pool(
                        self.trainer.logger
                    )
                )
                if push_model_success:
                    self.trainer.logger.info(f"train first model file push to modelpool success")
                else:
                    self.trainer.logger.error(f"train first model file push to modelpool failed")

        self.trainer.logger.info("OffPolicyStrategy before_run")

    def get_current_sync_model_version_from_learner(self):
        return self.current_sync_model_version_from_learner

    def process_policy_specific(self, model_file_id):
        """
        OffPolicy不需要特殊处理
        """
        pass

    def cleanup(self):
        """
        OffPolicy清理操作
        """
        # 非on-policy的才需要主动关闭self.model_file_sync_wrapper
        self.trainer.model_file_saver.stop()
        self.trainer.logger.info("train self.agent_wrapper.close success")

        self.trainer.model_file_sync_wrapper.stop()
        self.trainer.logger.info("train self.model_file_sync_wrapper.stop success")

    def train_stat(self):
        """
        OffPolicy特定的统计信息
        """
        monitor_data = {}

        return monitor_data

    def periodic_operation(self):
        """
        定时的策略特定操作
        """
        pass

    def train_condition(self, current_size):
        """
        是否满足训练条件
        """

        """
        这里需要区分下:
        1. 如果是off-policy, 只是需要判断current_size大于batch_size, 当learner在读取reverb样本时, 阻塞了aisrv也能给learner发送样本
        """

        condition = current_size >= int(CONFIG.replay_buffer_capacity * CONFIG.preload_ratio)

        return condition

    def strategy_name(self):
        """
        策略特定的名字
        """
        return "off_policy"
