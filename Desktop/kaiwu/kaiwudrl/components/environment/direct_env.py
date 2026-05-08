#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
import importlib
from kaiwudrl.components.environment.env_wrapper import process_error_env_obs


class DirectEnv:
    def __init__(self, logger, monitor_proxy) -> None:
        self.logger = logger
        self.monitor_proxy = monitor_proxy

        # 根据配置获取到具体的对象
        module = importlib.import_module(CONFIG.env_business_module_name)
        cls = getattr(module, CONFIG.env_business_class_name)
        # 实例化业务对象
        self.env = cls()

        self.logger.info(f"direct_env start success, env is {self.env}")

    def init(self):
        self.env.init()

    def reset(self, usr_conf):
        """
        处理reset的请求和响应
        """
        if usr_conf is None:
            self.logger.error(f"direct_env usr_conf is None, please check")
            return process_error_env_obs(-1, f"usr_conf is None, please check")

        # 是否是评估是框架侧知晓的, 故这里设置进去
        usr_conf["is_eval"] = CONFIG.run_mode in [KaiwuDRLDefine.RUN_MODE_EVAL, KaiwuDRLDefine.RUN_MODE_EXAM]
        game_id = f'{os.getenv("KAIWU_TASK_ID", "1")}_{os.getenv("KAIWU_ROUND_INDEX", "1")}_1'
        usr_conf["game_id"] = game_id

        try:
            result = self.env.reset(usr_conf)
            if result is None:
                self.logger.error(f"direct_env self.env.reset return None, please check")
                return process_error_env_obs(-2, f"self.env.reset return None, please check")

            return result

        except Exception as e:
            self.logger.error(f"direct_env failed to get env_info during reset, error: {str(e)}")
            return process_error_env_obs(-3, f"failed to get env_info during reset, error: {str(e)}")

    def step(self, action_data):
        """
        处理step的请求和响应
        """
        if action_data is None:
            self.logger.error(f"direct_env action_data is None, please check")
            return process_error_env_obs(-4, f"action_data is None, please check")

        try:
            result = self.env.step(action_data)
            if result is None:
                self.logger.error(f"direct_env self.env.step return None, please check")
                return process_error_env_obs(-5, f"self.env.step return None, please check")

            return result
        except Exception as e:
            self.logger.error(f"direct_env failed to get env_info during step, error: {str(e)}")
            return process_error_env_obs(-6, f"failed to get env_info during step, error: {str(e)}")

    def apply_function(self, func_name, *args, **kwargs):
        """
        直接调用业务的实现
        """
        # 检查环境是否有这个函数
        try:
            if hasattr(self.env, func_name) and callable(getattr(self.env, func_name)):
                # 获取函数并调用
                func = getattr(self.env, func_name)
                result = func(*args, **kwargs)
                return result
            else:
                # 函数不存在
                self.logger.error(f"direct_env function {func_name} not found in environment")
                return None
        except Exception as e:
            # 函数执行出错
            self.logger.error(f"direct_env executing function {func_name}: {str(e)}")
            return None

    def close(self):
        pass
