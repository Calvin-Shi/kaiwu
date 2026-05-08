#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class EnvWrapper:
    """
    目前支持的env类型:
    1. kaiwu_env, 开悟自研的env, 典型项目如hok系列, 峡谷系列等, 标准化后不再使用
    2. issac, 基于开源的改造, 典型项目如四足机器人, 无人机等, 凡是用到了issacgym或者issacsim的都使用该类型
    3. kaiwu_env_proxy, 开悟自研的, 用于直接对接项目, 标准化后主推, 用于替换kaiwu_env
    4. direct, 开悟自研的, 主要用于那些想在一个容器里执行环境, 并且不需要评估的情况
    """

    def __init__(self, logger, monitor_proxy) -> None:
        self.logger = logger
        self.monitor_proxy = monitor_proxy
        self.env_object = None

    def init(self, ip, port):
        if CONFIG.aisrv_framework == KaiwuDRLDefine.AISRV_FRAMEWORK_ENV_TYPE_KAIWU_ENV_PROXY:
            from kaiwudrl.components.environment.kaiwu_env_proxy import KaiwuEnvProxy

            self.env_object = KaiwuEnvProxy(self.logger, self.monitor_proxy)
            self.env_object.init(ip, port)

        elif CONFIG.aisrv_framework == KaiwuDRLDefine.AISRV_FRAMEWORK_ENV_TYPE_ISSAC:
            from kaiwudrl.components.environment.isaac_env import IsaacEnv

            self.env_object = IsaacEnv(self.logger, self.monitor_proxy)

        elif CONFIG.aisrv_framework == KaiwuDRLDefine.AISRV_FRAMEWORK_ENV_TYPE_DIRECT:
            from kaiwudrl.components.environment.direct_env import DirectEnv

            self.env_object = DirectEnv(self.logger, self.monitor_proxy)

        elif CONFIG.aisrv_framework == KaiwuDRLDefine.AISRV_FRAMEWORK_ENV_TYPE_KAIWU_ENV:
            import kaiwu_env

            kaiwu_env.setup(run_mode="proxy", skylarena_url=f"tcp://{ip}:{port}")
            self.env_object = kaiwu_env.make(CONFIG.app, logger=self.logger)

        else:
            pass

    def reset(self, usr_conf):
        return self.env_object.reset(usr_conf)

    def step(self, action_data):
        return self.env_object.step(action_data)

    def apply_function(self, func_name, *args, **kwargs):
        return self.env_object.apply_function(func_name, *args, **kwargs)

    def close(self):
        return self.env_object.close()


# 返回错误码和错误字符串, 支持按照字典或者属性读取, 规避各个项目的访问方法不一致问题
def process_error_env_obs(result_code, result_message):
    """
    rest返回错误的env_info信息的组装
    """
    env_obs = {"extra_info": {"result_code": result_code, "result_message": result_message}}
    return env_obs


# 返回错误码和错误字符串, 支持按照字典或者属性读取, 规避各个项目的访问方法不一致问题
class ExtraInfo:
    def __init__(self, result_code, result_message):
        self.result_code = result_code
        self.result_message = result_message

    def __getitem__(self, key):
        # 通过字典键访问时触发
        return self.__dict__[key]  # 直接返回属性值

    def __setitem__(self, key, value):
        # 支持字典式赋值（可选）
        self.__dict__[key] = value

    def get(self, key, default=None):
        # 实现类似字典的 get 方法（可选）
        return self.__dict__.get(key, default)
