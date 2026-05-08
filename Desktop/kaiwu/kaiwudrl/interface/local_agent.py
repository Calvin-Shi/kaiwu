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
from kaiwudrl.interface.none_agent import BaseAgent


class LocalAgent(BaseAgent):
    """
    本地单机单进程模式的Agent

    在单机单进程模式下，aisrv都在同一个进程中，通过本地调用
    业务Agent应该继承此类，并实现 learn(), predict(), exploit() 等方法

    框架会自动处理 aisrv 之间的分发逻辑
    """

    def __init__(self, agent_type="player", device="cpu", logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)

    def __init_subclass__(cls, **kwargs):
        """
        当子类继承LocalAgent时自动调用，用于安装方法拦截器
        拦截的方法：load_model, save_model, predict, exploit, learn, reset, init_config 等
        """
        super().__init_subclass__(**kwargs)
        # 调用基类的公共方法，传入 LocalAgent 作为框架类
        BaseAgent._install_method_interceptors(cls, LocalAgent)

    @staticmethod
    def _check_server_type():
        """
        防御性检查：LocalAgent 只应该在 aisrv 上执行
        Returns:
            bool: True 表示在正确的服务器上，False 表示在错误的服务器上
        """
        return CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV

    def send_sample_data(self, list_sample_data, *args, **kwargs):
        """
        单机单进程模式下发送样本数据（只在aisrv上执行）

        在单机单进程模式下，aisrv/learner在同一进程中，通常不需要显式发送样本
        如果需要，可以直接调用learn()方法进行训练

        注意：send_sample_data 是分布式模式专用接口，单机模式不应调用
        """
        if not LocalAgent._check_server_type():
            return None

        # 单机模式下不应该调用 send_sample_data，这是分布式接口
        raise RuntimeError(
            f"运行模式不匹配: 当前为单机模式(LocalAgent)，但调用了分布式专用接口 send_sample_data()。\n"
            f"单机模式下 aisrv/learner 在同一进程中，应该直接调用 agent.learn() 进行训练。\n"
            f"如需使用分布式模式，请将 CONFIG.wrapper_type 设置为 '{KaiwuDRLDefine.WRAPPER_REMOTE}'，"
            f"并继承 RemoteAgent 类。"
        )

    def learn(self, list_sample_data, *args, **kwargs) -> dict:
        """
        单机模式下的学习逻辑（只在aisrv上执行）

        框架层：处理 aisrv 分发到 framework_handler
        业务层：子类重写此方法实现训练逻辑

        Args:
            list_sample_data: list[SampleData]，业务层通过torch.cat合并batch

        Returns:
            dict: 训练指标字典
        """
        if not LocalAgent._check_server_type():
            return None

        # 如果是业务层调用（带framework标记），直接调用业务逻辑
        if kwargs.get("framework"):
            del kwargs["framework"]
            return self.__class__._business_learn(self, list_sample_data, *args, **kwargs)

        # 框架层：分发到framework_handler
        """
        单机单进程的情况下, 有2种代码写法:
        1. list_sample_data是具体的样本数据, agent.learn包括了训练1个操作
        2. list_sample_data传空, agent.learn里包括了读取样本数据和训练2个操作
        """
        self._ensure_framework_handler()
        return self.framework_handler.train_local(self, list_sample_data, *args, **kwargs)

    # ── obs 类方法的通用分发逻辑 ──

    def _dispatch_obs_method(self, method_name, list_obs_data, *args, **kwargs):
        """
        obs 类方法（predict/exploit/reset/init_config 等）的通用分发逻辑

        单机模式下统一处理模式：
        1. 检查是否在 aisrv 上
        2. 如果是框架层回调业务层（framework 标记），直接调用 _business_xxx
        3. 否则分发到 framework_handler._dispatch_obs_local(method_name, ...)

        Args:
            method_name: 方法名（如 "predict"），用于查找 _business_xxx 和分发到 handler
            list_obs_data: 观测数据列表
        """
        if not LocalAgent._check_server_type():
            return None

        # 如果是业务层调用（带framework标记），直接调用业务逻辑
        if kwargs.get("framework"):
            del kwargs["framework"]
            business_method = getattr(self.__class__, f"_business_{method_name}", None)
            if business_method is None:
                return None
            return business_method(self, list_obs_data, *args, **kwargs)

        # 框架层：分发到 framework_handler._dispatch_obs_local()
        self._ensure_framework_handler()
        return self.framework_handler._dispatch_obs_local(method_name, self, list_obs_data)

    def save_model(self, path=None, id="1", *args, **kwargs):
        """
        单机单进程模式下的保存模型逻辑（只在aisrv上执行）
        框架层：处理 aisrv 分发到 framework_handler
        业务层：子类重写此方法实现保存模型逻辑
        """
        if not LocalAgent._check_server_type():
            return None

        # 如果是业务层调用（带framework标记），直接调用业务逻辑
        if kwargs.get("framework"):
            del kwargs["framework"]
            return self.__class__._business_save_model(self, path=path, id=id, *args, **kwargs)

        # 框架层：分发到framework_handler
        self._ensure_framework_handler()
        return self.framework_handler.save_model_local(self, path=path, id=id, *args, **kwargs)

    def load_model(self, path=None, id="1", *args, **kwargs):
        """
        单机单进程模式下的加载模型逻辑（只在aisrv上执行）
        框架层：处理 aisrv 分发到 framework_handler
        业务层：子类重写此方法实现加载模型逻辑
        """
        if not LocalAgent._check_server_type():
            return None

        # 如果是业务层调用（带framework标记），直接调用业务逻辑
        if kwargs.get("framework"):
            del kwargs["framework"]
            return self.__class__._business_load_model(self, path=path, id=id, *args, **kwargs)

        # 框架层：分发到framework_handler
        self._ensure_framework_handler()
        return self.framework_handler.load_model_local(self, path=path, id=id, *args, **kwargs)

    def get_training_metrics(self, *args, **kwargs):
        """
        单机单进程模式下获取训练指标（只在aisrv上执行）
        """
        if not LocalAgent._check_server_type():
            return None

        # 框架层：分发到framework_handler
        self._ensure_framework_handler()
        return self.framework_handler.get_training_metrics_local(self, *args, **kwargs)


# ── 根据 _OBS_DISPATCH_METHODS 配置表自动生成 obs 类方法 ──
def _make_local_obs_method(method_name):
    """
    工厂函数：为 LocalAgent 生成 obs 类方法

    生成的方法等价于:
        def predict(self, list_obs_data, *args, **kwargs):
            return self._dispatch_obs_method("predict", list_obs_data, *args, **kwargs)
    """

    def obs_method(self, list_obs_data: list, *args, **kwargs):
        return self._dispatch_obs_method(method_name, list_obs_data, *args, **kwargs)

    obs_method.__name__ = method_name
    obs_method.__qualname__ = f"LocalAgent.{method_name}"
    obs_method.__doc__ = f"单机模式下的{method_name}逻辑（只在aisrv上执行），由 _OBS_DISPATCH_METHODS 配置表自动生成"
    return obs_method


for _method_conf in BaseAgent._OBS_DISPATCH_METHODS:
    setattr(
        LocalAgent,
        _method_conf["method_name"],
        _make_local_obs_method(_method_conf["method_name"]),
    )
