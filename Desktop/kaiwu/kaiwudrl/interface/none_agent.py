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


class BaseAgent(ABC):
    """
    Agent 的抽象基类，定义了所有 Agent 必须实现的接口

    业务 Agent 需要继承此类并实现:
    - learn(batch_tensor, *args, **kwargs) -> dict
    - predict(list_obs_data, *args, **kwargs) -> list
    - exploit(list_obs_data, *args, **kwargs) -> list
    - save_model(path, id, *args, **kwargs)
    - load_model(path, id, *args, **kwargs)

    可选实现:
    - init_config(list_obs_data, *args, **kwargs)
    - reset(list_obs_data, *args, **kwargs)

    扩展新的 obs 类方法:
        只需在 _OBS_DISPATCH_METHODS 中添加一行配置即可。
        框架层会自动处理 Local/Remote 模式下的分发逻辑，无需修改任何 agent 代码。
        如果业务层未实现对应的方法，框架层会自动 pass 返回 None。
    """

    # ── obs 类方法的分发配置表 ──
    # 这些方法共享相同的分发模式（predict/exploit/reset/init_config 等）：
    #   - 签名统一为 (self, list_obs_data, *args, **kwargs)
    #   - LocalAgent: 仅在 aisrv 上通过 framework_handler._dispatch_obs_local(method_name, ...) 执行
    #   - RemoteAgent: 在 actor 上直接回调业务层；在 aisrv 上根据 remote_runtime_mode 分发
    #     - LOCAL_AISRV_WORKFLOW → framework_handler._dispatch_obs_local(method_name, ...)
    #     - REMOTE_*_PREDICT    → framework_handler._dispatch_obs_remote(method_name, ...)
    #
    # 配置项说明:
    #   method_name:  业务层需要重写的方法名
    #   is_abstract:  是否为必须实现的抽象方法（False 表示可选，有默认空实现）
    #
    # 未来新增方法只需在此处添加一行配置即可，框架层自动处理分发
    _OBS_DISPATCH_METHODS = [
        {"method_name": "predict", "is_abstract": True},
        {"method_name": "exploit", "is_abstract": True},
        {"method_name": "reset", "is_abstract": False},
        {"method_name": "init_config", "is_abstract": False},
    ]

    @staticmethod
    def _get_methods_to_intercept():
        """
        获取需要拦截的方法列表（obs 类方法 + 特殊方法）

        Returns:
            list[str]: 需要拦截的方法名列表
        """
        # obs 类方法从配置表中获取
        obs_methods = [m["method_name"] for m in BaseAgent._OBS_DISPATCH_METHODS]
        # 特殊方法（有独立分发逻辑，不走通用分发）
        special_methods = ["load_model", "save_model", "learn", "load_opponent_agent"]
        return special_methods + obs_methods

    @staticmethod
    def _install_method_interceptors(cls, agent_class):
        """
        为子类安装方法拦截器的公共逻辑

        参数:
            cls: 业务 Agent 子类
            agent_class: 框架 Agent 类（LocalAgent 或 RemoteAgent）

        说明:
            这个方法会拦截业务层重写的方法，保存到 _business_xxx 属性，
            然后用框架层方法替换，实现框架层和业务层的分离
        """
        # 需要拦截的方法列表（从配置表 + 特殊方法中获取）
        methods_to_intercept = BaseAgent._get_methods_to_intercept()

        # 保存业务层的原始方法并创建包装器
        for method_name in methods_to_intercept:
            if method_name in cls.__dict__:
                # 保存到类属性 _business_xxx
                original_business_method = cls.__dict__[method_name]
                setattr(cls, f"_business_{method_name}", original_business_method)

                # 获取框架层方法
                framework_method = getattr(agent_class, method_name)

                # 创建包装方法 - 使用默认参数固定捕获当前值
                def make_wrapper(fname=framework_method, bname=original_business_method, mname=method_name):
                    def wrapper(self, *args, **kwargs):
                        # 调用框架层方法
                        return fname(self, *args, **kwargs)

                    return wrapper

                # 替换类方法
                setattr(cls, method_name, make_wrapper())

    def __init__(self, agent_type="player", device="cpu", logger=None, monitor=None) -> None:
        self.agent_type = agent_type
        self.device = device
        self.logger = logger
        self.monitor = monitor

        # KaiwuDRL传递的句柄
        self.framework_handler = None

    def set_framework_handler(self, framework_handler):
        self.framework_handler = framework_handler

    def _ensure_framework_handler(self):
        """
        确保 framework_handler 已初始化，否则抛出异常
        """
        if not self.framework_handler:
            raise NotImplementedError(f"framework_handler not initialized on {CONFIG.svr_name}")

    @abstractmethod
    def learn(self, list_sample_data, *args, **kwargs) -> dict:
        """
        学习函数，接收SampleData对象列表

        框架层已自动完成数据转换：
        - reverb返回numpy → dataset转tensor → remote_agent反序列化为list[SampleData]
        - 业务层直接使用SampleData对象，通过torch.cat合并batch

        Args:
            list_sample_data: list[SampleData]，框架层已自动反序列化

        Returns:
            dict: 训练指标字典，如 {'loss': 0.5, 'accuracy': 0.9}

        业务Agent必须实现此方法
        """
        pass

    def send_sample_data(self, list_sample_data, *args, **kwargs):
        """
        发送样本数据到learner

        在集群模式（RemoteAgent）下，此方法在aisrv上调用，将收集的样本数据发送给learner
        在单机模式（LocalAgent）下，此方法可能不需要（因为aisrv和learner在同一进程）

        与learn()的职责分离：
        - send_sample_data: 负责样本数据的传输（在aisrv上调用）
        - learn: 负责模型训练逻辑（在learner上执行）

        框架层自动处理序列化：
        - 读取SampleData.FIELD_DIMS元信息
        - 将SampleData对象序列化为numpy数组
        - 发送到learner通过reverb
        - 无需业务层提供转换函数

        Args:
            list_sample_data: list[SampleData]，框架层会自动序列化
            train_data_prioritized: (可选) 样本优先级列表，用于优先经验回放
        """
        pass

    @abstractmethod
    def predict(self, list_obs_data: list, *args, **kwargs) -> list:
        """
        预测函数，接受一个 ObsData 的列表, 返回一个动作列表
        业务Agent必须实现此方法
        """
        pass

    @abstractmethod
    def exploit(self, list_obs_data: list, *args, **kwargs) -> list:
        """
        exploit函数，接受一个 ObsData 的列表, 返回一个动作列表
        业务Agent必须实现此方法
        """
        pass

    @abstractmethod
    def save_model(self, path, id="1", *args, **kwargs):
        """
        保存模型
        业务Agent必须实现此方法
        """
        pass

    @abstractmethod
    def load_model(self, path, id="1", *args, **kwargs):
        """
        加载模型
        业务Agent必须实现此方法
        """
        pass

    def get_training_metrics(self, *args, **kwargs):
        """
        获取训练指标，可选实现
        """
        return None

    def observation_process(self, env_obs, *args, **kwargs):
        """
        针对观测做处理
        """
        return None

    def action_process(self, act_data, *args, **kwargs):
        """
        针对动作做处理
        """
        return None
