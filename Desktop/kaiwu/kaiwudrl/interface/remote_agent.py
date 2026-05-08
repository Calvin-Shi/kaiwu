#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import importlib
import numpy as np
import torch
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.interface.none_agent import BaseAgent


class RemoteAgent(BaseAgent):
    """
    远程集群模式的Agent

    在集群模式下，aisrv/learner/actor分布在不同的进程/机器上，需要通过消息队列通信
    业务Agent应该继承此类
    """

    def __init__(self, agent_type="player", device="cpu", logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        # 设置agent的运行模式, 同一个agent可以不断的变化运行模式, 参考hok系列的对手模型
        self.remote_runtime_mode = None

        # 获取业务层的SampleData类和字段元信息（懒加载，第一次使用时才加载）
        self._sample_data_class = None
        self._sample_data_field_info = None
        self._sample_data_info_loaded = False  # 标记是否已尝试加载

    def __init_subclass__(cls, **kwargs):
        """
        当子类继承RemoteAgent时自动调用，用于安装方法拦截器
        这个钩子在类定义时执行，早于__init__，确保方法拦截生效

        拦截的方法：load_model, save_model, predict, exploit, learn, reset
        """
        super().__init_subclass__(**kwargs)
        # 调用基类的公共方法，传入 RemoteAgent 作为框架类
        BaseAgent._install_method_interceptors(cls, RemoteAgent)

    def learn(self, list_sample_data, *args, **kwargs) -> dict:
        """
        集群模式下的学习逻辑

        框架层：处理 aisrv/learner 分发
        业务层：子类重写此方法实现训练逻辑

        注意：在分布式模式下，learn() 只应该在 learner 上执行

        数据流（优化后）：
        1. reverb返回numpy
        2. reverb_dataset转为tensor, 避免冗余转换
        3. 框架层反序列化为list[SampleData]（根据FIELD_DIMS自动切割）
        4. 业务层使用torch.cat合并batch进行训练

        Args:
            list_sample_data: batch tensor，shape为 (batch_size, total_dim)
                             框架层会自动反序列化为list[SampleData]传给业务层
                             tensor已在正确设备上（GPU/CPU）

        Returns:
            dict: 训练指标字典，如 {'loss': 0.5, 'accuracy': 0.9}
        """
        if list_sample_data is None:
            return None

        # 如果是业务层调用（带framework标记），直接调用业务逻辑
        if kwargs.get("framework") or CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            if kwargs.get("framework"):
                del kwargs["framework"]

            # 懒加载：确保SampleData信息已加载
            self._ensure_sample_data_info()

            # 在learner端，框架层直接将batch_tensor反序列化为SampleData对象
            if self._sample_data_class is not None and self._sample_data_field_info is not None:
                # 框架层实现：将batch tensor按照字段信息切割成SampleData对象
                sample_data_list = self._batch_tensor_to_sample_data(list_sample_data)
                return self.__class__._business_learn(self, sample_data_list, *args, **kwargs)
            else:
                # 兼容模式：直接传递原始tensor
                return self.__class__._business_learn(self, list_sample_data, *args, **kwargs)
        # 在 aisrv 上调用 learn() 是错误的，应该使用 send_sample_data()
        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            raise RuntimeError(
                f"运行模式不匹配: 当前为分布式模式(RemoteAgent)，但在 aisrv 上调用了 learn()。\n"
                f"分布式模式下，learn() 只应在 learner 上执行，aisrv 应该调用 send_sample_data() 发送样本数据。\n"
                f"如需在单机模式下直接调用 learn()，请将 CONFIG.wrapper_type 设置为 '{KaiwuDRLDefine.WRAPPER_LOCAL}'，"
                f"并继承 LocalAgent 类。"
            )
        else:
            return None

    def send_sample_data(self, list_sample_data, *args, **kwargs):
        """
        集群模式下发送样本数据到learner

        只在aisrv上调用，用于将收集的样本数据发送给learner进行训练

        与learn()的区别：
        - send_sample_data: 在aisrv上发送样本数据（数据传输）
        - learn: 在learner上执行训练逻辑（模型训练）

        框架层自动处理序列化（无需业务层提供转换函数）：
        1. 读取SampleData.FIELD_DIMS元信息（由create_cls自动生成）
        2. 将SampleData对象序列化为numpy数组
        3. 发送到reverb server

        优势：
        - 业务层只需定义SampleData字段和维度（如 obs=153）
        - 无需手动编写转换函数
        - 框架层统一处理，维护成本低

        Args:
            list_sample_data: list[SampleData]，框架层会自动序列化
            train_data_prioritized: (可选) 样本优先级列表，用于优先经验回放
        """
        if list_sample_data is None:
            return None

        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            self._ensure_framework_handler()

            # 懒加载：确保SampleData信息已加载
            self._ensure_sample_data_info()

            # 框架层直接实现序列化：将SampleData对象转换为numpy数组
            if self._sample_data_field_info is not None:
                numpy_array = [self._sample_data_to_numpy(sample) for sample in list_sample_data]
            else:
                # 兼容模式：直接传递
                numpy_array = list_sample_data

            # 从参数中获取优先级列表，若未提供则生成默认全1列表
            train_data_prioritized_list = kwargs.pop("train_data_prioritized", None)
            if train_data_prioritized_list is None:
                train_data_prioritized_list = [1] * len(numpy_array)
            else:
                if len(train_data_prioritized_list) != len(numpy_array):
                    raise ValueError("train_data_prioritized 必须与 numpy_array 长度相同")

            # 从aisrv传递给learner
            return self.framework_handler.send_train_data(self, numpy_array, train_data_prioritized_list)
        else:
            return None

    def _ensure_remote_runtime_mode(self):
        """
        懒加载：确保remote_runtime_mode已初始化

        优势：
        - 即使子类忘记调用super().__init__()，也能正常工作
        - 自动容错，避免AttributeError

        调用时机：
        - predict() / exploit() 访问 remote_runtime_mode 前
        """
        if not hasattr(self, "remote_runtime_mode"):
            # 初始化缺失的属性（容错处理）
            self.remote_runtime_mode = None

    def _ensure_sample_data_info(self):
        """
        懒加载：确保SampleData信息已加载

        优势：
        - 按需加载，只在真正使用时才加载
        - 即使子类忘记调用super().__init__()，也能正常工作
        - 避免在__init__时加载失败导致初始化中断

        调用时机：
        - send_sample_data() 发送样本前
        - learn() 接收样本前
        """
        # 防御性检查：如果属性不存在，说明子类没有调用 super().__init__()
        if not hasattr(self, "_sample_data_info_loaded"):
            # 初始化缺失的属性（容错处理）
            self._sample_data_class = None
            self._sample_data_field_info = None
            self._sample_data_info_loaded = False

        # 懒加载：只在第一次使用时加载
        if not self._sample_data_info_loaded:
            self._load_sample_data_info()
            self._sample_data_info_loaded = True

    def _load_sample_data_info(self):
        """
        加载业务层的SampleData类和字段维度信息

        固定路径规则：agent_{algo}/feature/definition.py

        新方式（推荐）：
        ```python
        SampleData = create_cls(
            "SampleData",
            obs=153,           # 字段名=维度
            legal_actions=8,
            actions=1,
            ...
        )
        # 自动生成 SampleData.FIELD_DIMS = {"obs": 153, "legal_actions": 8, ...}
        ```

        旧方式（兼容）：
        业务层手动定义 SAMPLE_DATA_DIMS 字典

        优势：
        - 只需定义一次，字段名和维度统一管理
        - 框架层自动处理序列化/反序列化
        - 减少业务层代码 ~80-90 行
        """
        module_name = f"agent_{CONFIG.algo}.feature.definition"

        try:
            module = importlib.import_module(module_name)

            # 获取SampleData类
            if hasattr(module, "SampleData"):
                self._sample_data_class = getattr(module, "SampleData")

                # 优先从SampleData.FIELD_DIMS获取（新方式）
                if hasattr(self._sample_data_class, "FIELD_DIMS"):
                    self._sample_data_field_info = self._sample_data_class.FIELD_DIMS
                # 兼容旧方式：从模块的SAMPLE_DATA_DIMS获取
                elif hasattr(module, "SAMPLE_DATA_DIMS"):
                    self._sample_data_field_info = getattr(module, "SAMPLE_DATA_DIMS")
                else:
                    # 如果都没有定义，尝试使用旧的转换函数（完全兼容）
                    self._sample_data_field_info = None

        except (ImportError, Exception) as e:
            # 如果加载失败，保持为None（兼容模式）
            pass

    def _sample_data_to_numpy(self, sample_data):
        """
        框架层实现：将SampleData对象序列化为numpy数组

        调用时机：aisrv端调用send_sample_data时

        处理流程：
        1. 按照FIELD_DIMS定义的字段顺序遍历
        2. 提取每个字段的值（支持numpy/tensor/list）
        3. 统一转换为numpy并展平
        4. 使用np.hstack拼接为1D数组

        Args:
            sample_data: SampleData对象

        Returns:
            numpy.ndarray: 拼接后的1D数组（按照FIELD_DIMS中定义的顺序）
        """
        arrays = []
        for field_name in self._sample_data_field_info.keys():
            value = getattr(sample_data, field_name, None)
            if value is not None:
                # 转换为numpy数组
                if isinstance(value, np.ndarray):
                    arr = value
                elif isinstance(value, torch.Tensor):
                    arr = value.cpu().numpy()
                elif isinstance(value, (list, tuple)):
                    arr = np.array(value, dtype=np.float32)
                else:
                    arr = np.array([value], dtype=np.float32)

                # 展平并添加到列表
                arrays.append(arr.flatten())

        # 使用np.hstack拼接所有字段
        return np.hstack(arrays) if arrays else np.array([], dtype=np.float32)

    def _batch_tensor_to_sample_data(self, batch_data):
        """
        框架层实现：将batch数据（tensor/numpy）反序列化为SampleData对象列表

        调用时机：learner端接收到batch数据后，传给业务层learn前

        处理流程：
        1. 统一转换为tensor（支持numpy、tensor输入）
        2. 按照FIELD_DIMS定义的顺序切割tensor（零拷贝切片）
        3. 为batch中的每个样本创建SampleData对象
        4. 返回list[SampleData]供业务层使用

        数据来源：
        - Off-policy: reverb_dataset_v1直接返回（tensor或numpy）
        - On-policy: standard_agent_wrapper_pytorch过滤后统一返回tensor

        优化说明：
        - 推荐配置sample_data_return_data_type='tensor'，零拷贝
        - 零拷贝切片：tensor[i:i+1]只创建view，无内存拷贝
        - on-policy场景已优化为统一tensor输出，避免list转换

        Args:
            batch_data: torch.Tensor或numpy.ndarray, shape为 (batch_size, total_dim)
                       standard_agent_wrapper_pytorch已确保统一输出tensor

        Returns:
            list[SampleData]: SampleData对象列表，每个元素对应batch中的一个样本
                             字段值为tensor，业务层通过torch.stack()合并
        """
        # 统一转换为tensor
        if isinstance(batch_data, np.ndarray):
            batch_tensor = torch.from_numpy(batch_data).float()
        elif isinstance(batch_data, torch.Tensor):
            batch_tensor = batch_data
        else:
            raise TypeError(
                f"不支持的数据类型: {type(batch_data)}。\n"
                f"期望类型: torch.Tensor 或 numpy.ndarray。\n"
                f"如果看到此错误，说明 standard_agent_wrapper_pytorch 的输出格式不符合预期。"
            )

        # 确保是2D tensor
        if batch_tensor.dim() == 1:
            batch_tensor = batch_tensor.unsqueeze(0)

        batch_size = batch_tensor.shape[0]

        # 按照FIELD_DIMS定义的顺序切割tensor（切割为batch维度的字段）
        field_tensors_batch = {}
        idx = 0
        for field_name, field_dim in self._sample_data_field_info.items():
            if field_dim == 1:
                # 标量字段：shape为 (batch_size)
                field_tensors_batch[field_name] = batch_tensor[:, idx]
            else:
                # 向量字段：shape为 (batch_size, field_dim)
                field_tensors_batch[field_name] = batch_tensor[:, idx : idx + field_dim]
            idx += field_dim

        # 将batch tensor拆分为单个样本的SampleData对象列表
        sample_data_list = []
        for i in range(batch_size):
            # 提取第i个样本的所有字段
            sample_fields = {}
            for field_name, field_tensor in field_tensors_batch.items():
                # 每个字段取第i行（保持tensor格式，零拷贝）
                sample_fields[field_name] = field_tensor[i]

            # 创建SampleData对象
            sample_data_list.append(self._sample_data_class(**sample_fields))

        return sample_data_list

    def _dispatch_obs_method(self, method_name, list_obs_data, *args, **kwargs):
        """
        obs 类方法（predict/exploit/reset/init_config 等）的通用分发逻辑

        集群模式下统一处理模式：
        1. actor 上或带 framework 标记 → 直接回调 _business_xxx
        2. aisrv 上 → 根据 remote_runtime_mode 分发：
           - LOCAL_AISRV_WORKFLOW → framework_handler._dispatch_obs_local(method_name, ...)
           - REMOTE_*_PREDICT    → framework_handler._dispatch_obs_remote(method_name, ...)

        Args:
            method_name: 方法名（如 "predict"），用于查找 _business_xxx 和分发到 handler
            list_obs_data: 观测数据列表
        """
        # 如果是业务层调用（带framework标记或在actor上），直接调用业务逻辑
        if kwargs.get("framework") or CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            if kwargs.get("framework"):
                del kwargs["framework"]
            business_method = getattr(self.__class__, f"_business_{method_name}", None)
            if business_method is None:
                return None
            return business_method(self, list_obs_data, *args, **kwargs)

        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            self._ensure_framework_handler()
            # 懒加载：确保 remote_runtime_mode 已初始化
            self._ensure_remote_runtime_mode()

            # 如果使用运行在workflow上的agent则直接调用 framework_handler._dispatch_obs_local()
            if self.remote_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW:
                return self.framework_handler._dispatch_obs_local(method_name, self, list_obs_data)
            # 如果使用运行在predictor上的agent则调用 framework_handler._dispatch_obs_remote()
            elif self.remote_runtime_mode in [
                KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
                KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_REMOTE_ACTOR_PREDICT,
            ]:
                return self.framework_handler._dispatch_obs_remote(method_name, self, list_obs_data)

        return None

    def save_model(self, path=None, id="1", *args, **kwargs):
        """
        集群模式下的保存模型逻辑
        框架层：处理 learner 分发到 framework_handler
        - AISRV: framework_handler 是 KaiWuRLHelper
        - LEARNER: framework_handler 是 agent_wrapper
        业务层：子类重写此方法实现保存模型逻辑
        """
        try:
            # 如果是业务层调用（带framework标记），直接调用业务逻辑
            if kwargs.get("framework"):
                del kwargs["framework"]
                return self.__class__._business_save_model(self, path=path, id=id, *args, **kwargs)

            # 框架层调用
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                self._ensure_framework_handler()
                # framework_handler 是 agent_wrapper，直接调用 save_param_by_source
                return self.framework_handler.save_param_by_source(
                    path=path,
                    id=id,
                    source=kwargs.pop("source", KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER),
                    **kwargs,
                )

            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                self._ensure_framework_handler()
                # framework_handler 是 KaiWuRLHelper，调用 send_save_model_file_data
                return self.framework_handler.send_save_model_file_data(self, path, id, *args, **kwargs)
            else:
                return None
        except Exception as e:
            raise RuntimeError(f"save model error: {e}")

    def load_model(self, path=None, id="1", *args, **kwargs):
        """
        集群模式下的加载模型逻辑
        框架层：处理 actor/learner/aisrv 分发到 framework_handler
        - AISRV: framework_handler 是 KaiWuRLHelper
        - ACTOR/LEARNER: framework_handler 是 agent_wrapper
        业务层：子类重写此方法实现加载模型逻辑
        """
        try:
            # 如果是业务层调用（带framework标记），直接调用业务逻辑
            if kwargs.get("framework"):
                del kwargs["framework"]
                return self.__class__._business_load_model(self, path=path, id=id, *args, **kwargs)

            # 框架层调用
            # ACTOR 或 LEARNER：framework_handler 是 agent_wrapper
            if CONFIG.svr_name in [KaiwuDRLDefine.SERVER_ACTOR, KaiwuDRLDefine.SERVER_LEARNER]:
                self._ensure_framework_handler()
                # framework_handler 是 agent_wrapper，直接调用 load_model_by_source
                return self.framework_handler.load_model_by_source(
                    path=path,
                    id=id,
                    source=kwargs.pop("source", KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER),
                    **kwargs,
                )

            # AISRV：framework_handler 是 KaiWuRLHelper，需要根据运行模式处理
            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                self._ensure_framework_handler()
                # 懒加载：确保 remote_runtime_mode 已初始化
                self._ensure_remote_runtime_mode()

                # 调用load_model会刷新remote_runtime_mode
                self.remote_runtime_mode = CONFIG.remote_agent_only_default_runtime_mode

                # 如果使用运行在workflow上的agent则调用 KaiWuRLHelper.load_model_local
                if self.remote_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW:
                    # framework_handler 是 KaiWuRLHelper，调用 load_model_local
                    # 内部会获取 agent_wrapper 并调用 load_model_by_source
                    return self.framework_handler.load_model_local(self, path, id, **kwargs)

                # 如果使用运行在predictor上的agent则调用framework_handler的send_load_model_file_data函数
                elif self.remote_runtime_mode in [
                    KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_REMOTE_ACTOR_PREDICT,
                    KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
                ]:
                    return self.framework_handler.send_load_model_file_data(self, path, id, *args, **kwargs)

            return None
        except Exception as e:
            raise RuntimeError(f"load model error: {e}")

    def load_opponent_agent(self, path=None, id="1", *args, **kwargs):
        """
        集群模式下加载对手agent
        只在aisrv上调用，调用后会刷新runtime_mode为REMOTE_AISRV_PREDICT
        """
        try:
            # 如果是业务层调用（带framework标记），直接调用业务逻辑
            if kwargs.get("framework"):
                del kwargs["framework"]
                return self.__class__._business_load_model(self, path=path, id=id, *args, **kwargs)

            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                self._ensure_framework_handler()
                # 懒加载：确保 remote_runtime_mode 已初始化
                self._ensure_remote_runtime_mode()

                # 对手模型必须要确保remote_agent_default_runtime_mode是REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT或者
                # REMOTE_AGENT_RUNTIME_MODE_REMOTE_ACTOR_PREDICT
                if CONFIG.remote_agent_default_runtime_mode not in [
                    KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_REMOTE_ACTOR_PREDICT,
                    KaiwuDRLDefine.REMOTE_AGENT_ONLY_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
                ]:

                    raise RuntimeError(
                        f"采用了对手模型, 则remote_agent_default_runtime_mode是REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT或者REMOTE_AGENT_RUNTIME_MODE_REMOTE_ACTOR_PREDICT"
                    )
                self.remote_runtime_mode = CONFIG.remote_agent_default_runtime_mode
                return self.framework_handler.send_load_model_file_data(self, path, id, *args, **kwargs)

            return None
        except Exception as e:
            raise RuntimeError(f"load opponent agent error: {e}")

    def get_training_metrics(self, *args, **kwargs):
        """
        集群模式下获取训练指标
        只在aisrv上调用，通过zmq发送到learner获取
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            self._ensure_framework_handler()
            return self.framework_handler.get_training_metrics(self, *args, **kwargs)
        else:
            return None


# ── 根据 _OBS_DISPATCH_METHODS 配置表自动生成 obs 类方法 ──
def _make_remote_obs_method(method_name):
    """
    工厂函数：为 RemoteAgent 生成 obs 类方法

    生成的方法等价于:
        def predict(self, list_obs_data, *args, **kwargs):
            return self._dispatch_obs_method("predict", list_obs_data, *args, **kwargs)
    """

    def obs_method(self, list_obs_data: list, *args, **kwargs):
        return self._dispatch_obs_method(method_name, list_obs_data, *args, **kwargs)

    obs_method.__name__ = method_name
    obs_method.__qualname__ = f"RemoteAgent.{method_name}"
    obs_method.__doc__ = f"集群模式下的{method_name}逻辑，由 _OBS_DISPATCH_METHODS 配置表自动生成"
    return obs_method


for _method_conf in BaseAgent._OBS_DISPATCH_METHODS:
    setattr(
        RemoteAgent,
        _method_conf["method_name"],
        _make_remote_obs_method(_method_conf["method_name"]),
    )
