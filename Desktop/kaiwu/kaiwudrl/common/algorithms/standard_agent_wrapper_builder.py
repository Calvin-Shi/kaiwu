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


class StandardAgentWrapperBuilder:
    """
    对AgentWrapper类的封装, 目前存在多种使用场景, 支持多种强化学习框架
    1. tensorflow, 框架加载业务类, 定义graph, session, 业务使用
    2. tensorflow, 框架加载业务类, 业务定义graph, session, 业务使用
    3. 采用pytorch类, 目前实现的主流
    4. 采用tcnn类
    5. 采用其他方式
    """

    def __init__(self) -> None:
        pass

    def create_agent_wrapper(self, agent, logger, server=None):
        """
        根据配置项加载不同的AgentWrapper
        """

        try:
            # 根据配置项选择不同的AgentWrapper类进行实例化并返回
            if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
                from kaiwudrl.common.algorithms.standard_agent_wrapper_tensorflow_simple import (
                    StandardAgentWrapperTensorflowSimple,
                )

                return StandardAgentWrapperTensorflowSimple(agent, logger, server)

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
                from kaiwudrl.common.algorithms.standard_agent_wrapper_tensorflow_complex import (
                    StandardAgentWrapperTensorflowComplex,
                )

                return StandardAgentWrapperTensorflowComplex(agent, logger, server)

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
                from kaiwudrl.common.algorithms.standard_agent_wrapper_pytorch import (
                    StandardAgentWrapperPytorch,
                )

                return StandardAgentWrapperPytorch(agent, logger, server)

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TCNN == CONFIG.use_which_deep_learning_framework:
                from kaiwudrl.common.algorithms.standard_agent_wrapper_tcnn import (
                    StandardAgentWrapperTcnn,
                )

                return StandardAgentWrapperTcnn(agent, logger, server)

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT == CONFIG.use_which_deep_learning_framework:
                from kaiwudrl.common.algorithms.standard_agent_wrapper_tensorrt import (
                    StandardAgentWrapperTensorRT,
                )

                return StandardAgentWrapperTensorRT(agent, logger, server)

            else:
                # 如果配置项不匹配任何已知的AgentWrapper类，则返回None
                return None
        except Exception as e:
            raise Exception(e)
