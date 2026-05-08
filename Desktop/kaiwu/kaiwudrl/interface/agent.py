#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent 基类模块，根据 CONFIG.wrapper_type 自动选择对应的实现类。

用户只需要继承 BaseAgent，框架会根据配置自动加载：
- WRAPPER_REMOTE: RemoteAgent (集群模式)
- WRAPPER_LOCAL: LocalAgent (单机模式)
- WRAPPER_NONE: BaseAgent (抽象基类)

业务 Agent 需要实现以下方法:
- learn(batch_tensor, *args, **kwargs) -> dict
- predict(list_obs_data, *args, **kwargs) -> list
- exploit(list_obs_data, *args, **kwargs) -> list
- save_model(path, id, *args, **kwargs)
- load_model(path, id, *args, **kwargs)

可选实现:
- reset(list_obs_data, *args, **kwargs)

使用方式:
    from kaiwudrl.interface.agent import BaseAgent
    
    class Agent(BaseAgent):
        def learn(self, batch_tensor):
            return {"loss": 0.5}
        
        def predict(self, list_obs_data):
            return actions
    
    # 对外调用框架层方法
    agent = Agent()
    agent.learn(batch_data)      # 框架自动处理分发
    agent.predict(observations)  # 框架自动处理分发
"""


from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


# 根据 CONFIG.wrapper_type 自动选择对应的 Agent 基类
if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
    # 集群模式: 使用 RemoteAgent
    from kaiwudrl.interface.remote_agent import RemoteAgent as BaseAgent

elif CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
    # 单机模式: 使用 LocalAgent
    from kaiwudrl.interface.local_agent import LocalAgent as BaseAgent

elif CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_NONE:
    # 抽象基类模式: 使用 BaseAgent
    from kaiwudrl.interface.none_agent import BaseAgent

else:
    # 默认使用集群模式
    from kaiwudrl.interface.remote_agent import RemoteAgent as BaseAgent


# 导出符号
__all__ = [
    "BaseAgent",
]
