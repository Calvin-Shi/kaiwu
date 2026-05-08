#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
from pydoc import locate
import toml
from kaiwudrl.common.utils.singleton import Singleton
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


@Singleton
class _AlgoConf:
    """算法配置类, 实例如下:
    dp:
    actor_agent: agent_dp.agent.Agent
    learner_agent: agent_dp.agent.Agent
    aisrv_agent: agent_dp.agent.Agent
    train_workflow: agent_dp.workflow.train_workflow.workflow
    eval_workflow: agent_dp.workflow.eval_workflow.workflow
    exam_workflow: agent_dp.workflow.exam_workflow.workflow
    """

    class StandardTupleType:
        def __init__(
            self,
            actor_agent,
            learner_agent,
            aisrv_agent,
            train_workflow,
            eval_workflow,
            exam_workflow,
        ):
            self._actor_agent = actor_agent
            self._learner_agent = learner_agent
            self._aisrv_agent = aisrv_agent
            self._train_workflow = train_workflow
            self._eval_workflow = eval_workflow
            self._exam_workflow = exam_workflow

        def __getattr__(self, key):
            value = super().__getattribute__("_" + key)
            value = locate(value)
            return value

    _instance = None

    def __init__(self):
        self.config_map = {}

    def __getitem__(self, key):
        return self.config_map[key]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def load_conf(self, file_name):
        """
        如果是配置文件存在, 则加载配置文件
        如果是配置文件不存在, 则在代码里会自己构造
        """
        if os.path.exists(file_name):
            with open(file_name, "r") as file_obj:
                self._load_conf(file_obj.read())

    def _load_conf(self, algo_str):
        algo_obj = toml.loads(algo_str)

        for algo in algo_obj:
            self.config_map[algo] = self.StandardTupleType(**algo_obj[algo])

    # 获取AlgoConf的配置文件内容
    def get_algo_conf(self, algo, item):
        """
        获取配置项
        1. 如果配置文件存在, 则返回读取的配置
        2. 如果配置文件不存在, 则返回构造的类
        """
        if algo is None or item is None:
            return None

        if algo in self.config_map:
            return getattr(self.config_map[algo], item)
        else:
            if item == "actor_agent" or item == "learner_agent" or item == "aisrv_agent":
                return locate(f"agent_{algo}.agent.Agent")
            elif item == "train_workflow":
                return locate(f"agent_{algo}.workflow.train_workflow.workflow")
            elif item == "eval_workflow":
                return locate(f"tools.eval.workflow.eval_workflow.workflow")
            elif item == "exam_workflow":
                return locate(f"tools.eval.workflow.exam_workflow.workflow")
            else:
                raise NotImplementedError


AlgoConf = _AlgoConf()
