#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
因为具身智能这些项目必须要求torch在isaacgym导入之后, 故这里先导入
"""
try:
    import isaacgym
except ImportError:
    pass

import os
import threading
import multiprocessing
import time
import datetime
import numpy as np
import torch
import dill
import copy
from typing import Any, Tuple
from msgpack import ExtType
import msgpack
from kaiwudrl.common.utils.common_func import (
    Context,
    get_host_ip,
    set_schedule_event,
)
from kaiwudrl.interface.agent_context import AgentContext
from common_python.config.config_control import CONFIG
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.config.algo_conf import AlgoConf
from common_python.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.checkpoint.model_file_common import (
    update_id_list,
    clear_id_list_file,
    clear_user_ckpt_dir,
    check_id_valid,
)
from kaiwudrl.components.environment.env_wrapper import EnvWrapper
import warnings
from kaiwudrl.server.common.load_model_common import LoadModelCommon
from kaiwudrl.common.utils.common_func import get_machine_device_by_config
from kaiwudrl.common.algorithms.agent_wrapper_common import (
    create_standard_agent_wrapper,
)
import random
import schedule
from kaiwudrl.common.checkpoint.model_file_common import (
    get_checkpoint_id_by_re,
)
from contextlib import contextmanager


# 实现标准的强化学习训练流程
class KaiWuRLStandardHelper(threading.Thread):
    __slots__ = (
        "policies",
        "simu_ctx",
        "exit_flag",
        "client_address",
        "slot_id",
        "data_queue",
        "client_id",
        "episode_start_time",
        "ep_frame_cnt",
        "agent_ctxs",
        "logger",
        "steps",
        "use_sample_server",
        # __init__ 中赋值的属性
        "current_pid",
        "ip",
        "from_actor_model_version",
        "from_learner_model_version",
        "agent_ids",
        "app_monitor_data",
        "monitor_proxy",
        "policy_agent_wrapper_maps",
        "current_agents",
        "pool",
        "last_report_monitor_time",
        "model_file_sync_wrapper",
        "last_error_time",
        "error_interval",
        # before_run 中赋值的属性
        "env_wrapper",
        "policy_conf",
        "load_model_common_object",
    )

    def __init__(self, parent_simu_ctx) -> None:
        super().__init__()

        self.policies = {}
        # 根据policy来设置下, 强化学习是AsyncPolicy, 形如train --> AsyncPolicy
        for policy_name, policy_builder in parent_simu_ctx.policies_builder.items():
            self.policies[policy_name] = policy_builder.build()

        # 上下文放在该变量里
        self.simu_ctx = Context(**parent_simu_ctx.__dict__)

        # 是否结束标志位
        self.exit_flag = self.simu_ctx.exit_flag
        # 客户端ID
        self.client_address = self.simu_ctx.client_address
        # policy
        self.simu_ctx.policies = self.policies
        # slot_id
        self.slot_id = self.simu_ctx.slot_id

        # 数据队列
        self.data_queue = self.simu_ctx.data_queue

        # 设置线程名字
        self.setName(f"kaiwu_rl_helper_{self.slot_id}")

        self.client_id = None

        # 下面是episode的统计指标
        self.episode_start_time = 0
        self.ep_frame_cnt = 0

        # 智能体agent的上下文agent_ctxs, 格式为{"agent_id" : agent_ctx}
        self.agent_ctxs = {}

        """
        由于调用的是workflow是业务侧书写的, 可能存在打印日志比较多的场景, 故满足下面条件的即可打印日志:
        1. workflow进程里第0号进程
        2. 评估和评测

        注意这里没有采用host是担心在单机部署时少量的进程情况下没有日志产生
        """
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()

        self.ip = get_host_ip()
        if (
            self.simu_ctx.index not in CONFIG.process_start_index_allowed_print_log_list
            and CONFIG.run_mode != KaiwuDRLDefine.RUN_MODE_EVAL
            and CONFIG.run_mode != KaiwuDRLDefine.RUN_MODE_EXAM
        ):
            self.logger.add_not_allowed_pid(self.current_pid)

        params = {
            "compression": CONFIG.compression,
            "encoding": CONFIG.encoding,
            "rotation": CONFIG.rotation,
            "level": CONFIG.level,
            "serialize": CONFIG.serialize,
            "retention": CONFIG.retention,
            "max_single_message_len": CONFIG.max_single_message_len,
            "max_calls_log_per_min": CONFIG.max_calls_log_per_min,
        }
        self.logger.set_logger_format(
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/aisrv_kaiwu_rl_helper_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            None,
            params,
        )
        self.logger.info(
            f"kaiwu_rl_helper start at pid {self.current_pid}, "
            f"ppid is {threading.currentThread().ident}, thread id is {self.get_pid()}"
        )

        # 将日志句柄作为参数传递
        self.simu_ctx.logger = self.logger

        # 动作执行步数，用于样本计数
        self.steps = 0

        # 是否用sample_server进行样本存储
        self.use_sample_server = CONFIG.use_sample_server

        # aisrv发给actor的请求返回给处理该值的model文件版本号
        self.from_actor_model_version = -1

        # learner通知aisrv此时最新的model文件版本号
        self.from_learner_model_version = -1

        # 有多少agent_id
        self.agent_ids = []

        # 业务会在aisrv里上报自定义的监控指标, 故这里增加上, map形式, 由业务自己定义
        self.app_monitor_data = {}

        # 由于某些场景下kaiwu_rl_helper会退出, 此时不确定aisrv_server_standard能否退出, 故将监控对象传递下做最后的上报
        self.monitor_proxy = None

        # policy和model对象的map, 为了支持多agent
        self.policy_agent_wrapper_maps = {}
        self.current_agents = []

        # 采用进程池技术, 主要是处理多个agent同时进行预测/利用的场景, 比如边境突围100个agent同时预测
        if CONFIG.multi_agent_predict == KaiwuDRLDefine.MULTI_AGENT_PREDICT_PARALLEL:
            self.pool = self.simu_ctx.pool

        # 统计值
        if (CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL) or (
            CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE
            and CONFIG.remote_agent_default_runtime_mode
            in [
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
            ]
        ):
            # 进程上报监控时间
            self.last_report_monitor_time = 0

        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            # 如果是在actor远程执行的REMOTE_AGENT_RUNTIME_MODE_REMOTE_ACTOR_PREDICT则不需要启动model_file_sync的
            if CONFIG.remote_agent_default_runtime_mode in [
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
            ]:
                self.model_file_sync_wrapper = self.simu_ctx.model_file_sync_wrapper

        self.last_error_time = 0  # 上一次打印错误的时间戳
        self.error_interval = 300  # 错误日志打印间隔（单位：秒）

    def set_monitor_proxy(self, monitor_proxy):
        self.monitor_proxy = monitor_proxy

    # 获取当前使用的actor和learner列表
    def get_current_actor_learner_address(self):
        actor_addrs, learner_addrs = None, None
        policy_build = self.policies[CONFIG.policy_name]
        if policy_build:
            (
                actor_addrs,
                learner_addrs,
            ) = policy_build.get_current_actor_learner_proxy_list()

        return actor_addrs, learner_addrs

    def kaiwu_rl_helper_change_actor_learner_ip(
        self, actor_add_or_reduce, actor_ips, learner_add_or_reduce, learner_ips
    ):
        """
        修改kaiwu_rl_helper的actor和learner地址
        1. actor_add_or_reduce, 针对actor的增减
        2. actor_ips, actor_ip列表
        3. learner_add_or_reduce, 针对learner的增减
        4. learner_ips, learner_ip列表

        返回的参数:
        1. False, 即本次没有更新, 不能修改old_actor_address和old_learner_address
        2. True, 即本次更新完成, 需要修改old_learner_address和old_learner_address
        """

        if actor_add_or_reduce and not actor_ips:
            return False

        if learner_add_or_reduce and not learner_ips:
            return False

        # 针对当前的policy_name进行处理
        policy = self.policies[CONFIG.policy_name]

        # 下面针对具体的actor和learner的增减进行处理
        if actor_add_or_reduce and actor_ips:
            for actor_ip in actor_ips:
                if KaiwuDRLDefine.PROCESS_ADD == actor_add_or_reduce:
                    policy.add_actor_proxy_list(actor_ip)
                elif KaiwuDRLDefine.PROCESS_REDUCE == actor_add_or_reduce:
                    policy.reduce_actor_proxy_list(actor_ip)
                else:
                    pass

        if learner_add_or_reduce and learner_ips:
            for learner_ip in learner_ips:
                if KaiwuDRLDefine.PROCESS_ADD == learner_add_or_reduce:
                    policy.add_learner_proxy_list(learner_ip)
                elif KaiwuDRLDefine.PROCESS_REDUCE == learner_add_or_reduce:
                    policy.reduce_learner_proxy_list(learner_ip)
                else:
                    pass

        # 操作完成后需要继续线程活动
        self.logger.info(
            f"kaiwu_rl_helper {actor_add_or_reduce} {actor_ips} {learner_add_or_reduce} {learner_ips} "
            f"expansion success"
        )

        return True

    # 返回policies
    def get_policies(self):
        return self.policies

    # 获取线程ID
    def get_pid(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

        return -1

    @property
    def identity(self):
        return f"kaiwu_rl_helper_{self.slot_id}"

    def get_current_agents(self):
        """
        获取当前配置的policy里的Agent
        """
        if (CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL) or (
            CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE
            and CONFIG.remote_agent_default_runtime_mode
            in [
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
            ]
        ):
            for policy, agent_wrapper in self.policy_agent_wrapper_maps.items():
                self.current_agents.append(agent_wrapper.get_model_object())

        elif CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE and CONFIG.remote_agent_default_runtime_mode not in [
            KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
            KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
        ]:
            # 机器上的device
            machine_device = get_machine_device_by_config(CONFIG.use_which_deep_learning_framework, CONFIG.svr_name)

            # 支持多个agent操作, 此时是多个agent_wrapper对象
            for policy_name, policy_conf in AppConf.get_app_conf(CONFIG.app, "policies").items():
                algo = policy_conf.algo
                model = AlgoConf.get_algo_conf(algo, "aisrv_agent")(
                    agent_type=CONFIG.svr_name,
                    device=machine_device,
                    logger=self.logger,
                    monitor=self.monitor_proxy if CONFIG.use_prometheus else None,
                )
                model.framework_handler = self

                self.current_agents.append(model)
        else:
            pass

    # ── obs 类方法（predict/exploit/reset/init_config）的统一分发 ──

    # predict/exploit 在 on-policy 模式下需要额外返回 model_version
    _ON_POLICY_METHODS = {"predict", "exploit"}

    def _dispatch_obs_local(self, method_name, agent, data):
        """
        本地 obs 方法统一分发（单机单进程 或 分布式下预测在 workflow 进程执行）

        替代原有的 predict_local / exploit_local / reset_local 三个独立方法，
        同时新增 init_config 支持。

        Args:
            method_name: 方法名（predict / exploit / reset / init_config）
            agent: Agent 对象
            data: 观测数据
        """
        agent_wrapper = self.policy_agent_wrapper_maps.get(agent.ctx.policy_name)
        result = getattr(agent_wrapper, method_name)(data)

        if method_name in self._ON_POLICY_METHODS:
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                return result, -1

        return result

    # train 函数, 单机单进程版本使用
    def train_local(self, agent, data, *args, **kargs):
        # 根据agent的policy获取对应的agent_wrapper（agent_wrapper内部有self.agent，无需传递）
        agent_ctx = agent.ctx
        agent_wrapper = self.policy_agent_wrapper_maps.get(agent_ctx.policy_name)
        (
            app_monitor_data,
            has_model_file_changed,
            model_file_id,
        ) = agent_wrapper.train_local(data, *args, **kargs)

        if app_monitor_data and isinstance(app_monitor_data, dict):
            self.app_monitor_data = app_monitor_data

        return app_monitor_data

    # 上报监控数据
    def train_predict_stat(self):
        """
        下面的情况需要在kaiwu_rl_helper线程上报监控
        1. 单机单进程
        2. 分布式, 但是是在workflow里进行预测, 利用, 加载模型
        """
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL or (
            CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE
            and CONFIG.remote_agent_default_runtime_mode
            in [
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
            ]
        ):
            if int(CONFIG.use_prometheus):
                now = time.time()
                if now - self.last_report_monitor_time >= CONFIG.prometheus_stat_per_minutes * 60:
                    monitor_data = self.get_training_metrics_local(use_app_monitor_data=False)
                    self.monitor_proxy.put_data({self.current_pid: monitor_data})
                    self.last_report_monitor_time = now

    def standard_load_last_new_model(self, agent, func=None, *args, **kargs):
        """
        单机单进程版本的没有learner/actor之间的model文件传递, 但是需要在评估时加载model文件
        注意：func 参数已废弃，不再使用（保留用于向后兼容）
        """
        try:
            agent_ctx = agent.ctx
            agent_wrapper = self.policy_agent_wrapper_maps.get(agent_ctx.policy_name)
            # 使用统一的 load_model_by_source，USER 模式
            agent_wrapper.load_model_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER, *args, **kargs)
        except Exception as e:
            raise RuntimeError(f"standard_load_last_new_model error")

    @contextmanager
    def preserve_config(self):
        """保存和恢复 CONFIG 的上下文管理器"""
        global CONFIG

        # 深度拷贝 CONFIG 的所有属性
        original_attrs = copy.deepcopy(CONFIG.__dict__)
        try:
            yield
        finally:
            # 恢复所有属性
            CONFIG.__dict__.update(original_attrs)

    # before_run函数
    def before_run(self):
        """
        由于在kaiwu_env的调用里可能存在修改CONFIG的配置情况, 因为是在同一个进程里调用的
        故需要对CONFIG对象做下备份操作
        """
        with self.preserve_config():
            self.env_wrapper = EnvWrapper(self.logger, self.monitor_proxy)
            client_address = self.simu_ctx.client_address.split(":")
            self.env_wrapper.init(client_address[0], client_address[1])

        # 在评测模式时传入需要的exam_random_seed参数
        random_generator = random.Random()
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            if CONFIG.random_seed != "default":
                random_generator.seed(int(CONFIG.random_seed))

        # 如果是KaiwuDRLDefine.WRAPPER_LOCAL模式需要走下面逻辑
        if (CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL) or (
            CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE
            and CONFIG.remote_agent_default_runtime_mode
            in [
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
            ]
        ):
            # policy_name 主要是和kaiwudrl/conf/app_conf.json设置一致
            self.policy_conf = AppConf.get_app_conf(CONFIG.app, "policies")

            # 创建agent_wrapper
            create_standard_agent_wrapper(
                self.policy_conf,
                self.policy_agent_wrapper_maps,
                None,
                self.logger,
                self.monitor_proxy,
            )

            # 在多种场景下多需要使用agent_wrapper的
            agent_wrapper = self.policy_agent_wrapper_maps.get(CONFIG.policy_name)

            # 单机单进程的需要执行的逻辑
            if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
                # 清空id_list文件, 否则文件会持续增长
                clear_id_list_file(framework=True)

                # 第一次保存模型时id的默认值即0
                agent_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK)

                # 更新id_list文件
                update_id_list(0, framework=True)

                # 清空使用者保存的文件目录
                clear_user_ckpt_dir()

                # eval下加载业务侧的模型文件
                if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
                    self.load_model_common_object = LoadModelCommon(self.logger)
                    self.load_model_common_object.set_policy_agent_wrapper_maps(self.policy_agent_wrapper_maps)
                    self.load_model_common_object.standard_load_last_new_model_by_framework_local()

        # 获取当前的agent对象
        self.get_current_agents()
        # 创建游戏环境和智能体
        self.init_agent_runtime(self.current_agents, [self.env_wrapper])

        # 如果单机单进程或者预测需要在aisrv的workflow进程执行的需要支持预加载功能
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL or (
            CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE
            and CONFIG.remote_agent_default_runtime_mode
            == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW
        ):
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                if CONFIG.preload_model:
                    # 遍历所有policy的agent_wrapper进行预加载, 确保每个policy都加载了预训练模型
                    for policy, policy_agent_wrapper in self.policy_agent_wrapper_maps.items():
                        success = policy_agent_wrapper.preload_model_file(
                            CONFIG.preload_model_dir, CONFIG.preload_model_id
                        )
                        if success:
                            self.logger.info(
                                f"kaiwu_rl_helper preload model file success, policy is {policy}, "
                                f"preload_model_dir is {CONFIG.preload_model_dir}, "
                                f"preload_model_id is {CONFIG.preload_model_id}"
                            )
                        else:
                            self.logger.error(
                                f"kaiwu_rl_helper preload model file failed, policy is {policy}, "
                                f"preload_model_dir is {CONFIG.preload_model_dir}, "
                                f"preload_model_id is {CONFIG.preload_model_id}"
                            )

                            # 如果失败了需要提前退出
                            error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR
                            self.send_process_stop_request(self.current_agents, error_code)
                            return False
        return True

    # 设置下run_time
    def init_agent_runtime(self, agents, envs):
        # 设置下agent阵营信息
        self.start_all_agents(agents)

    def workflow(self):
        """
        该函数主要是标准化调用的run函数, 由于是使用者调用的, 故这里需要加上处理Error, Warning的逻辑
        """
        error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_SUCCESS

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)

            try:
                # 记录运行时间
                process_start_time = time.monotonic()

                if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                    # 直接调用训练的workflow
                    AlgoConf.get_algo_conf(CONFIG.algo, "train_workflow")(
                        [self.env_wrapper],
                        self.current_agents,
                        self.logger,
                        self.monitor_proxy,
                        **{"process_index": self.simu_ctx.index},
                    )

                elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL:
                    # 直接调用配置的评估的workflow
                    AlgoConf.get_algo_conf(CONFIG.algo, "eval_workflow")(
                        [self.env_wrapper],
                        self.current_agents,
                        self.logger,
                        self.monitor_proxy,
                        **{"process_index": self.simu_ctx.index},
                    )

                elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
                    # 直接调用配置的评测的workflow
                    AlgoConf.get_algo_conf(CONFIG.algo, "exam_workflow")(
                        [self.env_wrapper],
                        self.current_agents,
                        self.logger,
                        self.monitor_proxy,
                        **{"process_index": self.simu_ctx.index},
                    )

                else:
                    self.logger.error(
                        f"kaiwu_rl_helper CONFIG.run_mode is not supported, please check CONFIG.run_mode, supported run_mode: {KaiwuDRLDefine.RUN_MODE_TRAIN}, {KaiwuDRLDefine.RUN_MODE_EVAL}, {KaiwuDRLDefine.RUN_MODE_EXAM}"
                    )

                error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_SUCCESS

            except RuntimeError:
                self.logger.exception(f"kaiwu_rl_helper workflow() RuntimeError Exception, ")

                error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR

            except Exception as e:
                self.logger.exception(f"kaiwu_rl_helper workflow() Exception {str(e)}")

                error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR

            finally:

                # aisrv与learner进行通信, 告诉其需要安全退出
                self.send_process_stop_request(self.current_agents, error_code)
                self.logger.info(f"kaiwu_rl_helper aisrv send process_stop_request {error_code} to learner success")

                self.logger.info("kaiwu_rl_helper finally")

                time.sleep(CONFIG.handle_sigterm_sleep_seconds)

    # run 主函数
    def run(self):
        if not self.before_run():
            self.logger.error(f"kaiwu_rl_helper before_run failed, so return")
            return

        # 注意：不要在这里重置 self.agent_ctxs
        # before_run() 内部调用 init_agent_runtime() → start_all_agents() 已经填充了 agent_ctxs，
        # 如果重置会导致 self_play=True 场景下 send_train_data 对空字典
        # 调用 next(iter(self.agent_ctxs.values())) 抛出 StopIteration。
        self.simu_ctx.agent_ctxs = self.agent_ctxs

        """
        调用业务侧的主函数代码
        """
        self.workflow()

    def predict_detail_for_multi_agent(self, agent, predict_data, msg_type):
        """
        针对多个agent的单个predict/exploit请求
        """
        if not agent or not predict_data:
            return None

        if len(agent) != len(predict_data):
            return None

        # 采用进程池处理, 并行返回结果
        with self.pool as pool:
            # 将agent和predict_data打包成一个元组的列表
            agent_data_list = list(zip(agent, predict_data, msg_type))
            # 在进程池中并行执行predict_detail_for_multi_agent函数
            results = pool.map(self.predict_detail_for_multi_agent, agent_data_list)
            # 返回结果列表
            return results

    def predict_detail_for_multi_agent(self, agent_data):
        """
        针对单个agent的单个predict/exploit请求
        函数参数里agent_data其实是agent和predict_data一起的
        """
        if not agent_data:
            return None

        agent, predict_data, msg_type = agent_data
        return self.predict_detail_for_single_agent(agent, predict_data, msg_type)

    def predict_detail_for_single_agent(self, agent, predict_data, msg_type):
        """
        针对单个agent的单个predict/exploit请求
        函数参数里agent和predict_data是分开的
        """
        if not agent or not predict_data:
            return None

        return self.predict_detail_inner(agent, predict_data, msg_type)

    def smart_serialize(self, data: Any) -> Tuple[bytes, str]:
        """三层递进式序列化策略"""
        try:
            # 第一层：带自定义处理的增强版msgpack
            packed = msgpack.packb(data, default=self.obsdata_convert, use_bin_type=True, strict_types=True)  # 注入自定义逻辑
            return packed, KaiwuDRLDefine.SERIALIZE_TYPE_MSGPACK_EXTEND

        except (TypeError, msgpack.PackException) as e:
            # 第二层：尝试原生msgpack（处理基础类型）
            try:
                packed = msgpack.packb(data, use_bin_type=True, strict_types=True)
                return packed, KaiwuDRLDefine.SERIALIZE_TYPE_MSGPACK
            except (TypeError, msgpack.PackException):
                # 第三层：最终回退到dill
                return dill.dumps(data), KaiwuDRLDefine.SERIALIZE_TYPE_DILL

        except Exception as unexpected_error:
            # 捕获其他未知异常（如内存错误等）
            self.logger.error(f"Unexpected serialization error: {unexpected_error}")
            return dill.dumps(data), KaiwuDRLDefine.SERIALIZE_TYPE_DILL

    def obsdata_convert(self, obj):
        """ObsData 专用转换器"""
        if type(obj).__name__ == "ObsData":  # 通过类名识别
            # 动态捕获所有公有属性（排除私有属性和方法）
            attrs = {k: v for k, v in vars(obj).items() if not k.startswith("_") and not callable(v)}
            return ExtType(KaiwuDRLDefine.OBSDATA_EXT_TYPE, msgpack.packb(attrs, use_bin_type=True))  # 仅序列化数据字段
        raise TypeError(f"Unsupported type: {type(obj)}")

    def predict_detail_inner(self, agent, predict_data, msg_type):
        """
        由于需要多处调用, 则抽取成公共函数
        """

        serialize_data, serialize_type = self.smart_serialize(predict_data)

        # 传递过来的predict_data先进行序列化再处理, 因为可能是不同的数据结构的, msgpack只能是aisrv-->actor方向, 反之则因为无法序列化python对象失败
        predict_datas = {
            KaiwuDRLDefine.MESSAGE_TYPE: msg_type,
            KaiwuDRLDefine.MESSAGE_VALUE: {
                "model_policy": agent.ctx.model_policy,
                "predict_data": serialize_data,
                "serialize_type": serialize_type,
            },
        }

        """
        针对单个agent的数据预测
        """
        agent_ctx = agent.ctx
        agent_id = agent_ctx.agent_id
        for policy_name in agent_ctx.policy:
            # 调用AsyncPolicy的send_pred_data函数
            success, actor_address = agent_ctx.policy[policy_name].send_pred_data(
                self.slot_id, predict_datas, agent_ctx
            )
            # self.logger.debug(f'kaiwu_rl_helper aisrv {self.slot_id} send to actor: {predict_datas}')
            if not success:
                self.logger.error(
                    f"kaiwu_rl_helper policy_name {policy_name} agent_id {agent_id} "
                    f"send_pred_data to actor {actor_address} failed"
                )
                continue

        agent_ctx.pred_output = {}
        for policy_name in agent_ctx.policy:
            pred_output = agent_ctx.policy[policy_name].get_pred_result(self.slot_id, agent_ctx)

            # self.logger.debug(f'kaiwu_rl_helper aisrv {self.slot_id} recv from actor: {pred_output}')
            if not pred_output:
                self.logger.error("kaiwu_rl_helper get_pred_result failed")
            else:
                agent_ctx.pred_output[policy_name] = pred_output

        # 提取数据
        preds = []
        for policy_name in agent_ctx.policy:
            pred = dill.loads(agent_ctx.pred_output[policy_name][agent_id]["pred"])
            preds.append(pred)
            self.from_actor_model_version = agent_ctx.pred_output[policy_name][agent_id]["model_version"]

        """
        由于on-policy的样本组件时需要样本的版本号, 故这里返回值加上了样本的版本号
        1. predict, 返回业务数据和model_version
        2. exploit, 返回业务数据和model_version
        3. reset/init_config, 返回业务数据
        """
        if msg_type in [
            KaiwuDRLDefine.MESSAGE_PREDICT,
            KaiwuDRLDefine.MESSAGE_EXPLOIT,
        ]:
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                return preds[0], self.from_actor_model_version
            else:
                return preds[0]
        elif msg_type in [KaiwuDRLDefine.MESSAGE_RESET, KaiwuDRLDefine.MESSAGE_INIT_CONFIG]:
            return preds[0]
        else:
            raise NotImplementedError(f"unsupported msg_type: {msg_type}")

    def get_model_info_from_workflow(self, id):
        """
        直接调用model_file_sync的get_current_available_model_file函数
        """
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            # 如果是在actor远程执行的REMOTE_AGENT_RUNTIME_MODE_REMOTE_ACTOR_PREDICT则不需要启动model_file_sync的
            if CONFIG.remote_agent_default_runtime_mode in [
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
            ]:
                return self.model_file_sync_wrapper.ckpt_sync_warper.get_current_available_model_file(id, self.logger)
            else:
                return -1, None
        else:
            return -1, None

    def load_model_local(self, agent, path=None, id="1", *args, **kwargs):
        """
        单机模式下加载模型，调用 agent_wrapper.load_model_by_source
        """
        id, path = self.get_model_info_from_workflow(id)
        # 只有本次有存在model文件才进行加载
        if path is not None:
            agent_ctx = agent.ctx
            agent_wrapper = self.policy_agent_wrapper_maps.get(agent_ctx.policy_name)
            return agent_wrapper.load_model_by_source(
                path=path, id=id, source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER
            )
        return None

    def save_model_local(self, agent, path=None, id="1", *args, **kwargs):
        """
        单机模式下保存模型，调用 agent_wrapper.save_param_by_source
        """
        agent_ctx = agent.ctx
        agent_wrapper = self.policy_agent_wrapper_maps.get(agent_ctx.policy_name)

        # 使用统一的 save_param_by_source
        source = kwargs.pop("source", KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER)
        return agent_wrapper.save_param_by_source(path=path, id=id, source=source, **kwargs)

    # predict_detail函数, 因为exploit和predict函数通用
    def predict_detail(self, agent, predict_data, msg_type):
        """
        根据agent的输入参数的类型, 采取不同的操作
        1. agent是Agent对象, 采用单次预测即可
        2. agent是列表, 则采用并行的去预测
        """
        if isinstance(agent, list):
            return self.predict_detail_for_multi_agent(agent, predict_data, msg_type)
        else:
            return self.predict_detail_for_single_agent(agent, predict_data, msg_type)

    # ── obs 类方法名 → 远程消息类型的映射 ──
    _OBS_METHOD_MSG_TYPE = {
        "predict": KaiwuDRLDefine.MESSAGE_PREDICT,
        "exploit": KaiwuDRLDefine.MESSAGE_EXPLOIT,
        "reset": KaiwuDRLDefine.MESSAGE_RESET,
        "init_config": KaiwuDRLDefine.MESSAGE_INIT_CONFIG,
    }

    def _dispatch_obs_remote(self, method_name, agent, data):
        """
        远程 obs 方法统一分发（集群版本，aisrv → actor 网络通信）

        替代原有的 predict / exploit / reset 三个独立方法，
        同时新增 init_config 支持。

        Args:
            method_name: 方法名（predict / exploit / reset / init_config）
            agent: Agent 对象
            data: 观测数据
        """
        if not data:
            return None

        msg_type = self._OBS_METHOD_MSG_TYPE.get(method_name)
        if msg_type is None:
            self.logger.error(f"kaiwu_rl_helper _dispatch_obs_remote unsupported method: {method_name}")
            return None

        return self.predict_detail(agent, data, msg_type)

    # 发送样本数据, 集群版本使用
    def send_train_data(self, agent, train_data, train_data_prioritized):
        if not train_data:
            return

        # 如果是没有启动learner进程, 则不需要发送样本请求
        if not CONFIG.need_to_start_learner:
            self.logger.error(f"kaiwu_rl_helper need_to_start_learner set false but send_train_data, please check")
            return

        """
        采用下面规则:
        1. 如果是对战, 默认是主策略上发送样本数据, 此时只有1个learner
        2. 如果是非对战的, 在其各自的策略上发送样本数据
        """
        if CONFIG.self_play:
            agent_ctx = next(iter(self.agent_ctxs.values()))
        else:
            agent_ctx = agent.ctx

        policy = agent_ctx.policy[agent_ctx.main_id]

        # 防御编程：确保numpy数组是C-contiguous的
        # 使用flags.c_contiguous检查，仅在需要时才调用ascontiguousarray
        train_data_safe = []
        for item in train_data:
            if isinstance(item, np.ndarray):
                if item.flags.c_contiguous:
                    train_data_safe.append(item)
                else:
                    train_data_safe.append(np.ascontiguousarray(item))
            else:
                train_data_safe.append(item)

        train_data_detail = {
            "train_data": train_data_safe,
            "train_data_prioritized": train_data_prioritized,
        }
        train_data_msg = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_TRAIN,
            KaiwuDRLDefine.MESSAGE_VALUE: train_data_detail,
        }

        success = policy.send_train_data(train_data_msg, agent_ctx)
        if not success:
            current_time = time.time()
            if current_time - self.last_error_time > self.error_interval:
                self.logger.warning(f"kaiwu_rl_helper policy.send_train_data failed, please check")
                self.last_error_time = current_time

    # 发送保存模型文件到learner
    def send_save_model_file_data(self, agent, path=None, id=None):

        # 如果是没有启动learner进程, 则不需要发送save_model请求
        if not CONFIG.need_to_start_learner:
            self.logger.error(
                f"kaiwu_rl_helper need_to_start_learner set false but send_save_model_file_data, please check"
            )
            return

        """
        采用下面规则:
        1. 如果是对战, 默认是主策略上发送样本数据, 此时只有1个learner
        2. 如果是非对战的, 在其各自的策略上发送样本数据
        """
        if CONFIG.self_play:
            agent_ctx = next(iter(self.agent_ctxs.values()))
        else:
            agent_ctx = agent.ctx

        policy = agent_ctx.policy[agent_ctx.main_id]
        save_model_data_detail = {
            "ip": f"{self.ip}_{self.slot_id}",
            "path": path,
            "id": id,
        }
        save_model_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_SAVE_MODEL,
            KaiwuDRLDefine.MESSAGE_VALUE: save_model_data_detail,
        }

        success = policy.send_control_data(save_model_data, agent_ctx)
        if not success:
            self.logger.error(f"kaiwu_rl_helper send_save_model_file_data, please check")

    # 发送获取训练指标的请求到 learner
    def get_training_metrics(self, agent, path=None, id=None):

        get_training_metrics_detail = {"ip": f"{self.ip}_{self.slot_id}"}

        get_training_metrics_request_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_GET_TRAINING_METRICS,
            KaiwuDRLDefine.MESSAGE_VALUE: get_training_metrics_detail,
        }

        # FIXME：是否需要考虑 self_play
        if CONFIG.self_play:
            agent_ctx = next(iter(self.agent_ctxs.values()))
        else:
            agent_ctx = agent.ctx

        policy = agent_ctx.policy[agent_ctx.main_id]

        # 调用 AsyncPolicy 的 get_training_metrics_request 函数
        training_metrics = policy.get_training_metrics(self.slot_id, get_training_metrics_request_data, agent_ctx)

        # self.logger.debug(f'kaiwu_rl_helper aisrv {self.slot_id} recv from learner: {training_metrics}')
        if not training_metrics:
            self.logger.info("kaiwu_rl_helper get_training_metrics failed")

        return training_metrics

    # 获取训练指标 单机版本
    def get_training_metrics_local(self, use_app_monitor_data=True):

        predict_count = 0
        train_count = 0
        load_model_count = 0
        for policy, agent_wrapper in self.policy_agent_wrapper_maps.items():
            predict_count += agent_wrapper.predict_stat
            load_model_count += agent_wrapper.load_model_stat
            current_train_count, preload_model_train_count = agent_wrapper.train_stat
            train_count += current_train_count

        sample_production_and_consumption_ratio = 0
        if predict_count > 0:
            sample_production_and_consumption_ratio = round(
                (train_count - preload_model_train_count) / predict_count, 3
            )

        training_metrics = {
            KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT: predict_count,
            KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_SUCC_CNT: load_model_count,
        }

        # 单机单进程的情况下需要给出的监控项
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            training_metrics[KaiwuDRLDefine.MONITOR_TRAIN_SUCCESS_CNT] = train_count
            training_metrics[KaiwuDRLDefine.MONITOR_TRAIN_GLOBAL_STEP] = train_count
            training_metrics[
                KaiwuDRLDefine.SAMPLE_PRODUCTION_AND_CONSUMPTION_RATIO
            ] = sample_production_and_consumption_ratio
            training_metrics[KaiwuDRLDefine.MONITOR_SENDTO_REVERB_SUCC_CNT] = train_count
            training_metrics[KaiwuDRLDefine.SAMPLE_RECEIVE_CNT] = train_count
            # 在单机单进程的情况下, model文件加载和预测次数是一致的
            training_metrics[KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_SUCC_CNT] = predict_count

        if use_app_monitor_data:
            for key, value in self.app_monitor_data.items():
                training_metrics[key] = float(value)

        return training_metrics

    # 发送process_stop请求, 单机版本和集群版本都会使用到
    def send_process_stop_request(self, agents, error_code):
        """
        agents, 智能体集合
        error_code, 退出码, 分为正确的和错误的退出码
        """

        # 如果是没有启动learner进程, 则不需要发送process_stop请求
        if not CONFIG.need_to_start_learner:
            self.logger.error(
                f"kaiwu_rl_helper need_to_start_learner set false but send_process_stop_request, please check"
            )
            return

        process_stop_request_detail = {"ip": f"{self.ip}_{self.slot_id}", "error_code": error_code}

        process_stop_request_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_PROCESS_STOP,
            KaiwuDRLDefine.MESSAGE_VALUE: process_stop_request_detail,
        }

        """
        采用下面规则:
        1. 如果是对战, 默认是主策略上发送样本数据, 此时只有1个learner
        2. 如果是非对战的, 在其各自的策略上发送样本数据
        """
        if CONFIG.self_play:
            agent_ctx = next(iter(self.agent_ctxs.values()))
        else:
            agent_ctx = agents[0].ctx
        policy = agent_ctx.policy[agent_ctx.main_id]

        success = policy.send_control_data(process_stop_request_data, agent_ctx)
        if not success:
            self.logger.error(f"kaiwu_rl_helper send_process_stop_request failed, please check")

    # 发送load_model请求, 集群版本使用
    def send_load_model_file_data(self, agent, path=None, id=None):

        # id必须为正常的
        if not check_id_valid(id):
            self.logger.error(f"kaiwu_rl_helper send_load_model_file_data failed, id {id} is not valid, please check")
            return

        # 如果是没有启动learner进程, 则不需要进行model文件同步, 则不需要执行load_model操作
        if not CONFIG.need_to_start_learner:
            if id == KaiwuDRLDefine.ID_LATEST or id == KaiwuDRLDefine.ID_RANDOM:
                self.logger.error(
                    f"kaiwu_rl_helper need_to_start_learner set false but load_model by {id}, please check"
                )
                return

        # 无论是对战或者是非对战场景下预测进程都需要load_model
        agent_ctx = agent.ctx

        # 赋值加载模型的策略
        agent.ctx.model_policy = id

        load_model_data_detail = {
            "ip": f"{self.ip}_{self.slot_id}",
            "policy": agent_ctx.main_id,
            "path": path,
            "id": id,
        }
        load_model_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_LOAD_MODEL,
            KaiwuDRLDefine.MESSAGE_VALUE: load_model_data_detail,
        }

        policy = agent_ctx.policy[agent_ctx.main_id]

        success, actor_address = policy.send_pred_data(self.slot_id, load_model_data, agent_ctx)
        if not success:
            self.logger.error(f"kaiwu_rl_helper send_load_model_file_data to actor {actor_address} failed")

    def stop(self):
        self.exit_flag.value = True
        for __, policy in self.policies.items():
            policy.stop()

        # 上报监控指标
        monitor_data = {}
        for key, value in self.app_monitor_data.items():
            monitor_data[key] = value

        if monitor_data:
            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        time.sleep(1)

        self.logger.info("kaiwu_rl_helper success stop")

    def normalize_policy_names(self, ids):
        assert isinstance(ids, (str, list)), "only str or list of str is supported"
        if isinstance(ids, str):
            ids = [ids]
        return ids

    def stop_all_agents(self, agents):
        """
        停止所有的agents
        """
        for i, agent in enumerate(agents):
            self.stop_agent(i)

    # 单个agent_id的停止
    def stop_agent(self, agent_id):
        self.logger.info(f"kaiwu_rl_helper stop agent {agent_id}")
        agent_ctx = self.agent_ctxs[agent_id]

        policy_names = list(agent_ctx.policy.keys())
        for policy_name in policy_names:
            if agent_ctx.policy[policy_name].need_train():
                if not self.use_sample_server:
                    agent_ctx.expr_processor[policy_name].finalize()

        del self.agent_ctxs[agent_id]

    def load_model_in_eval_or_exam_mode(self, agent):
        """
        部分场景下为了追求性能, 评估的eval_workflow进程是直接调用exploit函数, 故需要先加载模型文件
        """
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            if agent.remote_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW:
                path = CONFIG.eval_model_dir
                id = CONFIG.eval_model_id
                try:
                    agent.load_model(path=path, id=id, framework=True)
                    self.logger.info(f"kaiwu_rl_helper load_model success, path is {path}, id is {id}")
                except Exception as e:
                    self.logger.exception(f"kaiwu_rl_helper load_model failed, path is {path}, id is {id}")

    def start_all_agents(self, agents):
        """
        启动所有的agents, 主要是对齐agent和policy的关系
        智能体个数区分为:
        1. 单智能体
        2. 2个智能体, 对战模式
        3. 2个以上智能体, 大乱斗模式
        """
        multi_agent = False
        if len(agents) > 2:
            multi_agent = True

        for i, agent in enumerate(agents):
            # 这里强制做了下赋值即framework_handler设置为self
            agent.framework_handler = self

            # agent的远程运行模式, remote_wrapper会根据该配置选择执行的方式
            agent.remote_runtime_mode = CONFIG.remote_agent_default_runtime_mode

            self.start_agent(i, multi_agent)
            agent.ctx = self.agent_ctxs[i]

            # 某些评估场景下加载模型文件
            self.load_model_in_eval_or_exam_mode(agent)

    # 单个agent_id的启动
    def start_agent(self, agent_id, multi_agent=False):
        """
        形如以下配置
        "policies": {
                    "train": {
                        "policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    },
                    "predict": {
                        "policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    }

        """
        agent_ctx = AgentContext()
        agent_ctx.done = False
        agent_ctx.agent_id = agent_id

        # 设置主要main_id, policy_names为策略列表
        np.random.seed(int(time.time() * 1000) % (2**20))
        policy_names = list(AppConf.get_app_conf(CONFIG.app, "policies").keys())

        # 每一个agent在启动时就需要确定唯一的policy，但是两边对弈的agent可以是不同policy
        if int(CONFIG.self_play):
            if not multi_agent:
                # 如果agent为self_play_agent那么其策略为策略列表中的对应策略
                if agent_id == CONFIG.self_play_agent_index:
                    agent_ctx.main_id = policy_names[CONFIG.self_play_agent_index]
                    policy_names = [policy_names[CONFIG.self_play_agent_index]]
                    assert agent_ctx.main_id == CONFIG.self_play_policy, "Check your config of self_play_policy"

                elif agent_id == CONFIG.self_play_old_agent_index:
                    # 当agent_id为对手策略时，80%设置为新策略，20%为旧策略
                    if np.random.uniform() <= (1 - float(CONFIG.self_play_new_ratio)):
                        agent_ctx.main_id = policy_names[CONFIG.self_play_old_agent_index]
                        policy_names = [policy_names[CONFIG.self_play_old_agent_index]]
                        assert (
                            agent_ctx.main_id == CONFIG.self_play_old_policy
                        ), "Check your config of self_play_old_policy"
                    else:
                        agent_ctx.main_id = policy_names[CONFIG.self_play_agent_index]
                        policy_names = [policy_names[CONFIG.self_play_agent_index]]
                        assert agent_ctx.main_id == CONFIG.self_play_policy, "Check your config of self_play_policy"
            else:
                agent_ctx.main_id = policy_names[agent_id]
                policy_names = [policy_names[agent_id]]
        else:
            # 如果不是self-play模式，那么agent自动加载第一种policy
            agent_ctx.main_id = policy_names[0]
            policy_names = [policy_names[0]]

        self.logger.info(f"kaiwu_rl_helper start agent {agent_id} with {policy_names[0]}")

        """ policy conf, 形如
                    "train_one": {
                        "policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                        "algo": "ppo",
                        "state": "app.gym.gym_proto.GymState",
                        "action": "app.gym.gym_proto.GymAction",
                        "reward": "app.gym.gym_proto.GymReward",
                        "actor_network": "app.gym.gym_network.GymDeepNetwork",
                        "learner_network": "app.gym.gym_network.GymDeepNetwork",
                        "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper",
                        "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
                    }
        """
        agent_ctx.policy_conf = {}
        """
        policy, 形如"policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
        """
        agent_ctx.policy = {}
        # 预测的响应结果
        agent_ctx.pred_output = {}

        agent_ctx.start_time = time.monotonic()

        # aisrv发送给actor的message id, 从1自增
        agent_ctx.message_id = 1

        # aisrv发送给actor的model_version, 由actor负责赋值
        agent_ctx.model_version = -1

        # 表征该agent是采用哪种模型? latest, random, 某个具体的ID, 默认是latest
        agent_ctx.model_policy = KaiwuDRLDefine.ID_RANDOM

        # policy_names的列表长度根据运行模式不一致, 比如self-play是1, 非self-play的需要看具体情况
        for policy_name in policy_names:
            policy_conf = AppConf.get_app_conf(CONFIG.app, "policies")[policy_name]
            policy = self.policies[policy_name]
            agent_ctx.policy_conf[policy_name] = policy_conf
            agent_ctx.policy[policy_name] = policy
            agent_ctx.policy_name = policy_name

            if policy.need_train():
                assert hasattr(policy_conf, "algo"), "trainable policy need to specify algo"
                """
                 {
                     "ppo": {
                         "actor_agent": "kaiwudrl.common.algorithms.model.Model",
                         "learner_agent": "kaiwudrl.common.algorithms.model.Model",
                         "trainer": "kaiwudrl.server.learner.ppo_trainer.PPOTrainer",
                         "predictor": "kaiwudrl.server.actor.ppo_predictor.PPOPredictor",
                         "expr_processor": "kaiwudrl.common.algorithms.ppo_processor.PPOProcessor",
                         "default_config": "kaiwudrl.common.algorithms.ppo.PPODefaultConfig"
                     }
                 }
                 """

        self.agent_ctxs[agent_id] = agent_ctx

    def handle_sigterm(self, sig, frame):
        self.stop()
        agent_wrapper = self.policy_agent_wrapper_maps.get(CONFIG.policy_name)
        self.logger.info(f"kaiwu_rl_helper {self.current_pid} is starting to handle the SIGTERM signal.")
        if agent_wrapper is not None:
            agent_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_BY_SIGTERM)

        # 处理完保存最新模型,等待其他进程工作,避免pod提前退出
        time.sleep(CONFIG.handle_sigterm_sleep_seconds)
