#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.config.algo_conf import AlgoConf
from common_python.config.config_control import CONFIG
from kaiwudrl.common.algorithms.standard_agent_wrapper_builder import (
    StandardAgentWrapperBuilder,
)
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.common_func import TimeIt
from kaiwudrl.common.checkpoint.model_file_common import process_stop_write_file
from kaiwudrl.common.utils.common_func import get_machine_device_by_config


def create_standard_agent_wrapper(policy_conf, policy_agent_wrapper_maps, replay_buffer_wrapper, logger, monitor_proxy):
    """
    建立agent_wrapper, 支持多个agent操作, 多个agent_wrapper对象, 标准化场景调用
    actor_proxy_local, actor, learner进程会调用
    """

    # 在用户 agent 代码加载前激活文件操作防护(仅 eval/exam 模式且配置开启时)
    if CONFIG.run_mode in (KaiwuDRLDefine.RUN_MODE_EVAL, KaiwuDRLDefine.RUN_MODE_EXAM) and getattr(
        CONFIG, "wrapper_enable_file_guard", False
    ):
        try:
            from kaiwudrl.common.security.file_guard import get_file_guard

            file_guard = get_file_guard(logger)
            if not file_guard.is_active():
                file_guard.activate()
                logger.info("FileOperationGuard activated before agent creation")
        except Exception as e:
            logger.warning(f"FileOperationGuard activation failed: {e}")

    try:
        # 机器上的device
        machine_device = get_machine_device_by_config(CONFIG.use_which_deep_learning_framework, CONFIG.svr_name)

        # 支持多个agent操作, 此时是多个agent_wrapper对象
        for policy_name, _policy_conf in policy_conf.items():
            algo = _policy_conf.algo
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
                agent = AlgoConf.get_algo_conf(algo, "actor_agent")(
                    agent_type=CONFIG.svr_name,
                    device=machine_device,
                    logger=logger,
                    monitor=monitor_proxy if CONFIG.use_prometheus else None,
                )
            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                agent = AlgoConf.get_algo_conf(algo, "aisrv_agent")(
                    agent_type=CONFIG.svr_name,
                    device=machine_device,
                    logger=logger,
                    monitor=monitor_proxy if CONFIG.use_prometheus else None,
                )
            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                agent = AlgoConf.get_algo_conf(algo, "learner_agent")(
                    agent_type=CONFIG.svr_name,
                    device=machine_device,
                    logger=logger,
                    monitor=monitor_proxy if CONFIG.use_prometheus else None,
                )
            else:
                continue

            agent_wrapper = StandardAgentWrapperBuilder().create_agent_wrapper(agent, logger)
            agent.framework_handler = agent_wrapper
            # self.workflow = AlgoConf.get_algo_conf(CONFIG.algo, "actor_workflow")()

            if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
                agent_wrapper.build_predict_graph(input_tensors)
                agent_wrapper.add_predict_hooks(predict_hooks())
                agent_wrapper.create_predict_session()

                global_step = agent_wrapper.get_global_step()

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
                # 注意单机单进程可能遇见
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    agent_wrapper.set_dataset(replay_buffer_wrapper)
                agent_wrapper.build_model()

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
                # 注意单机单进程可能遇见
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    agent_wrapper.set_dataset(replay_buffer_wrapper)

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TCNN == CONFIG.use_which_deep_learning_framework:
                pass

            elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT == CONFIG.use_which_deep_learning_framework:
                # 注意单机单进程可能遇见
                if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                    agent_wrapper.set_dataset(replay_buffer_wrapper)
                agent_wrapper.build_model()

            else:
                logger.error(
                    f"error use_which_deep_learning_framework "
                    f"{CONFIG.use_which_deep_learning_framework}, only support "
                    f"{KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TCNN}, {KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH}, "
                    f"{KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX}, "
                    f"{KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE}",
                )

                continue

            policy_agent_wrapper_maps[policy_name] = agent_wrapper

            logger.info(f"policy_name {policy_name}, algo {algo}, agent_wrapper is {agent_wrapper.name}")

    except Exception as e:
        logger.exception(f" failed to run create_standard_agent_wrapper. exit. Error is: {e}, ")

        # 报错后让其提前退出去
        error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_ERROR
        process_stop_write_file(error_code, logger)
