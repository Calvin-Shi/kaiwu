#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.common_func import TimeIt

from common_python.logging.kaiwu_logger import g_not_server_label
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    stop_process_by_pid,
    clean_dir,
)

from kaiwudrl.common.checkpoint.model_file_common import (
    model_file_signature_verify,
    check_path_id_valid,
    get_checkpoint_id_by_re,
)


class LoadModelCommon(object):
    """
    该类主要是加载model文件公共类, 因为存在有actor, actor_proxy_local, aisrv的3个进程使用, 故将代码单独提出公共的, 只是维护一份即可
    1. local的主要是aisrv调用
    2. cluster的主要是actor, actor_proxy_local调用
    """

    def __init__(self, logger) -> None:

        # 下面是因为需要在使用时用到的变量, 故该类里只是定义, 由调用者进行赋值

        # policy和agent_wrapper对象的map, 为了支持多agent
        self.policy_agent_wrapper_maps = None
        self.model_file_sync_wrapper = None
        self.logger = logger

        # 统计使用
        self.actor_load_last_model_succ_cnt = 0
        self.actor_load_last_model_error_cnt = 0
        self.actor_load_last_model_cost_ms = 0

        # actor_predict_count, actor的predict进程数目
        if CONFIG.remote_agent_default_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT:

            if CONFIG.self_play:
                self.actor_predict_count = 2
            else:
                self.actor_predict_count = 1
        elif CONFIG.remote_agent_default_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_ACTOR_PREDICT:
            self.actor_predict_count = CONFIG.actor_predict_process_num
        else:
            pass

        # 增加特性是在eval模式下如果失败需要退出的逻辑
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            self.process_pid_list = []

    def set_actor_load_last_model_cost_ms(self, actor_load_last_model_cost_ms):
        self.actor_load_last_model_cost_ms = actor_load_last_model_cost_ms

    def get_actor_load_last_model_cost_ms(self):
        return self.actor_load_last_model_cost_ms

    def get_actor_load_last_model_succ_cnt(self):
        return self.actor_load_last_model_succ_cnt

    def get_actor_load_last_model_error_cnt(self):
        return self.actor_load_last_model_error_cnt

    def set_policy_agent_wrapper_maps(self, policy_agent_wrapper_maps):
        self.policy_agent_wrapper_maps = policy_agent_wrapper_maps

    def set_model_file_sync_wrapper(self, model_file_sync_wrapper):
        self.model_file_sync_wrapper = model_file_sync_wrapper

    def preload_model_file(self, policy_agent_wrapper_maps):
        """
        预加载模式功能是指将预先训练好的baseline文件加载到KaiwuDRL里, 只是learner需要处理, actor会通过learner<-->actor之间的model文件同步在某个时间阈值后替换
        1. tensorflow, 该框架自动支持
        2. pytorch, 需要手工调用下函数

        使用方法:
        1. 需要在/data/ckpt/app_algo下放置需要设置的model文件
        2. 修改/data/ckpt/app_algo下checkpoint文件内容, 指向1中的model文件

        如果是learner, 则默认CONFIG.policy_name对应的agent_wrapper加载
        如果是aisrv/actor, 则需要policy_agent_wrapper_maps里所有的agent_wrapper加载

        """

        if not int(CONFIG.preload_model):
            return False

        if (
            KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
            or KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT == CONFIG.use_which_deep_learning_framework
            or KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
        ):
            self.logger.info(f"predict tensorflow preload, not need to call function")
            return True

        elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
            if not check_path_id_valid(CONFIG.preload_model_dir, CONFIG.preload_model_id):
                self.logger.error(
                    f"predict pytorch preload, but preload_model_dir {CONFIG.preload_model_dir} or "
                    f"preload_model_id {CONFIG.preload_model_id} not valid, please check"
                )
                return False

            # learner调用预加载模型
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                agent_wrapper = policy_agent_wrapper_maps.get(CONFIG.policy_name)
                success = agent_wrapper.preload_model_file(CONFIG.preload_model_dir, CONFIG.preload_model_id)
                if success:
                    self.logger.info(
                        f"predict pytorch preload model file success, policy is {CONFIG.policy_name}, "
                        f"preload_model_dir is {CONFIG.preload_model_dir}, "
                        f"preload_model_id is {CONFIG.preload_model_id}"
                    )
                else:
                    self.logger.error(
                        f"predict pytorch preload model file failed, policy is {CONFIG.policy_name}, "
                        f"preload_model_dir is {CONFIG.preload_model_dir}, "
                        f"preload_model_id is {CONFIG.preload_model_id}"
                    )
                return success
            # aisrv/actor调用加载模型
            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                success = False
                for policy, agent_wrapper in policy_agent_wrapper_maps.items():
                    success = agent_wrapper.load_model_by_source(
                        path=CONFIG.preload_model_dir,
                        id=CONFIG.preload_model_id,
                        source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK,
                    )
                    if success:
                        self.logger.info(
                            f"predict pytorch preload model file success, policy is {policy}, "
                            f"preload_model_dir is {CONFIG.preload_model_dir}, "
                            f"preload_model_id is {CONFIG.preload_model_id}"
                        )
                    else:
                        # 如果是报错, 则提前退出
                        self.logger.error(
                            f"predict pytorch preload model file failed, policy is {policy}, "
                            f"preload_model_dir is {CONFIG.preload_model_dir}, "
                            f"preload_model_id is {CONFIG.preload_model_id}"
                        )
                        return False
                return success
            else:
                self.logger.error(f"predict pytorch preload, not support {CONFIG.svr_name}, please check")
                return False

        else:
            self.logger.error(
                f"predict preload just not support {CONFIG.use_which_deep_learning_framework}, "
                f"support list is KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE, KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT,"
                f"KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX, KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH"
            )
            return False

        return True

    def standard_load_last_new_model_by_framework_local(self):
        """
        该函数只是在评估时调用的, 单机单进程版本, aisrv调用
        """
        models_path = CONFIG.eval_model_dir
        id = CONFIG.eval_model_id

        try:
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:

                if CONFIG.digital_signature_verification:
                    if not model_file_signature_verify(models_path):
                        self.logger.error(
                            f"kaiwu_rl_helper run mode is {CONFIG.run_mode} model_file_signature_verify "
                            f"from {models_path} failed, so exit"
                        )
                        return False
                    else:
                        self.logger.info(
                            f"kaiwu_rl_helper run mode is {CONFIG.run_mode} model_file_signature_verify "
                            f"from {models_path} success"
                        )

                agent_wrapper = self.policy_agent_wrapper_maps.get(CONFIG.policy_name)
                success = agent_wrapper.load_model_by_source(
                    path=models_path, id=id, source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK
                )
                if success:
                    self.logger.info(
                        f"kaiwu_rl_helper standard_load_last_new_model_by_framework_local "
                        f"path {models_path}, id {id} success"
                    )

                    self.actor_load_last_model_succ_cnt += 1
                    return True

                else:
                    self.logger.error(
                        f"kaiwu_rl_helper standard_load_last_new_model_by_framework_local "
                        f"path {models_path}, id {id} failed"
                    )
                    return False

        except Exception as e:
            self.logger.exception(
                f"kaiwu_rl_helper standard_load_last_new_model_by_framework_local from {models_path}, "
                f"id {id} failed, error is {str(e)}"
            )
            return False

    def standard_load_last_new_model_by_framework(self, policy_name, models_path=None):
        """
        该函数只是在评估时调用的, 集群版本, actor/actor_proxy_local调用
        """

        # 因为tensorflow加载时是按照checkpoint文件来读取model文件的, 故id默认为0即可, pytorch是需要明确到path和id的
        id = 0
        if models_path is None:
            models_path = ""

        if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT == CONFIG.use_which_deep_learning_framework:
            if CONFIG.self_play_actor:
                models_path = (
                    f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/"
                    f"convert_models_{CONFIG.svr_name}/trt_weights.wts2_old"
                )
            else:
                models_path = (
                    f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/"
                    f"convert_models_{CONFIG.svr_name}/trt_weights.wts2"
                )

            # 判断文件不存在提前返回
            if not os.path.exists(models_path):
                return False

        else:
            """
            eval模式, 加载指定的eval_model_dir
            train模式, 标准化里交给使用者自动调用
            """
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
                """
                评估模式下, 下面的设置evale_model_dir的操作如下:
                1. 大规模场景, 因为actor进程是在不同的容器, 故直接赋值为CONFIG.eval_model_dir
                2. 小规模场景, 因为aisrv(actor)进程是在同一个容器里, 故需要按照self_play里的train_one和train_two来赋值
                """
                if (
                    CONFIG.remote_agent_default_runtime_mode
                    == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_ACTOR_PREDICT
                ):
                    models_path = CONFIG.eval_model_dir
                    id = CONFIG.eval_model_id
                elif CONFIG.remote_agent_default_runtime_mode in [
                    KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
                    KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
                ]:
                    if policy_name == CONFIG.self_play_policy:
                        models_path = CONFIG.eval_model_dir
                        id = CONFIG.eval_model_id
                    elif policy_name == CONFIG.self_play_old_policy:
                        models_path = CONFIG.self_play_eval_model_dir
                        id = CONFIG.self_play_eval_model_id
                    else:
                        pass
                else:
                    pass
            else:
                # 在训练模式下, on-policy的情况下需要主动加载model文件
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                    id = get_checkpoint_id_by_re(models_path)
                    models_path = os.path.dirname(models_path)

        try:

            # 评估模式下需要对model文件进行数字签名验证
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
                if CONFIG.digital_signature_verification:
                    if not model_file_signature_verify(models_path):
                        self.logger.error(
                            f"predict run mode is {CONFIG.run_mode} model_file_signature_verify from {models_path} failed, "
                            f"so exit",
                            g_not_server_label,
                        )
                        self.stop_process_when_eval_error()
                        return False

                    else:
                        self.logger.info(
                            f"predict run mode is {CONFIG.run_mode} model_file_signature_verify from {models_path} success",
                            g_not_server_label,
                        )

            # 调用业务加载最新模型, 可能会出现错误
            with TimeIt() as ti:
                for policy, agent_wrapper in self.policy_agent_wrapper_maps.items():
                    if agent_wrapper.load_model_by_source(
                        path=models_path, id=id, source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK
                    ):

                        """
                        关键信息打印INFO日志
                        """
                        self.logger.info(
                            f"predict standard_load_last_new_model_by_framework policy {policy} from path {models_path}, "
                            f"id {id} success",
                            g_not_server_label,
                        )

                        self.actor_load_last_model_succ_cnt += 1

                    else:
                        self.logger.error(
                            f"predict standard_load_last_new_model_by_framework policy {policy} from path {models_path}, "
                            f"id {id} failed",
                            g_not_server_label,
                        )
                        return False

            # 所有的模型加载完成后才返回结果
            return True

        except Exception as e:
            self.logger.exception(
                f"predict standard_load_last_new_model_by_framework from {models_path}, id {id} failed, "
                f"error is {str(e)}",
                g_not_server_label,
            )

            # 如果是eval模式, 加载失败就停止actor预测进程, 其他模式会周期性的加载model文件, 不做报错退出
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
                self.logger.error(
                    f"predict run mode is {CONFIG.run_mode} standard_load_last_new_model_by_framework from {models_path}, "
                    f"id {id} failed, so exit",
                    g_not_server_label,
                )
                self.stop_process_when_eval_error()

            return False

    def stop_process_when_eval_error(self):
        self.process_pid_list.append(os.getpid())
        stop_process_by_pid(self.process_pid_list)
