#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import numpy as np
from typing import Optional, List, Any
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
import random
from collections import defaultdict
from kaiwudrl.common.utils.common_func import TimeIt

from common_python.logging.kaiwu_logger import g_not_server_label

from kaiwudrl.common.checkpoint.model_file_common import (
    get_checkpoint_id_by_re,
)

import dill
import msgpack
from msgpack import ExtType
from multiprocessing import Lock
from kaiwudrl.server.common.multi_model_common import MultiModelManager


class PredictCommon(object):
    """
    该类主要是预测的公共类, 因为存在有actor, actor_proxy_local的2个进程使用, 故将代码单独提出公共的, 只是维护一份即可
    """

    def __init__(self, policy_name, monitor_proxy, logger) -> None:

        # 下面是因为需要在使用时用到的变量, 故该类里只是定义, 由调用者进行赋值

        # policy和agent_wrapper对象的map, 为了支持多agent
        self.policy_agent_wrapper_maps = None
        self.policy_conf = None
        self.current_sync_model_version_from_learner = 0
        self.model_file_sync_wrapper = None
        self.last_load_model_aisrv_ip = None
        self.last_load_model_time = 0
        self.logger = logger
        self.monitor_proxy = monitor_proxy
        self.policy_name = policy_name

        # 统计使用
        self.actor_batch_predict_cost_time_ms = 0
        self.actor_load_last_model_succ_cnt = 0
        self.actor_load_last_model_error_cnt = 0
        self.actor_load_last_model_cost_ms = 0

        # 防止多个预测进程调用出现时序问题, 即A进程先调用清空了队列, B进程开始调用判断队列为空即返回的情况
        self.lock = Lock()

        # 预先需要在训练中评估加载的模型的对象, 即对手模型
        self.multi_model_manager = MultiModelManager(self.policy_name, self.monitor_proxy, self.logger)

    def is_multi_model_manager_init_success(self):
        """
        检查 MultiModelManager 是否初始化成功
        """
        return self.multi_model_manager.is_init_success()

    def set_actor_load_last_model_cost_ms(self, actor_load_last_model_cost_ms):
        self.actor_load_last_model_cost_ms = actor_load_last_model_cost_ms

    def get_actor_load_last_model_cost_ms(self):
        return self.actor_load_last_model_cost_ms

    def get_actor_load_last_model_succ_cnt(self):
        return self.actor_load_last_model_succ_cnt

    def get_actor_load_last_model_error_cnt(self):
        return self.actor_load_last_model_error_cnt

    def get_actor_batch_predict_cost_time_ms(self):
        return self.actor_batch_predict_cost_time_ms

    def set_actor_batch_predict_cost_time_ms(self, actor_batch_predict_cost_time_ms):
        self.actor_batch_predict_cost_time_ms = actor_batch_predict_cost_time_ms

    def set_policy_agent_wrapper_maps(self, policy_agent_wrapper_maps):
        self.policy_agent_wrapper_maps = policy_agent_wrapper_maps

    def set_model_file_sync_wrapper(self, model_file_sync_wrapper):
        self.model_file_sync_wrapper = model_file_sync_wrapper

    def set_policy_conf(self, policy_conf):
        self.policy_conf = policy_conf

    def set_current_sync_model_version_from_learner(self, current_sync_model_version_from_learner):
        self.current_sync_model_version_from_learner = current_sync_model_version_from_learner

    def predict_stat(self):
        """
        分为下面部分:
        1. 正常的运行预测/利用
        2. 模型池里预测/利用
        """
        predict_count = 0
        if self.policy_agent_wrapper_maps is not None:
            # 针对有多个policy的预测次数, 则结果是直接加起来, 因为分开启普罗米修斯和不开启普罗米修斯的场景, 这里将重要的指标都打印下
            for policy, agent_wrapper in self.policy_agent_wrapper_maps.items():
                predict_count += agent_wrapper.predict_stat

        predict_count += self.multi_model_manager.get_predict_stat()

        return predict_count

    def predict_detail(self, policy, model_policy, predict_data, batch_size, msg_type=None):
        """
        根据不同的model_policy找到不同的Agent进行预测
        """
        if model_policy in [KaiwuDRLDefine.ID_RANDOM, KaiwuDRLDefine.ID_LATEST]:
            agent_wrapper = self.policy_agent_wrapper_maps.get(
                policy, next(iter(self.policy_agent_wrapper_maps.values()))
            )

            return self.predict_detail_by_current_model(agent_wrapper, predict_data, batch_size, msg_type)
        else:
            return self.predict_detail_by_multi_model(model_policy, predict_data, msg_type)

    def predict_detail_batch_common(self, policy_map):
        """
        无论是standard或者normal, 批处理调用的公共函数
        """
        if not policy_map:
            return []

        # 收集所有预测结果
        all_preds = []
        for key, state_dicts in policy_map.items():
            policy, model_policy, message_type = key
            preds = self.predict_detail(policy, model_policy, state_dicts, len(state_dicts), message_type)
            if preds:
                all_preds.append(preds)
            else:
                all_preds.append([None] * len(state_dicts))

        return all_preds

    def standard_predict_detail_batch(self, policy_map):
        """
        标准化版本批处理方式
        """
        return self.predict_detail_batch_common(policy_map)

    # 支持的预测消息类型白名单，值与方法名一致，可直接用 getattr 分发
    PREDICT_MSG_TYPES = frozenset(
        {
            KaiwuDRLDefine.MESSAGE_PREDICT,
            KaiwuDRLDefine.MESSAGE_EXPLOIT,
            KaiwuDRLDefine.MESSAGE_RESET,
            KaiwuDRLDefine.MESSAGE_INIT_CONFIG,
        }
    )

    def predict_detail_by_multi_model(self, model_policy, predict_data, msg_type):
        """
        从预测里先放置的模型池里读取数据，通过 msg_type 动态分发到对应方法
        """
        handler = getattr(self.multi_model_manager, msg_type, None) if msg_type in self.PREDICT_MSG_TYPES else None
        return handler(model_policy, predict_data) if handler else None

    def predict_detail_by_current_model(self, agent_wrapper, predict_data, batch_size, msg_type):
        """
        通过 msg_type 动态分发: predict / exploit / reset / init_config
        """
        handler = getattr(agent_wrapper, msg_type, None) if msg_type in self.PREDICT_MSG_TYPES else None
        return handler(predict_data) if handler else None

    # actor上的预测predict主函数, 使用框架的predict
    def predict_tensorflow(self):
        size = 0
        pred = None

        batch_size = 1

        try:
            extra_tensors = {
                KaiwuDRLDefine.CLIENT_ID_TENSOR: self.dequeue_tensors[-2],
                KaiwuDRLDefine.COMPOSE_ID_TENSOR: self.dequeue_tensors[-1],
            }

            # 获取policy对应的agent_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个agent_wrapper
            policy = CONFIG.policy_name
            agent_wrapper = self.policy_agent_wrapper_maps.get(
                policy, next(iter(self.policy_agent_wrapper_maps.values()))
            )

            pred = agent_wrapper.predict(extra_tensors)
            size = next(iter(pred.values())).shape[0]

            pred["s"] = np.array([self.global_step] * size)

        except Exception as e:
            self.logger.exception(f"predict failed to run predictor, as {e}")

        # self.logger.debug(f'predict after predict, size is {size}, pred is {pred}')

        return size, pred

    # actor上的预测predict主函数, 使用TensorRT
    def predict_tensorrt(self, datas):
        batch_size = len(datas)

        # 数据整理
        state_dict = {}
        state_space = self.policy_conf[CONFIG.policy_name].state.state_space()
        for i, key in enumerate(state_space.keys()):
            state_dict[key] = [datas[j][i].flatten() for j in range(batch_size)]

        res_msgs = []
        for i in range(batch_size):
            # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                datas[i][-1][0][3] = self.current_sync_model_version_from_learner

            res_msgs.append(
                {
                    KaiwuDRLDefine.CLIENT_ID_TENSOR: datas[i][-2],
                    KaiwuDRLDefine.COMPOSE_ID_TENSOR: datas[i][-1],
                }
            )

        sizes = []
        try:
            # 获取policy对应的agent_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个agent_wrapper
            policy = CONFIG.policy_name
            model_policy = KaiwuDRLDefine.ID_LATEST
            pred = self.predict_detail(policy, model_policy, state_dict, batch_size)
            if pred:
                for i in range(batch_size):
                    res_msgs[i]["pred"] = [p[i] for p in pred]
                    res_msgs[i]["need_response"] = True
                    sizes.append(len(pred))

        except Exception as e:
            self.logger.exception(f"predict failed to run predictor, as {e}")

        # self.logger.debug(f'predict after predict, size is {sizes}')

        return sizes, res_msgs

    def standard_load_model_choose_one(self, policy, id):
        """
        直接调用model_file_sync的get_current_available_model_file函数
        """
        return self.model_file_sync_wrapper.ckpt_sync_warper.get_current_available_model_file(id, self.logger)

    def standard_load_model_detail(self, policy, id):
        """
        加载model文件的操作如下:
        1. 根据policy找到对应的agent_wrapper
        2. 根据id来处理
        2.1 如果是latest, random, 则在FIFO队列里加载
        2.2 如果是具体的数字则去multi_model_manager管理的model文件里加载
        """
        if not policy or not id:
            self.logger.error(f"predict param is error, policy {policy}, id {id}")
            return False

        # 如果是on_policy就直接返回True, on_policy是由框架自己驱动模型文件加载的, 用户不需要
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            return True

        # 如果还没有从modelpool拉取下来文件则返回
        if not self.model_file_sync_wrapper.ckpt_sync_warper.get_had_pull_the_first_model_file_success_value():
            self.logger.info(f"predict had not get pull model file, so return")
            return False

        if id in [KaiwuDRLDefine.ID_RANDOM, KaiwuDRLDefine.ID_LATEST]:
            agent_wrapper = self.policy_agent_wrapper_maps.get(policy, None)
            if not agent_wrapper:
                self.logger.error(f"predict agent_wrapper is None, policy is {policy}")
                return False

            # 在锁的保护下去访问standard_load_model_choose_one函数
            with self.lock:
                id, path = self.standard_load_model_choose_one(policy, id)
                if path is None:
                    self.logger.info(f"predict load_model_dir is empty, policy is {policy}, id is {id}")
                    return False

            with TimeIt() as ti:
                success = agent_wrapper.load_model_by_source(
                    path=path, id=id, source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER
                )

        else:
            # 加载模型池子里的model文件
            with TimeIt() as ti:
                success = self.multi_model_manager.load_model(id)
            path = self.multi_model_manager.get_model_path(id)

        if success:
            self.logger.info(f"predict standard_load_model success, policy {policy}, path {path}, id {id}")
            self.actor_load_last_model_succ_cnt += 1
        else:
            self.logger.error(f"predict standard_load_model failed, policy {policy}, path {path}, id {id}")
            self.actor_load_last_model_error_cnt += 1

        if self.actor_load_last_model_cost_ms < ti.interval * 1000:
            self.actor_load_last_model_cost_ms = ti.interval * 1000

        return success

    def standard_load_model(
        self,
        policy,
        id,
        ip,
    ):
        """
        1. 针对不同的aisrv来让同一个actor执行load_model的操作
        1.1 在时间间隔内如果获取到第一个即执行
        1.2 在最大时间间隔内如果1.1的aisrv持续的进行load_model时则直接执行, 否则转1.3
        1.3 重新接收第一个需要执行的aisrv, 然后更新时间
        """
        if not policy or not id or not ip:
            self.logger.error(f"predict param is error, policy {policy}, id {id}, ip {ip}")
            return

        now = time.time()

        # 如果没有上一次的加载模型请求，或者上一次的请求来自同一个ip，执行加载模型
        if not self.last_load_model_aisrv_ip or self.last_load_model_aisrv_ip == ip:
            if self.standard_load_model_detail(policy, id):
                self.last_load_model_aisrv_ip = ip
                self.last_load_model_time = now
        else:
            # 如果上一次的请求来自不同的ip，但已经超过了最大等待时间，也执行加载模型
            if now - self.last_load_model_time >= CONFIG.choose_aisrv_to_load_model_or_save_model_max_time_seconds:
                if self.standard_load_model_detail(policy, id):
                    self.last_load_model_aisrv_ip = ip
                    self.last_load_model_time = now

            # 否则，不执行加载模型，即忽略来自不同ip的请求，如果它们在最大等待时间内

    def standard_predict_simple_batch(self, datas):
        """
        actor上的预测predict主函数, 使用业务的predict, 批处理方式
        """

        # 数据整理, 回包时使用
        client_ids = []
        compose_ids = []

        # key为(policy, model_policy, message_type), value为具体的stat_dict
        policy_map = defaultdict(list)

        # 返回的数据结果
        res_msgs = []
        sizes = []

        try:
            for data in datas:
                """
                根据来进行不同的操作:
                1. 数据流, 进行预测
                2. 管理流, 加载model文件
                """
                raw_data = data.get("data")
                client_id = data.get("client_id")
                compose_id = data.get("compose_id")

                message_type = raw_data.get(KaiwuDRLDefine.MESSAGE_TYPE)
                message_value = raw_data.get(KaiwuDRLDefine.MESSAGE_VALUE)

                if message_type in [
                    KaiwuDRLDefine.MESSAGE_PREDICT,
                    KaiwuDRLDefine.MESSAGE_EXPLOIT,
                    KaiwuDRLDefine.MESSAGE_RESET,
                    KaiwuDRLDefine.MESSAGE_INIT_CONFIG,
                ]:
                    model_policy = message_value.get("model_policy")
                    predict_data = message_value.get("predict_data")
                    serialize_type = message_value.get("serialize_type")
                    state_dict = self.smart_deserialize(serialize_type, predict_data)

                    # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
                    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                        compose_id[0][3] = self.current_sync_model_version_from_learner

                    # 获取policy对应的agent_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个agent_wrapper
                    policy = compose_id[0][4]
                    client_ids.append(client_id)
                    compose_ids.append(compose_id)

                    # 注意传递的state_dict是列表
                    key = (policy, model_policy, message_type)
                    policy_map[key].append(state_dict[0])

                elif message_type == KaiwuDRLDefine.MESSAGE_LOAD_MODEL:
                    self.standard_load_model(
                        message_value.get("policy"),
                        message_value.get("id"),
                        message_value.get("ip"),
                    )

                    # load_model和predict, exploit保持一致都进行回包处理
                    result = {
                        "pred": True,
                        KaiwuDRLDefine.CLIENT_ID_TENSOR: client_id,
                        KaiwuDRLDefine.COMPOSE_ID_TENSOR: compose_id,
                        "need_response": False,
                    }
                    res_msgs.append(result)
                    sizes.append(len(result))

                else:
                    self.logger.error(
                        f"predict un support message_type is {message_type}",
                        g_not_server_label,
                    )

            # 批量预测
            if policy_map:
                preds = self.standard_predict_detail_batch(policy_map)
                for i, pred in enumerate(preds):
                    if pred:
                        result = {
                            "pred": pred,
                            KaiwuDRLDefine.CLIENT_ID_TENSOR: client_ids[i],
                            KaiwuDRLDefine.COMPOSE_ID_TENSOR: compose_ids[i],
                            "need_response": True,
                        }
                        # 此时返回的size就是1
                        res_msgs.append(result)
                        sizes.append(1)

        except Exception as e:
            self.logger.exception(
                f"predict failed to run predictor, as {e}",
                g_not_server_label,
            )

        return sizes, res_msgs

    def obsdata_decode(self, code, data):
        """ObsData 专用解码器"""
        if code == KaiwuDRLDefine.OBSDATA_EXT_TYPE:
            attrs = msgpack.unpackb(data, raw=False)
            # 动态重建 ObsData 实例（无需类注册）
            cls = type("ObsData", (), {})  # 创建空类
            instance = cls.__new__(cls)  # 绕过 __init__
            instance.__dict__.update(attrs)
            return instance
        return ExtType(code, data)

    def smart_deserialize(self, serialize_type: str, predict_data: bytes) -> Any:
        if serialize_type == KaiwuDRLDefine.SERIALIZE_TYPE_MSGPACK:
            state_dict = msgpack.unpackb(predict_data, raw=False)
        elif serialize_type == KaiwuDRLDefine.SERIALIZE_TYPE_MSGPACK_EXTEND:
            state_dict = msgpack.unpackb(predict_data, raw=False, ext_hook=self.obsdata_decode)
        else:
            state_dict = dill.loads(predict_data)

        return state_dict

    def standard_predict_simple_single(self, datas):
        """
        actor上的预测predict主函数, 使用业务的predict, 单个处理方式
        """

        # 组装batch, 目前由于是采用多policy和agent的, 故每个包就去预测的, 后期优化成批处理再看下这里
        batch_size = 1

        # 数据整理
        state_dict = {}

        res_msgs = []
        sizes = []

        try:
            for data in datas:
                """
                根据来进行不同的操作:
                1. 数据流, 进行预测
                2. 管理流, 加载model文件
                """
                raw_data = data.get("data")
                client_id = data.get("client_id")
                compose_id = data.get("compose_id")

                message_type = raw_data.get(KaiwuDRLDefine.MESSAGE_TYPE)
                message_value = raw_data.get(KaiwuDRLDefine.MESSAGE_VALUE)

                if message_type in self.PREDICT_MSG_TYPES:
                    model_policy = message_value.get("model_policy")
                    predict_data = message_value.get("predict_data")
                    serialize_type = message_value.get("serialize_type")
                    state_dict = self.smart_deserialize(serialize_type, predict_data)

                    # 如果是on-policy则返回actor预测用到的model版本号, data[i][-1]格式形如[[ 0  0 28 1]]
                    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                        compose_id[0][3] = self.current_sync_model_version_from_learner

                    # 获取policy对应的agent_wrapper处理, 其中policy是通过请求里传输的, 找不到则默认第一个agent_wrapper
                    policy = compose_id[0][4]

                    # 这里打开DEBUG日志会导致性能下降, 故默认不打开任何级别的日志, 如果需要跟进问题手工修改注释再运行
                    preds = self.predict_detail(policy, model_policy, state_dict, batch_size, message_type)
                    result = {
                        "pred": preds,
                        KaiwuDRLDefine.CLIENT_ID_TENSOR: client_id,
                        KaiwuDRLDefine.COMPOSE_ID_TENSOR: compose_id,
                        "need_response": True,
                    }
                    res_msgs.append(result)
                    sizes.append(1)

                elif message_type == KaiwuDRLDefine.MESSAGE_LOAD_MODEL:
                    self.standard_load_model(
                        message_value.get("policy"),
                        message_value.get("id"),
                        message_value.get("ip"),
                    )

                    # load_model和predict, exploit保持一致都进行回包处理
                    result = {
                        "pred": True,
                        KaiwuDRLDefine.CLIENT_ID_TENSOR: client_id,
                        KaiwuDRLDefine.COMPOSE_ID_TENSOR: compose_id,
                        "need_response": False,
                    }
                    res_msgs.append(result)
                    sizes.append(len(result))

                else:
                    self.logger.error(
                        f"predict un support message_type is {message_type}",
                        g_not_server_label,
                    )

        except Exception as e:
            self.logger.exception(
                f"predict failed to run predictor, as {e}",
                g_not_server_label,
            )

        return sizes, res_msgs

    def standard_predict_simple(self, datas):
        """
        根据datas大小来决定如何处理:
        1. 如果大小为1, 调用standard_predict_simple_single
        2. 如果大小大于1, 调用standard_predict_simple_batch
        """
        if len(datas) == 1:
            return self.standard_predict_simple_single(datas)

        return self.standard_predict_simple_batch(datas)

    def predict(self, datas):
        """
        预测函数
        """

        # 返回的数据格式
        size = 0
        pred = None

        if not datas:
            return size, pred

        if CONFIG.distributed_tracing:
            self.logger.info(
                f"predict distributed_tracing predict start",
                g_not_server_label,
            )

        if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.standard_predict_simple(datas)

            # tensorflow的运行机制, TensorFlow 首先会构建计算图（Computation Graph)，
            # 这是一个表示计算操作和数据流的图结构。构建计算图需要一些额外的时间，因此第一次执行 session.run() 时会比较耗时。
            # 作为统计, 因为该值是动态变化的, 故第一次可能比较高, 不影响后续统计
            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_tensorflow()

            # tensorflow的运行机制, TensorFlow 首先会构建计算图（Computation Graph)，
            # 这是一个表示计算操作和数据流的图结构。构建计算图需要一些额外的时间，因此第一次执行 session.run() 时会比较耗时。
            # 作为统计, 因为该值是动态变化的, 故第一次可能比较高, 不影响后续统计
            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.standard_predict_simple(datas)

            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TCNN == CONFIG.use_which_deep_learning_framework:
            pass

        elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT == CONFIG.use_which_deep_learning_framework:
            with TimeIt() as ti:
                size, pred = self.predict_tensorrt(datas)

            if self.actor_batch_predict_cost_time_ms < ti.interval * 1000:
                self.actor_batch_predict_cost_time_ms = ti.interval * 1000

        else:
            self.logger.error(
                f"predict error use_which_deep_learning_framework "
                f"{CONFIG.use_which_deep_learning_framework}, "
                f"only support {KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TCNN}, {KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH}, "
                f"{KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX}, {KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE}",
                g_not_server_label,
            )

            return size, pred

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing predict end", g_not_server_label)

        # 处理actor --> aisrv的回包
        if CONFIG.distributed_tracing:
            self.logger.info(
                f"predict distributed_tracing predict put actor_server predict result start",
                g_not_server_label,
            )

        if CONFIG.distributed_tracing:
            self.logger.info(
                f"predict distributed_tracing predict put actor_server predict result end",
                g_not_server_label,
            )

        return size, pred
