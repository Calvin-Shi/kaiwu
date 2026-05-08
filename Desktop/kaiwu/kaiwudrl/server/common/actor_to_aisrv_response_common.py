#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import dill
import numpy as np
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.common_func import TimeIt

from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.common_func import compress_data


class ActorToAisrvResponseCommon(object):
    """
    该类主要是actor朝aisrv发送回包时使用的, 因为存在有actor, actor_proxy_local的2个进程使用, 故将代码单独提出公共的, 只是维护一份即可
    """

    def __init__(self, logger) -> None:

        # 下面是因为需要在使用时用到的变量, 故该类里只是定义, 由调用者进行赋值

        # zmq_ops模式下使用
        self.zmq_server = None
        # zmq模式下使用
        self.send_server = None

        # 下面统计放在该类里计算, 外界需要时返回统计值即可
        self.send_to_aisrv_succ_cnt = 0
        self.recv_from_actor_succ_cnt = 0
        self.max_compress_time = 0
        self.max_compress_size = 0

        self.slots = None
        self.slot_group_name = None

        self.logger = logger

        self.zmq_server_ip = None

        # 按照需要加载dill
        import dill

    def set_zmq_server_ip(self, zmq_server_ip):
        self.zmq_server_ip = zmq_server_ip

    def set_zmq_send_server(self, zmq_send_server):
        self.send_server = zmq_send_server

    def set_zmq_server(self, zmq_server):
        self.zmq_server = zmq_server

    def set_slots(self, slots):
        self.slots = slots

    def set_slot_group_name(self, slot_group_name):
        self.slot_group_name = slot_group_name

    def set_recv_from_actor_succ_cnt(self, recv_from_actor_succ_cnt):
        self.recv_from_actor_succ_cnt = recv_from_actor_succ_cnt

    def get_recv_from_actor_succ_cnt(self):
        return self.recv_from_actor_succ_cnt

    def get_send_to_aisrv_succ_cnt(self):
        return self.send_to_aisrv_succ_cnt

    def get_max_compress_time(self):
        return self.max_compress_time

    def set_max_compress_time(self, max_compress_time):
        self.max_compress_time = max_compress_time

    def get_max_compress_size(self):
        return self.max_compress_size

    def set_max_compress_size(self, max_compress_size):
        self.max_compress_size = max_compress_size

    def send_response_to_aisrv_by_actor(self, size, pred):
        """
        aisrv(actor)给battlesrv回包, 是actor进程朝battlesrv回包, 非标准化下的zmq_ops模式下调用
        """
        if CONFIG.distributed_tracing:
            self.logger.info(
                f"actor_server distributed_tracing send_response_to_aisrv_by_actor start",
                g_not_server_label,
            )

        client_ids = pred.pop(KaiwuDRLDefine.CLIENT_ID_TENSOR)
        compose_ids = pred.pop(KaiwuDRLDefine.COMPOSE_ID_TENSOR)

        with TimeIt() as ti:
            dict_obj = {}
            for i in range(size):
                send_data = {k: v[i] for k, v in pred.items()}
                client_id = client_ids[i] if not CONFIG.use_rnn else client_ids[i][0]
                compose_id = compose_ids[i] if not CONFIG.use_rnn else compose_ids[i][0]
                list_obj = dict_obj.setdefault(client_id, [])
                list_obj.append((list(compose_id), send_data))

            for client_id, send_data in dict_obj.items():
                try:
                    self.zmq_server.recv(block=False, binary=True)
                except Exception as e:
                    pass

                self.zmq_server.send(str(client_id), send_data, binary=True)
                self.send_to_aisrv_succ_cnt += 1

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        f"actor_server distributed_tracing zmq server send a new msg to {compose_id} success",
                        g_not_server_label,
                    )

        if CONFIG.distributed_tracing:
            self.logger.info(
                f"actor_server distributed_tracing send_response_to_aisrv_by_actor end",
                g_not_server_label,
            )

    def send_response_to_aisrv_simple_fast_by_actor_common(self, size, preds):
        """
        aisrv(actor)给battlesrv回包, 是actor进程朝battlesrv回包, 抽离出来的代码, 其中区别是:
        1. 非标准化, pred是分format_action, network_sample_info, lstm_info等字段
        2. 标准化, pred是个对象, 形如values is [<kaiwu_agent.utils.common_func.ActData object at 0x7f9e8f46e510>]这种类型的数据, 需要序列化
        """
        if CONFIG.distributed_tracing:
            self.logger.info(
                "actor_server distributed_tracing send_response_to_aisrv_simple_fast_by_actor_common start",
                g_not_server_label,
            )

        dict_obj = {}

        for j, pred in enumerate(preds):
            client_ids = pred[KaiwuDRLDefine.CLIENT_ID_TENSOR]
            compose_ids = pred[KaiwuDRLDefine.COMPOSE_ID_TENSOR]

            # 需要处理回包时才进行处理
            send_data = {"pred": dill.dumps(pred["pred"]), "need_response": pred.get("need_response")}

            client_id = client_ids[0] if not CONFIG.use_rnn else client_ids[0][0]
            compose_id = compose_ids[0] if not CONFIG.use_rnn else compose_ids[0][0]
            list_obj = dict_obj.setdefault(client_id, [])
            list_obj.append((list(compose_id), send_data))

        for client_id, send_data in dict_obj.items():

            # 压缩耗时和压缩包大小
            with TimeIt() as ti:
                compressed_data = compress_data(send_data)

            if self.max_compress_time < ti.interval:
                self.max_compress_time = ti.interval

            compress_msg_len = len(compressed_data)
            if self.max_compress_size < compress_msg_len:
                self.max_compress_size = compress_msg_len

            # 这里直接放置的是client_id, compressed_data对
            self.send_server.put_predict_result_data([client_id, compressed_data])

            if CONFIG.server_use_processes == KaiwuDRLDefine.RUN_AS_THREAD:
                with self.send_server.predict_result_condition:
                    self.send_server.predict_result_condition.notify()

            if CONFIG.distributed_tracing:
                self.logger.info(
                    f"actor_server distributed_tracing zmq server send a new msg to {client_id} success",
                    g_not_server_label,
                )

        if CONFIG.distributed_tracing:
            self.logger.info("actor_server distributed_tracing send_response_to_aisrv_simple_fast_by_actor_common end")

    def standard_send_response_to_aisrv_simple_fast_by_actor(self, size, preds):
        """
        aisrv(actor)给battlesrv回包, 是actor进程朝battlesrv回包
        """
        return self.send_response_to_aisrv_simple_fast_by_actor_common(size, preds)

    def send_response_to_aisrv_simple_fast_by_aisrv_common(self, size, preds):
        """
        aisrv(actor)给battlesrv回包, 是actor_proxy_local进程朝battlesrv回包, 抽离出来的代码, 其中区别是:
        1. 非标准化, pred是分format_action, network_sample_info, lstm_info等字段
        2. 标准化, pred是个对象, 形如values is [<kaiwu_agent.utils.common_func.ActData object at 0x7f9e8f46e510>]这种类型的数据, 需要序列化
        """

        if CONFIG.distributed_tracing:
            self.logger.info(
                "actor_proxy_local distributed_tracing send_response_to_aisrv_simple_fast_by_aisrv_common start",
                g_not_server_label,
            )

        dict_obj = {}
        compose_id_indexs = []

        for j, pred in enumerate(preds):
            client_ids = pred[KaiwuDRLDefine.CLIENT_ID_TENSOR]
            compose_ids = pred[KaiwuDRLDefine.COMPOSE_ID_TENSOR]

            # 需要处理回包时才进行处理
            if pred.get("need_response"):
                send_data = {"pred": dill.dumps(pred["pred"])}

                client_id = client_ids[0] if not CONFIG.use_rnn else client_ids[0][0]
                compose_id = compose_ids[0] if not CONFIG.use_rnn else compose_ids[0][0]
                list_obj = dict_obj.setdefault(client_id, [])
                list_obj.append((list(compose_id), send_data))

        """
        此时预测请求是在actor_proxy_local进程里处理的, 不需要走网络传输, 故原则上是不需要client_id的, 但是为了代码兼容, 加上client_id
        此时不需要进行压缩和解压缩操作
        """
        for client_id, send_data in dict_obj.items():

            # 处理响应回包
            result_map = {}
            for compose_id, pred_result in send_data:
                slot_id, agent_id, message_id, model_version, agent_main_id = compose_id

                # 增加model_version值
                pred_result["model_version"] = model_version
                result_map.setdefault(slot_id, {})[agent_id] = pred_result

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        f"actor_proxy_local distributed_tracing compose_id {compose_id} "
                        f"from zmq server {self.zmq_server_ip} success",
                        g_not_server_label,
                    )

                # 超时处理
                index = (slot_id, agent_id, message_id)
                compose_id_indexs.append(index)

            for slot_id, client_results in result_map.items():
                output_pipe = self.slots.get_output_pipe(self.slot_group_name, slot_id)
                output_pipe.send(client_results)
                self.recv_from_actor_succ_cnt += 1
                # self.logger.debug(f'actor_proxy_local slot_id {slot_id} output_pipe send success', g_not_server_label)

            if CONFIG.distributed_tracing:
                self.logger.info(
                    f"actor_proxy_local distributed_tracing zmq server send a new msg to {client_id} success",
                    g_not_server_label,
                )

        if CONFIG.distributed_tracing:
            self.logger.info(
                "actor_proxy_local distributed_tracing send_response_to_aisrv_simple_fast_by_aisrv_common end",
                g_not_server_label,
            )

        """
        因为timeout_map和time_cost_map都在actor和aisrv进程内, 故这里简单的处理成返回这些值, 让其处理即可
        """
        return compose_id_indexs

    def standard_send_response_to_aisrv_simple_fast_by_aisrv(self, size, preds):
        """
        aisrv(actor)给battlesrv回包, 是actor_proxy_local进程朝battlesrv回包, 标准化场景使用
        """

        return self.send_response_to_aisrv_simple_fast_by_aisrv_common(size, preds)
