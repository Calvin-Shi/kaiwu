#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import json
import multiprocessing
import datetime
import os
import schedule
import time
import copy
import collections
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.common_func import (
    is_list_eq,
    list_diff,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
    Context,
    python_exec_shell,
    register_sigterm_handler,
    is_pid_alive,
)
from kaiwudrl.common.utils.slots import Slots
from kaiwudrl.common.config.app_conf import AppConf
from common_python.worker.worker import Worker, WorkerConfig


class AiSrvHandle(Worker):
    __slots__ = (
        "logger",
        "conn",
        "simu_ctx",
        "slots",
        "slot_id",
        "msg_buff",
        "data_queue",
        "kaiwu_rl_helper",
        "min_slot_id",
        "monitor_proxy",
    )

    def __init__(self, kaiwu_env_address, simu_ctx) -> None:
        # 进程pid
        self.current_pid = os.getpid()
        worker_config = WorkerConfig(
            worker_name="ai_server_handle",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        self.kaiwu_env_address = kaiwu_env_address
        self.simu_ctx = simu_ctx

        # 建立zmq通信
        """
        self.client_id = get_uuid()
        parts = self.kaiwu_env_address.split(":")
        ip = parts[0]
        port = parts[1]
        self.zmq_client = ZmqClient(str(self.client_id), ip, port)
        self.zmq_client.connect()
        """

        self.simu_ctx.client_address = self.kaiwu_env_address

        self.simu_ctx.exit_flag = multiprocessing.Value("b", False)

        # slot_id
        self.slots = self.simu_ctx.slots
        self.slot_id = self.slots.get_slot()
        self.simu_ctx.slot_id = self.slot_id

        # 七彩石的操作句柄由主进程传递
        if int(CONFIG.use_rainbow):
            self.rainbow_wrapper = self.simu_ctx.rainbow_wrapper

        # 负责统计kaiwu_rl_helper中产生的对局数据，线程和主进程间只用数据dequeue即可，减少cpu消耗
        self.data_queue = collections.deque(maxlen=CONFIG.max_queue_len)
        self.simu_ctx.data_queue = self.data_queue

        self.process_run_count = 0

    # aisrv在处理actor和learner的动态扩缩容逻辑
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return

        (
            current_actor_addrs,
            current_learner_addrs,
        ) = self.kaiwu_rl_helper.get_current_actor_learner_address()

        read_from_file_content = CONFIG.read_from_file(CONFIG.svr_name, ["actor_addrs", "learner_addrs"])

        # 本次读取的文件内容错误, 则跳过本次处理下次再进行处理
        try:
            new_actor_addrs = read_from_file_content["actor_addrs"][CONFIG.policy_name]
            new_learner_addrs = read_from_file_content["learner_addrs"][CONFIG.policy_name]
        except Exception as e:
            self.logger.info(f"ai_server_handle load actor address and learner address err, {str(e)}")
            return

        self.aisrv_with_different_actor_learner(
            current_actor_addrs,
            new_actor_addrs,
            current_learner_addrs,
            new_learner_addrs,
        )

    # actor和learner的IP区别判断, 采用2个参数进行返回
    def check_actor_ip_and_learner_ip_change(
        self, actor_address, old_actor_address, learner_address, old_learner_addrs
    ):
        actor_ip_change = False
        learner_ip_change = False

        if not actor_address and not learner_address:
            return actor_ip_change, learner_ip_change

        if actor_address:
            if not is_list_eq(actor_address, old_actor_address):
                actor_ip_change = True

        if learner_address:
            if not is_list_eq(learner_address, old_learner_addrs):
                learner_ip_change = True

        return actor_ip_change, learner_ip_change

    def aisrv_with_different_actor_learner(
        self,
        current_actor_addrs,
        new_actor_addrs,
        current_learner_addrs,
        new_learner_addrs,
    ):
        actor_ip_change, learner_ip_change = self.check_actor_ip_and_learner_ip_change(
            new_actor_addrs,
            current_actor_addrs,
            new_learner_addrs,
            current_learner_addrs,
        )

        if not actor_ip_change and not learner_ip_change:
            return

        # actor地址有变化
        if actor_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(current_actor_addrs, new_actor_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增actor_proxy
                actor_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    KaiwuDRLDefine.PROCESS_ADD, list_A_have_B_not_have, None, None
                )

            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少actor_ip
                actor_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    KaiwuDRLDefine.PROCESS_REDUCE,
                    list_B_have_A_not_have,
                    None,
                    None,
                )

        # learner地址有变化
        if learner_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(new_learner_addrs, current_learner_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增learner_proxy
                learner_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    None, None, KaiwuDRLDefine.PROCESS_ADD, list_A_have_B_not_have
                )

            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少learner_ip
                learner_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    None,
                    None,
                    KaiwuDRLDefine.PROCESS_REDUCE,
                    list_B_have_A_not_have,
                )

        # 修改配置文件内容落地
        if actor_add_result and actor_reduce_result and learner_add_result and learner_reduce_result:
            self.logger.info("ai_server_handle aisrv_with_different_actor_learner change finish sucess")

    def before_run(self):
        # 支持每局结束前, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            # 在本次对局开始前, aisrv看下参数修改情况
            self.rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
            self.rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        """
        aisrv下每1个客户端启动1个KaiWuRLHelper对象, 封装了强化学习流程
        1. 如果是主循环的内容在业务侧, 调用self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper
        2. 如果是主循环的内容在框架侧, 调用self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)
        """
        self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper(self.simu_ctx)

        # self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)

        self.kaiwu_rl_helper.daemon = True
        self.logger.info(f"ai_server_handle use kaiwu_rl_helper: {self.kaiwu_rl_helper}")

        self.min_slot_id, _ = self.slots.get_min_max_slot_id()
        self.logger.info(
            f"ai_server_handle established connect to {self.kaiwu_env_address}, "
            f"slot id is {self.slot_id}, min_slot_id is {self.min_slot_id}"
        )

        (
            current_actor_addrs,
            current_learner_addrs,
        ) = self.kaiwu_rl_helper.get_current_actor_learner_address()
        self.logger.info(
            f"ai_server_handle current_actor_addrs is {current_actor_addrs}, "
            f"current_learner_addrs is {current_learner_addrs}"
        )

        # 启动独立的进程, 负责aisrv与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            # 传递给kaiwu_rl_helper
            self.kaiwu_rl_helper.set_monitor_proxy(self.monitor_proxy)

        """
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        """
        if CONFIG.actor_learner_expansion:
            set_schedule_event(
                int(CONFIG.alloc_process_per_seconds),
                self.aisrv_with_new_actor_learner_change,
            )

        # 单局开始时
        self.episode_start()

        # 在before run最后打印启动成功日志
        self.logger.info(f"ai_server_handle start success at pid {self.current_pid}")

        # 开启kaiwu_rl_helper线程, 因为在该线程的主循环和其他的进程开始交互, 故放在before_run最后开始
        self.kaiwu_rl_helper.start()

        # 注册kaiwu_rl_helper线程到进程健康监控器
        if hasattr(self.simu_ctx, "process_health_monitor") and self.simu_ctx.process_health_monitor:
            self.simu_ctx.process_health_monitor.register_thread(
                thread=self.kaiwu_rl_helper,
                name=f"kaiwu_rl_helper_{self.slot_id}",
                thread_type="kaiwu_rl_helper",
            )

        # 注册SIGTERM信号处理
        register_sigterm_handler(self.handle_sigterm, CONFIG.sigterm_pids_file)

        # 设置kaiwu_rl_helper的周期性的监控上报, 因为kaiwu_rl_helper是调用了workflow的, 故提到该处解决
        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.train_predict_stat)

        return True

    def train_predict_stat(self):
        self.kaiwu_rl_helper.train_predict_stat()

    def run_once(self) -> None:
        # 步骤1, 例行任务
        schedule.run_pending()

    def after_run(self) -> bool:
        pass

    def run(self) -> None:
        # before_run
        if not self.before_run():
            self.logger.error(f"ai_server_handle before_run failed, so return")
            return

        # 主循环
        try:
            while not self.simu_ctx.exit_flag.value:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    # 因为该run函数内只有周期性的操作和on-policy的操作, 故增加sleep时间减少CPU占用
                    time.sleep(CONFIG.idle_sleep_second * 1000)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

        except Exception as e:
            self.logger.exception(f"ai_server_handle failed to handle message {str(e)}")
            self.simu_ctx.exit_flag.value = True

            self.episode_stop()

            raise e

    # 单次对局结束时处理
    def episode_stop(self):
        # 安全退出KaiWuRLHelper
        self.kaiwu_rl_helper.stop()

        # 回收slot_id
        self.slots.put_slot(self.slot_id)

        self.logger.info("ai_server_handle lost connection from {}", str(self.kaiwu_env_address))

    # 单次对局开始时处理
    def episode_start(self):
        pass

    def handle_sigterm(self, sig, frame):
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            self.logger.info(f"ai_server_handle {self.current_pid} is starting to handle the SIGTERM signal.")
            self.kaiwu_rl_helper.handle_sigterm(sig, frame)
        else:
            self.logger.info(
                f"ai_server_handle not KaiwuDRLDefine.WRAPPER_LOCAL, so {self.current_pid} not handle the SIGTERM signal."
            )
