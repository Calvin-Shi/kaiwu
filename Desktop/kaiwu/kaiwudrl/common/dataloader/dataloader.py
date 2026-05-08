#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import multiprocessing
import datetime
import os
import time
import json
import schedule
from common_python.config.config_control import CONFIG
from common_python.logging.kaiwu_logger import g_not_server_label
from kaiwudrl.common.utils.common_func import (
    is_list_eq,
    set_schedule_event,
    python_exec_shell,
)
from common_python.monitor.prometheus_utils import PrometheusUtils
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from common_python.worker.worker import Worker, WorkerConfig


class DataLoader(Worker):
    """
    该类主要是负责数据导入, 业务来定义数据格式
    """

    def __init__(self) -> None:

        # 进程pid
        self.current_pid = os.getpid()
        worker_config = WorkerConfig(
            worker_name="dataloader",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

    def before_run(self):
        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        # 访问普罗米修斯的类
        self.prometheus_utils = PrometheusUtils(self.logger)

        self.process_run_count = 0

        # 在before run最后打印启动成功日志
        self.logger.info(f"dataloader start success at pid {self.current_pid}", g_not_server_label)

        return True

    def after_run(self) -> bool:
        pass

    def run_once(self):

        # 启动定时器
        schedule.run_pending()

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("dataloader DataLoader stop success", g_not_server_label)

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"dataloader before_run failed, so return", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(
                    f"dataloader run error: {str(e)}",
                    g_not_server_label,
                )
