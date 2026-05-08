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
import traceback
import schedule
import copy
import json
from common_python.config.config_control import CONFIG
from common_python.alloc.alloc_utils import AllocUtils
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from common_python.utils.common_func import (
    is_list_eq,
    set_schedule_event,
    python_exec_shell,
)
from common_python.monitor.prometheus_utils import PrometheusUtils
from common_python.utils.common_define import CommonDefine


"""
该类主要是KaiwuDRL上的aisrv、actor、learner与alloc交互的进程, 独立出进程, 减少核心路径消耗
1. aisrv, 服务发现, IP分配
2. actor, 服务发现
3. learner, 服务发现
"""


class AllocProxy(multiprocessing.Process):
    def __init__(self, file_path=None, section=None) -> None:
        super(AllocProxy, self).__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

        self.config_file_path = file_path
        self.config_section = section

    def before_run(self):
        # 在 spawn 模式下,子进程需要重新解析配置 (fork 模式下配置已在内存中,重新解析也不影响)
        # In spawn mode, child process needs to reload config (no effect on fork mode as config is already in memory)
        if self.config_section and self.config_file_path:
            CONFIG.parse_configure([self.config_section], self.config_file_path)
        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.set_logger_format(
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/alloc_proxy_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "alloc_proxy",
        )
        self.logger.info(
            f"alloc_proxy start at pid {pid}, Due to the large amount of logs, the log is printed only when the registration is wrong. ",
            g_not_server_label,
        )

        # alloc 工具类, 与alloc交互操作
        self.alloc_util = AllocUtils(self.logger)

        # 第一次需要注册下
        self.alloc_interact()

        self.set_event_alloc_interact()

        self.process_run_count = 0

    def set_event_alloc_interact(self):
        set_schedule_event(int(CONFIG.alloc_process_per_seconds), self.alloc_interact, "seconds")

    """
    进程与alloc交互
    """

    def alloc_interact(self):
        code, msg = self.alloc_util.registry()
        # 服务发现的每隔N秒进行, 导致打印的日志比较多, 这里采用出错时打印方法
        if not code:
            self.logger.error(
                f"alloc_proxy alloc interact registry fail, will retry next time, error_msg is {msg}",
                g_not_server_label,
            )

            # 如果本次的注册失败, 表明alloc服务不稳定, 不需要进行下一步操作, 等下一次再操作
            return

    def run_once(self):

        # 启动定时器
        schedule.run_pending()

    """
    进程停止函数
    """

    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("alloc_proxy AllocProxy stop success", g_not_server_label)

    def run(self) -> None:
        self.before_run()

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
                self.logger.error(
                    f"alloc_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}",
                    g_not_server_label,
                )
