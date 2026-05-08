#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import multiprocessing
import queue
import datetime
import os
import signal
import time
import random
import traceback
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from common_python.utils.common_define import CommonDefine
from common_python.config.config_control import CONFIG
from common_python.monitor.prometheus_utils import PrometheusUtils, PrometheusConfig


class MonitorProxy(multiprocessing.Process):
    """
    此类用于aisrv, actor, learner进程与监控产品(当前是普罗米修斯, 后期可以按照需要调整)
    独立出进程, 减少核心路径消耗
    """

    def __init__(self, file_path=None, section=None) -> None:
        # 进程pid
        self.current_pid = os.getpid()
        super().__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

        self.config_file_path = file_path
        self.config_section = section

        """
        该队列使用场景:
        1. aisrv,actor,learner主进程向该队列放置监控的数据
        2. monitor_proxy从该队列拿出需要监控的数据, 发往普罗米修斯
        """
        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 记录最后上报监控的时间
        self.last_report_monitor_time = 0
        self.monitor_data_count_per_minutes = 0

        # get_data错误计数器，用于控制错误日志打印频率
        self.get_data_error_count = 0
        # 每1000次错误打印一次日志
        self.get_data_error_log_interval = 1000

    def before_run(self):
        # 在 spawn 模式下,子进程需要重新解析配置 (fork 模式下配置已在内存中,重新解析也不影响)
        # In spawn mode, child process needs to reload config (no effect on fork mode as config is already in memory)
        if self.config_section and self.config_file_path:
            print(
                f"MonitorProxy before_run: config_section={self.config_section}, config_file_path={self.config_file_path}"
            )

            # 设置配置文件路径
            # Set config file path
            CONFIG.set_configure_file(self.config_file_path)

            # 根据 section 调用对应的解析方法（会自动加载依赖的其他配置）
            # Call corresponding parse method based on section (will auto-load dependent configs)
            if self.config_section == "aisrv":
                CONFIG.parse_aisrv_configure()
            elif self.config_section == "actor":
                CONFIG.parse_actor_configure()
            elif self.config_section == "learner":
                CONFIG.parse_learner_configure()
            else:
                # 其他 section，使用通用解析（加载该 section + main）
                # Other sections, use generic parse (load that section + main)
                CONFIG.parse_configure([self.config_section], self.config_file_path)

            print(
                f"MonitorProxy before_run: CONFIG.idle_sleep_second={CONFIG.idle_sleep_second}, CONFIG.prometheus_stat_per_minutes={CONFIG.prometheus_stat_per_minutes}"
            )
        else:
            print(
                f"MonitorProxy before_run: No config provided, config_section={self.config_section}, config_file_path={self.config_file_path}"
            )

        # 子进程启动后,重新获取实际的进程 PID | Re-get actual process PID after subprocess starts
        self.current_pid = os.getpid()

        # 日志处理 | Log handling
        self.logger = KaiwuLogger()
        self.logger.set_logger_format(
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/monitor_proxy_pid{self.current_pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "monitor_proxy",
        )
        self.logger.info(
            f"monitor_proxy started at PID {self.current_pid}, serving as cross-process shared singleton. "
            f"All Workers will use this single monitor_proxy process.",
        )

        config = PrometheusConfig(
            pwd=CONFIG.prometheus_pwd,
            user=CONFIG.prometheus_user,
            pushgateway=CONFIG.prometheus_pushgateway,
            instance=CONFIG.prometheus_instance,
            db=CONFIG.prometheus_db,
            task_id=CONFIG.task_id,
            app=CONFIG.app,
            check_prometheus_way_availability=CONFIG.check_prometheus_way_availability,
            check_prometheus_way_availability_per_seconds=CONFIG.check_prometheus_way_availability_per_seconds,
            prometheus_stat_per_minutes=CONFIG.prometheus_stat_per_minutes,
        )

        # PrometheusUtils 工具类, 与普罗米修斯交互操作
        if (
            CONFIG.svr_name == CommonDefine.SERVER_ENV
            or CONFIG.svr_name == CommonDefine.SERVER_ARENA
            or CONFIG.svr_name == CommonDefine.SERVER_GAMECORE
        ):
            self.prometheus_utils = PrometheusUtils(self.logger, config, False)
        elif CONFIG.svr_name == CommonDefine.SERVER_LEARNER:
            self.prometheus_utils = PrometheusUtils(
                self.logger, config, getattr(CONFIG, "learner_should_clear_data", False)
            )
        elif CONFIG.svr_name == CommonDefine.SERVER_AISRV or CONFIG.svr_name == CommonDefine.SERVER_ACTOR:
            self.prometheus_utils = PrometheusUtils(self.logger, config, False)
        else:
            # 默认
            self.prometheus_utils = PrometheusUtils(self.logger, config, False)
        self.process_run_count = 0

        # pull模式需要启动server
        if CONFIG.use_prometheus_way == CommonDefine.USE_PROMETHEUS_WAY_PULL:
            self.prometheus_utils.prometheus_start_http_server(CONFIG.prometheus_server_port)

        # 在before run最后打印启动成功日志
        self.logger.info(f"monitor_proxy process start success at pid is {self.current_pid}")

        return True

    def put_data(self, monitor_data):
        """
        monitor_data采用map形式, 即key/value格式, 监控指标/监控值
        """
        if not monitor_data:
            return False

        # 检查label数量，如果大于4个直接丢弃
        label_count = len(monitor_data)
        if label_count > 4:
            self.logger.error(f"monitor_proxy put_data label count {label_count} > 4, data will be discarded")
            return False

        if self.msg_queue.full():
            return False
        else:
            self.msg_queue.put(monitor_data)
            return True

    def get_data(self):
        """
        采用queue.Queue类的get方法, 减少CPU损耗
        增加异常处理，防止EOFError导致进程崩溃
        错误日志采用计数方式，每1000次打印一次，避免日志过多
        """
        try:
            return self.msg_queue.get(timeout=1.0)  # 添加超时避免永久阻塞
        except queue.Empty:
            return None
        except (EOFError, BrokenPipeError, ConnectionResetError) as e:
            # Queue连接断开（通常是发送端进程退出）
            self.get_data_error_count += 1
            if self.get_data_error_count % self.get_data_error_log_interval == 1:
                self.logger.warning(
                    f"monitor_proxy msg_queue connection lost: {e}, returning None "
                    f"(error count: {self.get_data_error_count}, logged every {self.get_data_error_log_interval} times)"
                )
            return None
        except Exception as e:
            self.get_data_error_count += 1
            if self.get_data_error_count % self.get_data_error_log_interval == 1:
                self.logger.error(
                    f"monitor_proxy get_data unexpected error: {e} "
                    f"(error count: {self.get_data_error_count}, logged every {self.get_data_error_log_interval} times)"
                )
            return None

    def send_to_prometheus(self, monitor_data):
        if not monitor_data:
            return

        if not isinstance(monitor_data, dict):
            self.logger.error(f"monitor_proxy monitor_data is not dict, return")
            return

        # 根据监控版本确定指标名称前缀 | Determine metrics name prefix based on monitor version
        metrics_name_prefix = ""
        if CONFIG.monitor_version == CommonDefine.MONITOR_VERSION_V2:
            metrics_name_prefix = CommonDefine.MONITOR_METRICS_PREFIX

        """
        注意数据结构是{ pid : {key : value}}这种
        """
        model_id = None
        # 获取到对应的model_id
        if "model_id" in monitor_data:
            model_id = monitor_data["model_id"]
            del monitor_data["model_id"]

        # 只是剩下监控数据
        for pid, data in monitor_data.items():
            for monitor_name, monitor_value in data.items():
                # 添加指标名称前缀 | Add metrics name prefix
                full_monitor_name = f"{metrics_name_prefix}{monitor_name}"

                if isinstance(monitor_value, list):
                    for i in range(len(monitor_value)):
                        self.prometheus_utils.gauge_use(
                            CONFIG.svr_name,
                            full_monitor_name,
                            full_monitor_name,
                            monitor_value[i],
                            pid,
                            model_id,
                        )
                else:
                    self.prometheus_utils.gauge_use(
                        CONFIG.svr_name, full_monitor_name, full_monitor_name, monitor_value, pid, model_id
                    )

        # # 由于多个进程可能同时启动同时上报监控, 导致出现pushgateway的性能瓶颈, 故这里随机延长启动
        # random_number = random.randint(1, 30)
        # time.sleep(random_number)

        # push 模式需要主动推送
        if CONFIG.use_prometheus_way == CommonDefine.USE_PROMETHEUS_WAY_PUSH:
            self.prometheus_utils.push_to_prometheus_gateway()

        # self.logger.debug(f'monitor_proxy push_to_prometheus_gateway success')

    def run_once(self):

        # 获取需要监控的数据（带异常处理）
        monitor_data = self.get_data()
        if monitor_data:
            now = time.time()
            if now - self.last_report_monitor_time >= CONFIG.prometheus_stat_per_minutes * 60:
                self.monitor_data_count_per_minutes = 0
                self.last_report_monitor_time = now

            # 满足大于最小的CONFIG.min_report_monitor_seconds即开始上报避免普罗米修斯服务雪崩, 否则这期间的监控数据被丢弃并且打印日志
            if self.monitor_data_count_per_minutes < CONFIG.max_report_monitor_count_per_minutes:
                self.send_to_prometheus(monitor_data)
                self.monitor_data_count_per_minutes += 1
            else:
                self.logger.error(
                    f"monitor_proxy, monitor_data_count_per_minutes {self.monitor_data_count_per_minutes} >= CONFIG.max_report_monitor_count_per_minutes {CONFIG.max_report_monitor_count_per_minutes}, so monitor_data {monitor_data} will drop"
                )
        # 注释：不论是否获取到数据都继续循环，避免因get_data()异常导致进程退出

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("monitor_proxy MonitorProxy stop success")

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"monitor_proxy before_run failed, so return")
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                time.sleep(CONFIG.idle_sleep_second)

            except Exception as e:
                self.logger.error(
                    f"monitor_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}"
                )
