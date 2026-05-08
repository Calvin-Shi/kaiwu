#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
普罗米修斯的官网见:https://prometheus.io/
需要安装prometheus_client, 采用pip install prometheus_client, 见: https://github.com/prometheus/client_python
"""

"""
下面是KaiwuDRL上报到普罗米修斯的指标, 容器方面的采集指标复用k8s自带的

采用Counter、Gauge、Histogram、Summary

aisrv
1. aisrv --> actor --> aisrv的平均时延, 最大时延
2. aisrv的QPS
3. aisrv进程的CPU, 内存占用

actor
1. actor的单次预测平均时延, 最大时延
2. actor的GPU使用率

learner
1. learner的GPU使用率
2. learner的单次预测平均时延, 最大时延

"""


from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    Gauge,
    push_to_gateway,
    CollectorRegistry,
    start_http_server,
)

from prometheus_client.exposition import basic_auth_handler
from common_python.utils.common_func import get_host_ip
from common_python.utils.http_utils import http_utils_request, http_utils_delete
import os
import time


"""
采用配置类方式, 支持多个配置项

使用实例:
config = PrometheusConfig(
    pwd='your_password',
    user='your_username',
    pushgateway='http://pushgateway.example.com',
    instance='instance_name',
    db='prometheus_db_name'
)

使用方式prometheus_utils = PrometheusUtils(config=config, logger=logger)

"""


class PrometheusConfig:
    def __init__(
        self,
        pwd,
        user,
        pushgateway,
        instance,
        db,
        task_id,
        app,
        check_prometheus_way_availability=False,
        check_prometheus_way_availability_per_seconds=180,
        prometheus_stat_per_minutes=60,
    ):
        # 下面是普罗米修斯的配置
        self.prometheus_pwd = pwd
        self.prometheus_user = user
        self.prometheus_pushgateway = pushgateway
        self.prometheus_instance = instance
        self.prometheus_db = db

        # 下面是一些配置
        self.task_id = task_id
        self.app = app
        self.check_prometheus_way_availability = check_prometheus_way_availability
        self.check_prometheus_way_availability_per_seconds = check_prometheus_way_availability_per_seconds
        self.prometheus_stat_per_minutes = prometheus_stat_per_minutes


# 普罗米修斯监控方面
class PrometheusUtils(object):
    def __init__(self, logger, config, should_clear_data=False) -> None:

        # 下面是参数配置
        self.prometheus_pwd = config.prometheus_pwd
        self.prometheus_user = config.prometheus_user
        self.prometheus_pushgateway = config.prometheus_pushgateway
        self.prometheus_instance = config.prometheus_instance
        self.prometheus_db = config.prometheus_db
        self.task_id = config.task_id
        self.app = config.app
        self.check_prometheus_way_availability = config.check_prometheus_way_availability
        self.check_prometheus_way_availability_per_seconds = config.check_prometheus_way_availability_per_seconds
        self.prometheus_stat_per_minutes = config.prometheus_stat_per_minutes

        # 本机IP名
        self.host = get_host_ip()

        self.container_index = os.getenv("CONTAINER_INDEX", "0")

        # task_id
        self.task_id = self.task_id

        # 上报进程的pid
        self.current_pid = os.getpid()

        # job名, 格式固定, 采用多label形式
        self.job = f"kaiwu_jobs_{self.host}_{self.container_index}_{self.current_pid}"

        self.logger = logger

        # 注意每次push后复用问题
        self.registry = CollectorRegistry()

        # 注意每次定义时不能重复, 格式是{srv_name}_{item_name}, 确保每一项指标有对应的数据结构, 故采用map形式
        self.g_maps = {}
        self.c_maps = {}
        self.h_maps = {}
        self.s_maps = {}

        """
        label的设计:
        1. 如果是多进程里调用, 比如aisrv, learner, 则需要带上pid, 即host_index_pid
        2. 如果是单进程里调用, 比如kaiwu_env, 则不需要带上pid, 即host_index

        默认的labels, 需要区分出来各个进程的情况, task_id, instance规避不同进程间的数量覆盖的问题
        """
        self.default_labels = ["task_id", "app", "instance"]

        """
        设计时只是learner在清理数据, 其余进程不需要
        """
        self.should_clear_data = should_clear_data

        # 最后探测push_gateway是否健康的时间
        self.last_check_push_gateway_available_time = 0
        self.last_check_result = True
        self.last_check_success = False

        self.expired_threshold_seconds = (
            10 * self.prometheus_stat_per_minutes * 60
        )  # 过期阈值（秒）| Expiration threshold in seconds
        self.delete_batch_size = 10  # 每次删除的 job 数量上限 | Max jobs to delete per batch
        self.delete_interval_seconds = (
            10 * self.prometheus_stat_per_minutes * 60
        )  # 删除间隔（秒）| Deletion interval in seconds
        self.last_delete_expired_time = time.time()  # 上次删除时间 | Last deletion timestamp

        # 支持的进程名称
        self.valid_server_process_name = ["aisrv", "actor", "learner", "client", "gamecore", "arena", "env"]

        # 初始化时清理一次过期数据 | Clean up expired data once during initialization
        if self.should_clear_data:
            self._initial_cleanup()

    def prometheus_start_http_server(self, port):
        """
        机器上启动普罗米修斯服务器
        """
        start_http_server(port)

    # 检测进程名是否属于范围内
    def check_server_name(self, server_name):
        if server_name not in self.valid_server_process_name:
            self.logger.error(f"server_name {server_name} is not valid")
            return False

        return True

    # 认证
    def auth_handler(self, url, method, timeout, headers, data):
        return basic_auth_handler(
            url,
            method,
            timeout,
            headers,
            data,
            self.prometheus_user,
            self.prometheus_pwd,
        )

    # Counter使用, 只是增加不减少
    def counter_use(self, server_name, item_name, item_help, value, pid=None, model_id=None):
        if not self.check_server_name(server_name):
            return

        if not pid:
            pid = self.current_pid

        # item_help设置为""减少监控数据
        item_help = ""

        instance = f"kaiwu_tasks_{self.host}_{self.container_index}_{pid}"

        label_names = self.default_labels.copy()
        if model_id is not None:
            label_names.append("model_id")

        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.c_maps:
            self.c_maps[metric_key] = Counter(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=label_names,
            )

        if model_id is not None:
            self.c_maps[metric_key].labels(self.task_id, self.app, instance, model_id).inc(value)
        else:
            self.c_maps[metric_key].labels(self.task_id, self.app, instance).inc(value)

    # Histogram使用, 直方图
    def histogram_use(self, server_name, item_name, item_help, value, pid=None, model_id=None):
        if not self.check_server_name(server_name):
            return

        if not pid:
            pid = self.current_pid

        # item_help设置为""减少监控数据
        item_help = ""

        instance = f"kaiwu_tasks_{self.host}_{self.container_index}_{pid}"

        label_names = self.default_labels.copy()
        if model_id is not None:
            label_names.append("model_id")

        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.h_maps:
            self.h_maps[metric_key] = Histogram(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=label_names,
            )

        if model_id is not None:
            self.h_maps[metric_key].labels(self.task_id, self.app, instance, model_id).observe(value)
        else:
            self.h_maps[metric_key].labels(self.task_id, self.app, instance).observe(value)

    # Summary使用
    def summary_use(self, server_name, item_name, item_help, value, pid=None, model_id=None):
        if not self.check_server_name(server_name):
            return

        if not pid:
            pid = self.current_pid

        # item_help设置为""减少监控数据
        item_help = ""

        instance = f"kaiwu_tasks_{self.host}_{self.container_index}_{pid}"

        label_names = self.default_labels.copy()
        if model_id is not None:
            label_names.append("model_id")

        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.s_maps:
            self.s_maps[metric_key] = Summary(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=label_names,
            )

        if model_id is not None:
            self.s_maps[metric_key].labels(self.task_id, self.app, instance, model_id).observe(value)
        else:
            self.s_maps[metric_key].labels(self.task_id, self.app, instance).observe(value)

    # Gauge使用, 可增可减
    def gauge_use(self, server_name, item_name, item_help, item_value, pid=None, model_id=None):
        if not self.check_server_name(server_name):
            return

        if not pid:
            pid = self.current_pid

        # item_help设置为""减少监控数据
        item_help = ""

        instance = f"kaiwu_tasks_{self.host}_{self.container_index}_{pid}"

        label_names = self.default_labels.copy()
        if model_id is not None:
            label_names.append("model_id")

        # 需要保证是调用第一次来定义Gauge, 并且lablenames不能带上item_name
        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.g_maps:
            self.g_maps[metric_key] = Gauge(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=label_names,
            )

        if model_id is not None:
            self.g_maps[metric_key].labels(self.task_id, self.app, instance, model_id).set(item_value)
        else:
            self.g_maps[metric_key].labels(self.task_id, self.app, instance).set(item_value)

    def is_push_gateway_healthy(self):
        """
        探测push_gate_way是否是健康
        1. 如果不需要探测是否健康, 则直接返回True
        2. 如果没有达到需要探测健康的时间间隔, 则直接返回True
        3. 如果需要探测是否健康
        3.1 健康则返回True
        3.2 不健康则返回False
        """

        if not self.check_prometheus_way_availability:
            return True

        current_time = time.time()
        # 计算时间间隔, 没有探测成功就加快探测, 探测成功就按照常规配置探测
        if self.last_check_success:
            check_interval = self.check_prometheus_way_availability_per_seconds
        else:
            check_interval = int(self.check_prometheus_way_availability_per_seconds / 10)

        if current_time - self.last_check_push_gateway_available_time < check_interval:
            return self.last_check_result

        try:
            self.last_check_push_gateway_available_time = current_time

            # 针对url的设置
            url = self.prometheus_pushgateway
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"http://{url}"
            url = f"{url}/-/healthy"

            resp = http_utils_request(url, print_error_msg=False)
            if resp == "OK":
                self.last_check_result = True
                self.last_check_success = True
            else:
                self.last_check_result = False
                self.last_check_success = False
        except Exception as e:
            self.last_check_result = False
            self.last_check_success = False

        return self.last_check_result

    def push_to_prometheus_gateway(self):
        """
        1. 如果是push模式则需要调用:
            由于每次push_to_gateway需要和网络调用,
            故需要调用N次gauge_use或者summary_use或者histogram_use或者counter_use后再调用push_to_prometheus_gateway, 减少网络耗时
            不能每次就调用push_to_gateway

        2. 如果是pull模式不需要调用
        """
        try:
            if self.is_push_gateway_healthy():
                push_to_gateway(
                    self.prometheus_pushgateway,
                    job=self.job,
                    registry=self.registry,
                    handler=self.auth_handler,
                )
            else:
                # 此时可能存在情况是push_gate_way已经不正常了, 打印日志即可
                self.logger.info(
                    f"push_to_gateway is not healthy, prometheus_pushgateway is {self.prometheus_pushgateway}"
                )
        except Exception as e:
            self.logger.info(
                f"push_to_gateway failed, error is {str(e)}, prometheus_pushgateway is {self.prometheus_pushgateway}"
            )
        finally:
            # 确保无论如何都执行清理
            self.clear_data()

    def clear_data(self):
        """
        清理的数据包括:
        1. 客户端上报时的CollectorRegistry
        2. pushgateway上一段时间的旧数据

        支持两种清理模式:
        - 自动清理过期 job: 根据上报时间自动删除过期的 job
        - 手动清理当前 job: 删除当前 job 的所有数据（原有逻辑）
        """
        if not self.should_clear_data:
            return

        now = time.time()

        # 自动清理过期 job
        self._auto_delete_expired_jobs(now)

    def _auto_delete_expired_jobs(self, now, force=False):
        """
        自动删除过期的 job | Auto delete expired jobs

        工作流程 | Workflow:
        1. 从 Push Gateway 获取所有 job 及其最后上报时间 | Get all jobs and their last push time from Push Gateway
        2. 筛选出超过阈值未上报的 job | Filter out jobs that haven't pushed for longer than threshold
        3. 分批删除这些过期 job | Delete these expired jobs in batches

        Args:
            now: 当前时间戳 | Current timestamp
            force: 是否强制执行（忽略时间间隔检查）| Whether to force execution (ignore interval check)
        """
        # 检查是否到达删除间隔（force=True 时跳过）| Check if deletion interval has been reached (skip when force=True)
        if not force and now - self.last_delete_expired_time < self.delete_interval_seconds:
            return

        try:
            # 1. 获取所有过期的 job
            expired_jobs = self._get_expired_jobs()

            if not expired_jobs:
                self.logger.info("monitor_proxy No expired jobs found")
                self.last_delete_expired_time = now
                return

            # 2. 限制每次删除的数量
            jobs_to_delete = expired_jobs[: self.delete_batch_size]

            self.logger.info(
                f"monitor_proxy Found {len(expired_jobs)} expired jobs, will delete {len(jobs_to_delete)} jobs this batch"
            )

            # 3. 执行删除
            success_count = 0
            failed_count = 0

            for job_info in jobs_to_delete:
                job_name = job_info["job"]
                last_push_time = job_info.get("last_push_time", "unknown")

                delete_url = f"{self.prometheus_pushgateway}/metrics/job/{job_name}"
                if not delete_url.startswith("http://") and not delete_url.startswith("https://"):
                    delete_url = f"http://{delete_url}"

                status, success = http_utils_delete(
                    url=delete_url, auth=(self.prometheus_user, self.prometheus_pwd), print_error_msg=False
                )

                if success:
                    success_count += 1
                    self.logger.info(
                        f"monitor_proxy Deleted expired job: {job_name}, last_push: {last_push_time}, status: {status}"
                    )
                else:
                    failed_count += 1
                    self.logger.warning(f"monitor_proxy Failed to delete expired job: {job_name}, status: {status}")

            self.logger.info(
                f"monitor_proxy Auto delete expired jobs completed: success={success_count}, failed={failed_count}, "
                f"remaining={len(expired_jobs) - len(jobs_to_delete)}"
            )

        except Exception as e:
            self.logger.error(f"monitor_proxy Error in auto delete expired jobs: {str(e)}")
        finally:
            self.last_delete_expired_time = now

    def _get_expired_jobs(self):
        """
        从 Push Gateway 获取所有过期的 job

        Returns:
            list: 过期 job 列表，每个元素包含 {'job': job_name, 'last_push_time': timestamp}
        """
        try:
            # Push Gateway API: GET /api/v1/metrics
            # 返回所有 job 的 metrics 数据，包含 push_time_seconds
            metrics_url = f"{self.prometheus_pushgateway}/api/v1/metrics"
            if not metrics_url.startswith("http://") and not metrics_url.startswith("https://"):
                metrics_url = f"http://{metrics_url}"

            # 发送 GET 请求
            response = http_utils_request(url=metrics_url, print_error_msg=False)

            if not response:
                self.logger.warning("monitor_proxy Failed to get metrics from Push Gateway")
                return []

            # 解析响应数据
            # Push Gateway 的 /api/v1/metrics 返回格式:
            # {
            #   "status": "success",
            #   "data": [
            #     {
            #       "labels": {"job": "job_name", "instance": "instance_name", ...},
            #       "last_push_successful": true,
            #       "last_push_timestamp": "2024-01-01T12:00:00Z",
            #       ...
            #     }
            #   ]
            # }

            if response.get("status") != "success":
                self.logger.warning(f"monitor_proxy Get metrics failed, status: {response.get('status')}")
                return []

            data = response.get("data", {})
            if isinstance(data, list):
                # 如果返回的是 JSON 列表格式 | If data is JSON list format
                return self._parse_metrics_json(data)
            else:
                # 其他格式 | Other format
                self.logger.warning(f"monitor_proxy Unexpected data format: {type(data)}")
                return self._parse_metrics_json(data)

        except Exception as e:
            self.logger.error(f"monitor_proxy Error getting expired jobs: {str(e)}")
            return []

    def _parse_metrics_json(self, data):
        """
        解析 Push Gateway 返回的 JSON 格式 metrics

        示例数据格式:
        [
            {
                "labels": {"job": "kaiwu_jobs_172.17.1.133_0_275"},
                "push_time_seconds": {
                    "time_stamp": "2025-11-24T11:22:30.624852378Z",
                    "type": "GAUGE",
                    "metrics": [{"labels": {"instance": "", "job": "..."}, "value": "1.7639833506248524e+09"}]
                }
            }
        ]
        """
        expired_jobs = []
        now = time.time()
        threshold_seconds = self.expired_threshold_seconds

        job_push_times = {}  # {job_name: last_push_time}

        # 遍历每个 job 数据 | Iterate through each job data
        if isinstance(data, list):
            for item in data:
                # 获取 job 名称 | Get job name
                job_name = item.get("labels", {}).get("job")
                if not job_name:
                    continue

                # 获取 push_time_seconds 数据 | Get push_time_seconds data
                push_time_data = item.get("push_time_seconds", {})
                if not push_time_data:
                    continue

                # 从 metrics 数组中提取时间戳 | Extract timestamp from metrics array
                metrics = push_time_data.get("metrics", [])
                for metric in metrics:
                    value_str = metric.get("value")
                    if value_str:
                        try:
                            # 解析时间戳（支持科学计数法）| Parse timestamp (supports scientific notation)
                            push_time = float(value_str)

                            # 记录每个 job 的最新 push 时间 | Record the latest push time for each job
                            if job_name not in job_push_times or push_time > job_push_times[job_name]:
                                job_push_times[job_name] = push_time
                        except (ValueError, TypeError):
                            continue

        # 筛选过期的 job | Filter expired jobs
        for job_name, push_time in job_push_times.items():
            age_seconds = now - push_time
            if age_seconds > threshold_seconds:
                expired_jobs.append({"job": job_name, "last_push_time": push_time, "age_hours": age_seconds / 3600})

        # 按过期时间排序（最旧的优先删除）| Sort by push time (oldest first)
        expired_jobs.sort(key=lambda x: x["last_push_time"])

        return expired_jobs

    def get_all_jobs_info(self):
        """
        获取 Push Gateway 上所有 job 的信息

        Returns:
            list: job 信息列表，每个元素包含 {'job': job_name, 'last_push_time': timestamp, 'age_hours': hours}
        """
        try:
            metrics_url = f"{self.prometheus_pushgateway}/api/v1/metrics"
            if not metrics_url.startswith("http://") and not metrics_url.startswith("https://"):
                metrics_url = f"http://{metrics_url}"

            response = http_utils_request(url=metrics_url, print_error_msg=False)

            if not response or response.get("status") != 200:
                return []

            data = response.get("data", {})
            if isinstance(data, str):
                return self._parse_all_jobs_from_text(data)
            else:
                return self._parse_all_jobs_from_json(data)

        except Exception as e:
            self.logger.error(f"Error getting all jobs info: {str(e)}")
            return []

    def _parse_all_jobs_from_text(self, metrics_text):
        """解析所有 job 信息（文本格式）"""
        import re

        now = time.time()
        pattern = r'push_time_seconds\{job="([^"]+)"[^}]*\}\s+(\d+(?:\.\d+)?)'

        job_push_times = {}
        for line in metrics_text.split("\n"):
            match = re.search(pattern, line)
            if match:
                job_name = match.group(1)
                push_time = float(match.group(2))
                if job_name not in job_push_times or push_time > job_push_times[job_name]:
                    job_push_times[job_name] = push_time

        all_jobs = []
        for job_name, push_time in job_push_times.items():
            age_seconds = now - push_time
            all_jobs.append({"job": job_name, "last_push_time": push_time, "age_hours": age_seconds / 3600})

        all_jobs.sort(key=lambda x: x["last_push_time"], reverse=True)
        return all_jobs

    def _parse_all_jobs_from_json(self, data):
        """解析所有 job 信息（JSON 格式）"""
        # 类似 _parse_metrics_json 的实现
        return []

    def _initial_cleanup(self):
        """
        初始化时清理过期数据，使用大批量删除 | Initial cleanup of expired data with large batch size

        工作流程 | Workflow:
        1. 临时设置大批量 batch_size (1000) | Temporarily set large batch_size (1000)
        2. 强制执行一次清理（忽略时间间隔限制）| Force execute cleanup once (ignore interval check)
        3. 恢复原始 batch_size | Restore original batch_size
        """
        # 保存原始的 batch_size | Save original batch_size
        original_batch_size = self.delete_batch_size

        try:
            # 临时设置为 1000 用于初始化清理 | Temporarily set to 1000 for initial cleanup
            self.delete_batch_size = 1000

            self.logger.info(
                f"monitor_proxy Initial cleanup started with batch_size={self.delete_batch_size}, "
                f"original_batch_size={original_batch_size}"
            )

            # 强制执行一次清理（force=True 忽略时间间隔检查）| Force execute cleanup once (force=True to ignore interval check)
            self._auto_delete_expired_jobs(time.time(), force=True)

            self.logger.info(f"monitor_proxy Initial cleanup completed, restored batch_size to {original_batch_size}")

        except Exception as e:
            self.logger.error(f"monitor_proxy Error in initial cleanup: {str(e)}")
        finally:
            # 确保恢复原始值 | Ensure restore original value
            self.delete_batch_size = original_batch_size
