#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
"""
Training metrics collection engine.

Provides a reusable function to fetch and aggregate training metrics from
Prometheus Pushgateway. Each project only needs to define its own metrics_dict
and call collect_training_metrics().
"""


from collections import defaultdict

from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.http_utils import http_utils_request
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


# Default key renames applied to the "basic" category
# 默认键名重命名映射（应用于 "basic" 类别）
DEFAULT_KEY_RENAMES = {
    "actor_predict_succ_cnt": "predict_succ_cnt",
    "actor_load_last_model_succ_cnt": "load_model_succ_cnt",
}


def collect_training_metrics(
    metrics_dict: dict,
    group_by_label: dict = None,
    key_renames: dict = None,
    logger=None,
):
    """Fetch and aggregate training metrics from Prometheus Pushgateway.

    :param metrics_dict: Metric definitions grouped by category.
        Format: {"category": {"metric_name": "sum"|"avg", ...}, ...}
    :param group_by_label: Categories that need label-based grouping.
        Format: {"category_name": "label_key"}, e.g. {"env": "model_id"}
    :param key_renames: Key rename mapping for the "basic" category.
        Defaults to DEFAULT_KEY_RENAMES if None.
        Pass empty dict {} to disable renames.
    :param logger: Optional logger for result output.
    :returns: Aggregated metrics dict, None (Prometheus disabled), or False (request failed).
    """
    if not CONFIG.use_prometheus:
        return None

    if group_by_label is None:
        group_by_label = {}
    if key_renames is None:
        key_renames = DEFAULT_KEY_RENAMES

    # Step 1: Fetch data from Prometheus Pushgateway
    # 从 Prometheus Pushgateway 获取数据
    url = f"http://{CONFIG.prometheus_pushgateway}/api/v1/metrics"
    resp = http_utils_request(url)
    if not resp:
        return False

    datas = resp.get("data", [])

    # Step 2: Initialize aggregation structures
    # 初始化聚合统计结构
    metrics_sum = {}
    metrics_count = {}
    for category in metrics_dict:
        if category in group_by_label:
            metrics_sum[category] = defaultdict(lambda: defaultdict(float))
            metrics_count[category] = defaultdict(lambda: defaultdict(int))
        else:
            metrics_sum[category] = defaultdict(float)
            metrics_count[category] = defaultdict(int)

    # Step 3: Traverse and aggregate
    # 遍历数据并聚合
    for data in datas:
        for category, metrics in metrics_dict.items():
            for metric, method in metrics.items():
                # Check both original metric name and prefixed metric name
                # 检查原始指标名和带前缀的指标名
                metric_key = None
                if metric in data:
                    metric_key = metric
                elif f"{KaiwuDRLDefine.MONITOR_METRICS_PREFIX}{metric}" in data:
                    metric_key = f"{KaiwuDRLDefine.MONITOR_METRICS_PREFIX}{metric}"

                if not metric_key:
                    continue

                metric_list = data[metric_key].get("metrics", [])

                if category in group_by_label:
                    label_key = group_by_label[category]
                    for d in metric_list:
                        label_value = d.get("labels", {}).get(label_key, "unknown")
                        value = float(d.get("value", 0))
                        metrics_sum[category][label_value][metric] += value
                        metrics_count[category][label_value][metric] += 1
                else:
                    for d in metric_list:
                        value = float(d.get("value", 0))
                        metrics_sum[category][metric] += value
                        metrics_count[category][metric] += 1

    # Step 4: Calculate results
    # 计算平均值和总和
    metrics_result = {}
    for category in metrics_dict:
        if category in group_by_label:
            grouped_result = {}
            for group_key in metrics_sum[category]:
                group_metrics = {}
                for metric, method in metrics_dict[category].items():
                    total = metrics_sum[category][group_key].get(metric, 0)
                    count = metrics_count[category][group_key].get(metric, 0)
                    if method == "avg" and count > 0:
                        group_metrics[metric] = round(total / count, 2)
                    elif method == "sum":
                        group_metrics[metric] = round(total, 2)
                grouped_result[group_key] = group_metrics
            if grouped_result:
                metrics_result[category] = grouped_result
        else:
            category_result = {}
            for metric, method in metrics_dict[category].items():
                total = metrics_sum[category].get(metric, 0)
                count = metrics_count[category].get(metric, 0)
                if method == "avg" and count > 0:
                    category_result[metric] = round(total / count, 2)
                elif method == "sum":
                    category_result[metric] = round(total, 2)
            if category_result:
                metrics_result[category] = category_result

    # Step 5: Rename keys in "basic" category
    # 重命名 "basic" 类别中的键名
    if "basic" in metrics_result:
        basic = metrics_result["basic"]
        for old_key, new_key in key_renames.items():
            if old_key in basic:
                basic[new_key] = basic.pop(old_key)

    # Step 6: Log results if logger provided
    # 如有 logger 则输出日志
    if logger and metrics_result:
        for key, value in metrics_result.items():
            if key in group_by_label:
                for group_key, group_value in value.items():
                    logger.info(f"training_metrics {key} {group_key} is {group_value}")
            else:
                logger.info(f"training_metrics {key} is {value}")

    return metrics_result
