#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from typing import Dict, List, Optional, Any
import re
import yaml
import os
from common_python.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
import importlib


class MonitorConfigError(ValueError):
    """配置构建异常类"""

    pass


def add_log_prefix(msg: str, prefix: str = None) -> str:
    """
    为日志消息添加前缀的辅助函数

    Args:
        msg: 原始日志消息
        prefix: 要添加的前缀字符串，默认使用 MONITOR_INIT_LOG_FILTER

    Returns:
        添加前缀后的消息字符串

    Example:
        >>> add_log_prefix("开始加载配置")
        "monitor_init 开始加载配置"
    """
    if prefix is None:
        prefix = KaiwuDRLDefine.MONITOR_INIT_LOG_FILTER
    return f"{prefix} {msg}"


class MonitorConfigBuilder:
    # 字段校验正则表达式（已修改为支持中英文混合）
    # 标题：支持中英文、数字、指定符号（=+\/@#_-）、空格，1~100字符
    TITLE_PATTERN = re.compile(r"^[a-zA-Z0-9一-龥=+\/@#_\-\s]{1,100}$")
    # 面板组中文名称：支持中英文、数字、_-、空格，1~20字符
    GROUP_NAME_CN_PATTERN = re.compile(r"^[a-zA-Z一-龥0-9_\-\s]{1,20}$")
    # 面板组英文名称：（英文、数字、_-、空格），0~50字符（可选）
    GROUP_NAME_EN_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\s]{0,50}$")
    # 面板中文名称：支持中英文、数字、_-、空格，1~20字符
    PANEL_NAME_CN_PATTERN = re.compile(r"^[a-zA-Z一-龥0-9_\-\s]{1,20}$")
    # 面板英文名称：0~50字符（可选）
    PANEL_NAME_EN_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\s]{0,50}$")
    # 面板中文描述：支持中英文、数字、标点符号及常见特殊字符，0~200字符
    DESCRIPTION_CN_PATTERN = re.compile(
        r"^[a-zA-Z一-龥0-9\s~`!@#$%^&*()_\-+={}[\]|\\:;\"'<>,.?/，。！？；：" "''（）【】、]{0,200}$"
    )
    # 面板英文描述：支持英文、数字、标点符号及常见特殊字符，0~1000字符
    DESCRIPTION_EN_PATTERN = re.compile(r"^[a-zA-Z0-9\s~`!@#$%^&*()_\-+={}[\]|\\:;\"'<>,.?/]{0,1000}$")
    # 指标名称：支持中英文、数字、_-、空格、花括号{}，1~40字符
    METRIC_NAME_PATTERN = re.compile(r"^[a-zA-Z一-龥0-9_\-\s{}]{1,40}$")
    # 支持的图表类型
    SUPPORTED_CHART_TYPES = {"line", "stat"}

    def __init__(self, logger: Optional[KaiwuLogger] = None):
        """初始化监控配置构建器

        Args:
            logger: 可选的日志记录器实例。如果不提供，将创建新的 KaiwuLogger 实例
        """
        self.config: Dict[str, Any] = {"title": "", "groups": []}
        self._current_group: Optional[Dict[str, Any]] = None
        self._current_panel: Optional[Dict[str, Any]] = None
        self.logger = logger if logger is not None else KaiwuLogger()  # 使用传入的logger或创建新实例

    def _validate_title(self, title: str) -> None:
        """校验监控面板名称

        Args:
            title: 监控面板标题，1~100字符，支持中英文、数字及=+/@#_-空格符号

        Returns:
            None

        Raises:
            MonitorConfigError: 标题不符合规范时抛出异常
        """
        if not self.TITLE_PATTERN.match(title):
            error_msg = f"面板名称非法：{title}，需满足1~100字符，仅支持中英文字母、数字及=+/@#_-空格符号"
            self._errors.append(error_msg)

    def _validate_group_name(self, group_name: str, group_name_en: str) -> None:
        """校验面板组名称

        Args:
            group_name: 面板组中文名称，1~20字符（必填），支持中英文、数字及_-空格符号
            group_name_en: 面板组英文名称，0~50字符（可选），支持英文、数字及_-空格符号

        Returns:
            None

        Raises:
            MonitorConfigError: 组名称不符合规范时抛出异常
        """
        if not self.GROUP_NAME_CN_PATTERN.match(group_name):
            error_msg = f"面板组中文名称非法：{group_name}，需满足1~20字符，仅支持中英文、数字及_-空格符号"
            self._errors.append(error_msg)
        # group_name_en 为可选字段，仅在非空时校验
        if group_name_en and not self.GROUP_NAME_EN_PATTERN.match(group_name_en):
            error_msg = f"面板组英文名称非法：{group_name_en}，需满足0~50字符，仅支持英文、数字及_-空格符号"
            self._errors.append(error_msg)

    def _validate_panel_name(self, panel_name: str, panel_name_en: str) -> None:
        """校验面板名称

        Args:
            panel_name: 面板中文名称，1~20字符（必填），支持中英文、数字及_-空格符号
            panel_name_en: 面板英文名称，0~50字符（可选），支持英文、数字及_-空格符号

        Returns:
            None

        Raises:
            MonitorConfigError: 面板名称不符合规范或为空时抛出异常
        """
        if len(panel_name) > 20 or (panel_name and not self.PANEL_NAME_CN_PATTERN.match(panel_name)):
            error_msg = f"面板中文名称非法：{panel_name}，需满足1~20字符，仅支持中英文、数字及_-空格符号"
            self._errors.append(error_msg)
        if len(panel_name_en) > 50 or (panel_name_en and not self.PANEL_NAME_EN_PATTERN.match(panel_name_en)):
            error_msg = f"面板英文名称非法：{panel_name_en}，需满足0~50字符，仅支持英文、数字及_-空格符号"
            self._errors.append(error_msg)
        # 强制非空校验（仅校验中文名称）
        if not panel_name:
            error_msg = "面板中文名称不能为空"
            self._errors.append(error_msg)

    def _validate_description(self, description: str, description_en: str) -> None:
        """校验面板描述

        Args:
            description: 面板中文描述，0~200字符（可选），支持中英文、数字、标点符号及常见特殊字符
            description_en: 面板英文描述，0~1000字符（可选），支持英文、数字、标点符号及常见特殊字符

        Returns:
            None

        Raises:
            MonitorConfigError: 描述不符合规范时抛出异常
        """
        if len(description) > 200 or (description and not self.DESCRIPTION_CN_PATTERN.match(description)):
            error_msg = (
                f"面板中文描述非法：{description}，需满足0~200字符，支持中英文、数字、空格及常见特殊符号(~`!@#$%^&*()_-+=[]{{}}|\\:;\"'<>,.?/，。！？；："
                "''（）【】、)"
            )
            self._errors.append(error_msg)
        if len(description_en) > 1000 or (description_en and not self.DESCRIPTION_EN_PATTERN.match(description_en)):
            error_msg = f"面板英文描述非法：{description_en}，需满足0~1000字符，支持英文、数字、空格及常见特殊符号(~`!@#$%^&*()_-+=[]{{}}|\\:;\"'<>,.?/)"
            self._errors.append(error_msg)

    def _validate_metric(self, metrics_name: str) -> None:
        """校验指标名称

        Args:
            metrics_name: 指标名称，1~40字符，支持中英文、数字、_-、空格、花括号{}

        Returns:
            None

        Raises:
            MonitorConfigError: 指标名称不符合规范时抛出异常
        """
        if not self.METRIC_NAME_PATTERN.match(metrics_name):
            error_msg = f"指标名称非法：{metrics_name}，需满足1~40字符，仅支持中英文、数字、_-、空格、花括号{{}}"
            self._errors.append(error_msg)

    def _validate_chart_type(self, chart_type: str) -> None:
        """校验图表类型

        Args:
            chart_type: 图表类型，支持 'line'（折线图）或 'stat'（数值类型）（必填）

        Returns:
            None

        Raises:
            MonitorConfigError: 图表类型为空或不在支持列表中时抛出异常
        """
        # 强制非空校验
        if not chart_type:
            error_msg = "图表类型不能为空"
            self._errors.append(error_msg)
        elif chart_type not in self.SUPPORTED_CHART_TYPES:
            error_msg = f"不支持的图表类型：{chart_type}，仅支持{self.SUPPORTED_CHART_TYPES}"
            self._errors.append(error_msg)

    def _validate_metric_count(self, chart_type: str, metric_count: int) -> None:
        """根据图表类型校验指标数量

        Args:
            chart_type: 图表类型，'line' 或 'stat'
            metric_count: 当前指标数量

        Returns:
            None

        Raises:
            MonitorConfigError: 指标数量超出限制时抛出异常
                - stat类型：0~2个指标
                - line类型：0~20个指标
        """
        if chart_type == "stat" and not (0 <= metric_count <= 2):
            error_msg = f"stat类型面板最多支持2个指标，当前{metric_count}个"
            self._errors.append(error_msg)
        if chart_type == "line" and not (0 <= metric_count <= 20):
            error_msg = f"line类型面板最多支持20个指标，当前{metric_count}个"
            self._errors.append(error_msg)

    def _validate_expr_labels(self, expr: str, metrics_name: str) -> None:
        """校验 expr 中的 label 数量（0~4个）

        解析 PromQL 表达式中的 label 选择器，统计 label 数量并校验是否在限制范围内。

        Args:
            expr: PromQL 表达式，如 'avg(win_rate{task_uuid="$task_uuid"}) by (label)'
            metrics_name: 指标名称，用于错误提示

        Returns:
            None

        Raises:
            MonitorConfigError: label 数量超出 0~4 个限制时抛出异常

        Examples:
            - avg(win_rate{task_uuid="$task_uuid"}) by (label) -> 1个label（合法）
            - sum(metric{env="prod",region="us",app="test"}) -> 3个label（合法）
            - sum(metric{a="1",b="2",c="3",d="4",e="5"}) -> 5个label（非法）
        """
        if not expr or not isinstance(expr, str):
            return

        # 匹配 PromQL 中的 label 选择器：{key="value", key2="value2"}
        # 支持单引号、双引号、反引号，以及不带引号的值
        label_pattern = re.compile(r"\{([^}]+)\}")
        matches = label_pattern.findall(expr)

        if not matches:
            return  # 没有 label，合法

        # 统计所有 label 数量
        total_labels = 0
        for match in matches:
            # 分割每个 label 对（支持逗号分隔）
            # 匹配格式：key="value" 或 key='value' 或 key=`value` 或 key=value
            label_pairs = re.findall(r'(\w+)\s*[=!~]+\s*["\']?[^,}\'"]+["\']?', match)
            total_labels += len(label_pairs)

        # 校验 label 数量
        if not (0 <= total_labels <= 4):
            error_msg = f"指标 [{metrics_name}] 的 expr 中 label 数量超限：当前{total_labels}个，限制0~4个"
            self._errors.append(error_msg)

    def title(self, title: str) -> "MonitorConfigBuilder":
        """设置全局标题（必填）

        Args:
            title: 监控面板标题，1~100字符

        Returns:
            MonitorConfigBuilder: 返回自身以支持链式调用

        Note:
            校验会在 build() 时统一执行
        """
        self.config["title"] = title
        return self

    def add_group(self, group_name: str, group_name_en: str = "") -> "MonitorConfigBuilder":
        """添加面板组（必填）

        Args:
            group_name: 面板组中文名称，1~20字符（必填）
            group_name_en: 面板组英文名称，0~50字符（可选）

        Returns:
            MonitorConfigBuilder: 返回自身以支持链式调用

        Note:
            会自动结束上一个未完成的组，校验会在 build() 时统一执行
        """
        # 自动结束上一个未完成的组
        if self._current_group:
            self.end_group()
        # 初始化新组（不进行即时校验）
        self._current_group = {"group_name": group_name, "group_name_en": group_name_en, "panels": []}
        return self

    def add_panel(
        self,
        name: str,
        name_en: str = "",
        description: str = "",
        description_en: str = "",
        type: str = "",
        unit: str = "",
        custom: str = "",
    ) -> "MonitorConfigBuilder":
        """添加面板（必填）

        Args:
            name: 面板中文名称，1~20字符（必填）
            name_en: 面板英文名称，0~50字符（可选）
            description: 面板中文描述，0~200字符（可选）
            description_en: 面板英文描述，0~1000字符（可选）
            type: 图表类型，'line'（折线图）或 'stat'（数值类型）（必填）
            unit: 单位，可选
            custom: 自定义字段，可选

        Returns:
            MonitorConfigBuilder: 返回自身以支持链式调用

        Raises:
            MonitorConfigError: 未先创建组或面板参数不符合规范时抛出异常

        Note:
            - 如果当前面板名称与新面板名称相同，则复用当前面板（不创建新面板）
            - 如果名称不同，会自动结束上一个未完成的面板并创建新面板
        """
        if not self._current_group:
            error_msg = "Please call add_group() to create a panel group first"
            self.logger.error(error_msg)
            raise MonitorConfigError(error_msg)

        # 检查是否与当前面板同名
        if self._current_panel and self._current_panel.get("panel_name") == name:
            # 同名面板，复用当前面板，不创建新面板
            # 可以选择更新其他属性（如果提供了新值）
            if description:
                self._current_panel["description"] = description
            if description_en:
                self._current_panel["description_en"] = description_en
            if type:
                self._current_panel["type"] = type
            if unit:
                self._current_panel["unit"] = unit
            if custom:
                self._current_panel["custom"] = custom
            return self

        # 不同名，自动结束上一个未完成的面板
        if self._current_panel:
            self.end_panel()

        # 初始化新面板（不进行即时校验）
        self._current_panel = {
            "panel_name": name,
            "panel_name_en": name_en,
            "description": description,
            "description_en": description_en,
            "type": type,
            "unit": unit,
            "custom": custom,
            "metrics": [],
        }
        return self

    def add_metric(self, metrics_name: str, expr: str = "") -> "MonitorConfigBuilder":
        """添加指标（metrics_name必填）

        Args:
            metrics_name: 指标名称，1~40字符（必填）
            expr: PromQL 表达式（可选），其中 label 数量限制为 0~4 个

        Returns:
            MonitorConfigBuilder: 返回自身以支持链式调用

        Raises:
            MonitorConfigError: 未先创建面板、指标名称不符合规范、
                              expr 中 label 数量超限或指标数量超限时抛出异常

        Note:
            - 指标数量限制：stat类型 0~2个，line类型 0~20个
            - expr 中 label 数量限制：0~4个
            - 如果同一面板中已存在同名指标，后添加的指标会覆盖之前的指标
        """
        if not self._current_panel:
            error_msg = "Please call add_panel() to create a panel first"
            self.logger.error(error_msg)
            raise MonitorConfigError(error_msg)

        # 检查是否已存在同名指标
        metrics_list = self._current_panel["metrics"]
        existing_index = None
        for i, metric in enumerate(metrics_list):
            if metric.get("metrics_name") == metrics_name:
                existing_index = i
                break

        # 如果存在同名指标，更新它（后来的优先）
        if existing_index is not None:
            metrics_list[existing_index] = {"metrics_name": metrics_name, "expr": expr}
        else:
            # 不存在同名指标，添加新指标
            metrics_list.append({"metrics_name": metrics_name, "expr": expr})

        return self

    def end_panel(self) -> "MonitorConfigBuilder":
        """结束当前面板"""
        if self._current_panel:
            self._current_group["panels"].append(self._current_panel)
            self._current_panel = None
        return self

    def end_group(self) -> "MonitorConfigBuilder":
        """结束当前面板组"""
        if self._current_group:
            # 确保组内所有面板已结束
            self.end_panel()
            self.config["groups"].append(self._current_group)
            self._current_group = None
        return self

    def merge(self, other_config: Dict[str, Any]) -> "MonitorConfigBuilder":
        """合并另一个配置（带校验）
        核心优先级原则：待合并配置（other_config）优先级 > 当前配置（self）
        逻辑：用待合并配置覆盖相同内容，补充当前配置没有的内容
        """
        try:
            # 1. 先验证待合并配置的合法性（避免非法配置污染当前配置）
            temp_builder = self.from_yaml(other_config)
            temp_builder.build()
        except MonitorConfigError as e:
            self.logger.error(f"Config to be merged is invalid: {e}")
            raise

        # --------------------------
        # 2. 合并标题：待合并配置优先（非空则覆盖）
        # --------------------------
        if other_config.get("title"):
            self.config["title"] = other_config["title"]
        # 若待合并配置标题为空，保留当前配置标题（不覆盖）

        # --------------------------
        # 3. 合并组：按 group_name 去重，待合并配置优先
        # --------------------------
        # 先将当前配置的组转为字典（方便查找）
        existing_groups = {g["group_name"]: g for g in self.config["groups"]}
        # 遍历待合并配置的组，逐个处理
        for other_group in other_config.get("groups", []):
            other_group_name = other_group["group_name"]

            if other_group_name in existing_groups:
                # 3.1 组已存在：用待合并组的面板覆盖/补充当前组
                existing_panels = {p["panel_name"]: p for p in existing_groups[other_group_name]["panels"]}
                # 遍历待合并组的面板
                for other_panel in other_group["panels"]:
                    other_panel_name = other_panel["panel_name"]

                    if other_panel_name in existing_panels:
                        # 3.1.1 面板已存在：用待合并面板的指标覆盖/补充当前面板
                        existing_metrics = {m["metrics_name"]: m for m in existing_panels[other_panel_name]["metrics"]}
                        # 遍历待合并面板的指标
                        for other_metric in other_panel["metrics"]:
                            other_metric_name = other_metric["metrics_name"]
                            # 待合并指标优先：直接覆盖现有指标（或新增）
                            existing_metrics[other_metric_name] = other_metric
                        # 更新面板的指标（待合并指标覆盖后）
                        other_panel["metrics"] = list(existing_metrics.values())
                        # 用待合并面板覆盖现有面板（保留指标合并结果）
                        existing_panels[other_panel_name] = other_panel
                    else:
                        # 3.1.2 面板不存在：直接添加待合并面板
                        existing_panels[other_panel_name] = other_panel
                # 更新组的面板（待合并面板覆盖后）
                existing_groups[other_group_name]["panels"] = list(existing_panels.values())
            else:
                # 3.2 组不存在：直接添加待合并组
                existing_groups[other_group_name] = other_group

        # 4. 用合并后的组更新当前配置
        self.config["groups"] = list(existing_groups.values())

        self.logger.info("Config merge completed")
        return self

    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any], logger: Optional[KaiwuLogger] = None) -> "MonitorConfigBuilder":
        """从YAML字典加载配置（延迟校验，在build时统一执行）

        Args:
            yaml_data: YAML配置字典
            logger: 可选的日志记录器实例

        Returns:
            MonitorConfigBuilder: 配置构建器实例

        Note:
            所有校验会在调用build()时统一执行
        """
        builder = cls(logger=logger)

        # 加载标题
        title = yaml_data.get("title", "")
        if title:
            builder.title(title)

        # 加载组
        for group in yaml_data.get("groups", []):
            group_name = group.get("group_name", "")
            group_name_en = group.get("group_name_en", "")
            builder.add_group(group_name, group_name_en)

            # 加载面板
            for panel in group.get("panels", []):
                panel_name = panel.get("panel_name", "")
                panel_name_en = panel.get("panel_name_en", "")
                builder.add_panel(
                    name=panel_name,
                    name_en=panel_name_en,
                    description=panel.get("description", ""),
                    description_en=panel.get("description_en", ""),
                    type=panel.get("type", ""),
                    unit=panel.get("unit", ""),
                    custom=panel.get("custom", ""),
                )

                # 加载指标
                for metric in panel.get("metrics", []):
                    metrics_name = metric.get("metrics_name", "")
                    builder.add_metric(
                        metrics_name=metrics_name,
                        expr=metric.get("expr", ""),
                    )
                builder.end_panel()
            builder.end_group()

        return builder

    @staticmethod
    def dump_to_yaml_file(config: Dict[str, Any], file_path: str, logger=None) -> None:
        """
        将配置字典写入指定文件，自动创建不存在的前缀目录

        Args:
            config: 已构建的配置字典（通过 build() 生成）
            file_path: 目标文件路径（如 "/workspace/train/monitor.yaml"）
            logger: KaiwuLogger实例，用于记录日志（可选）
        """
        try:
            # 提取目录路径（自动处理多级目录）
            dir_path = os.path.dirname(file_path)
            if dir_path:  # 若存在目录前缀，确保目录存在
                os.makedirs(dir_path, exist_ok=True)  # exist_ok=True 避免目录已存在时报错

            # 写入YAML文件（保持中文显示、字段顺序和缩进格式）
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config,
                    f,
                    allow_unicode=True,  # 确保中文正常显示
                    sort_keys=False,  # 保持字段定义顺序
                    indent=2,  # 缩进2空格，增强可读性
                    default_flow_style=False,  # 禁用流式格式，强制块级显示
                )

            # 记录成功信息（如果提供了logger）
            if logger:
                logger.info(f"Config successfully written to file: {os.path.abspath(file_path)}")

        except Exception as e:
            if logger:
                logger.error(f"Failed to write config file: {str(e)}")
            # 捕获目录创建或文件写入异常
            raise RuntimeError(f"Failed to write config file: {str(e)}") from e

    def build(self) -> Dict[str, Any]:
        """构建最终配置（自动补全未结束的组/面板，统一执行全量校验）"""
        # 清空之前的错误
        self._errors = []

        # 自动结束所有未完成的组
        self.end_group()

        # 1. 校验标题
        if not self.config["title"]:
            self._errors.append("全局配置缺少必填项：title")
        else:
            self._validate_title(self.config["title"])

        # 2. 校验至少有一个组
        if not self.config["groups"]:
            # self._errors.append("配置至少需要包含一个面板组")  # 注释掉这行
            self.logger.info("No panel groups in config, will generate config with title only")

        # 3. 遍历所有组、面板、指标进行校验
        for group_idx, group in enumerate(self.config["groups"], 1):
            group_name = group.get("group_name", "")
            group_name_en = group.get("group_name_en", "")

            # 校验组名称
            self._validate_group_name(group_name, group_name_en)

            # 校验面板
            for panel_idx, panel in enumerate(group.get("panels", []), 1):
                panel_name = panel.get("panel_name", "")
                panel_name_en = panel.get("panel_name_en", "")
                description = panel.get("description", "")
                description_en = panel.get("description_en", "")
                panel_type = panel.get("type", "")

                # 校验面板名称
                self._validate_panel_name(panel_name, panel_name_en)

                # 校验面板描述
                self._validate_description(description, description_en)

                # 校验图表类型
                self._validate_chart_type(panel_type)

                # 校验指标
                metrics = panel.get("metrics", [])
                for metric_idx, metric in enumerate(metrics, 1):
                    metrics_name = metric.get("metrics_name", "")
                    expr = metric.get("expr", "")

                    # 校验指标名称
                    self._validate_metric(metrics_name)

                    # 校验 expr 中的 label 数量
                    self._validate_expr_labels(expr, metrics_name)

                # 校验指标数量
                self._validate_metric_count(panel_type, len(metrics))

        # 4. 如果有错误，统一抛出
        if self._errors:
            error_summary = f"配置校验失败，共发现 {len(self._errors)} 个错误：\n" + "\n".join(
                f"  {i+1}. {err}" for i, err in enumerate(self._errors)
            )
            raise MonitorConfigError(error_summary)

        return self.config


def load_monitor_config_from_yaml(
    yaml_file: str, logger: KaiwuLogger = None, log_prefix: str = None
) -> Optional[Dict[str, Any]]:
    """
    从 YAML 文件加载监控配置

    Args:
        yaml_file: YAML 文件路径
        logger: 可选的日志记录器，如果不提供则创建新实例
        log_prefix: 可选的日志前缀，如果不提供则使用默认的 MONITOR_INIT_LOG_FILTER

    Returns:
        配置字典或 None
    """
    if logger is None:
        logger = KaiwuLogger()

    if os.path.exists(yaml_file):
        logger.info(add_log_prefix(f"Loading monitor file", log_prefix))
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            # 从 YAML 数据构建配置（自动执行全量校验），传入logger实例
            logger.info(
                add_log_prefix(
                    "YAML file read successfully, starting validation and configuration build...", log_prefix
                )
            )
            builder = MonitorConfigBuilder.from_yaml(yaml_data, logger=logger)
            final_config = builder.build()
            logger.info(add_log_prefix("Config build validation successful!", log_prefix))
            return final_config
        except MonitorConfigError as e:
            # 捕获配置校验异常
            logger.error(add_log_prefix(f"Config validation failed: {e}", log_prefix))
        except yaml.YAMLError as e:
            # 捕获 YAML 语法错误
            logger.error(add_log_prefix(f"YAML syntax error: {e}", log_prefix))
        except AssertionError as e:
            # 捕获自定义验证异常
            logger.error(add_log_prefix(f"Config verification failed: {e}", log_prefix))
        except Exception as e:
            # 捕获其他未知异常
            logger.error(add_log_prefix(f"Unknown error: {e}", log_prefix))
    else:
        logger.warning(
            add_log_prefix(f"Failed to load monitor file, {yaml_file} does not exist, please confirm", log_prefix)
        )


def load_user_monitor_config(
    user_file_path: str, logger: KaiwuLogger = None, log_prefix: str = None
) -> Optional[Dict[str, Any]]:
    """
    从用户指定的文件路径加载自定义监控配置，自动推导模块路径和依赖
    仅给 PromQL 指标名添加 kaiwu_ 前缀，完全排除函数名

    Args:
        user_file_path: 用户配置文件路径
        logger: 可选的日志记录器，如果不提供则创建新实例
        log_prefix: 可选的日志前缀，如果不提供则使用默认的 MONITOR_INIT_LOG_FILTER

    Returns:
        配置字典或 None
    """
    if logger is None:
        logger = KaiwuLogger()

    user_file_path = os.path.abspath(user_file_path)

    # 精确匹配 PromQL 指标名（排除函数名）
    # 策略：匹配后面紧跟 { 或 [ 或特定符号的标识符
    #
    # 正则说明：
    # 1. (?:^|(?<=[\(,\s+\-*/])) - 前面是字符串开始、左括号、逗号、空格或运算符
    # 2. ([a-zA-Z_][a-zA-Z0-9_]*) - 捕获组：匹配标识符（指标名）
    # 3. (?=\s*[\{\[\)\s,+\-*/]|$) - 后面是花括号、方括号、右括号、空格、逗号、运算符或字符串结束
    #
    # 这样可以匹配：
    # - avg(win_rate{...}) 中的 win_rate（括号后+花括号前）
    # - avg(cpu_usage) 中的 cpu_usage（括号后+右括号前）
    # - metric1 + metric2 中的 metric1 和 metric2（运算符前后）
    # - rate(http_requests[5m]) 中的 http_requests（括号后+方括号前）
    # 但不会匹配：
    # - avg(...) 中的 avg（后面是左括号，不在匹配列表中）
    # - by (label) 中的 by（后面是空格+左括号）
    PROMQL_METRIC_PATTERN = re.compile(
        r"(?:^|(?<=[\(,\s+\-*/]))([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*[\{\[\)\s,+\-*/]|$)", re.UNICODE
    )

    # PromQL 关键字列表（不应添加前缀）
    PROMQL_KEYWORDS = {
        "by",
        "without",
        "on",
        "ignoring",
        "group_left",
        "group_right",
        "bool",
        "offset",
        "and",
        "or",
        "unless",
        # 聚合函数
        "sum",
        "min",
        "max",
        "avg",
        "stddev",
        "stdvar",
        "count",
        "count_values",
        "bottomk",
        "topk",
        "quantile",
        # 其他函数
        "rate",
        "irate",
        "increase",
        "delta",
        "idelta",
        "deriv",
        "predict_linear",
        "histogram_quantile",
        "label_replace",
        "label_join",
        "abs",
        "ceil",
        "floor",
        "round",
        "exp",
        "ln",
        "log2",
        "log10",
        "sqrt",
        "sort",
        "sort_desc",
        "time",
        "timestamp",
        "vector",
        "scalar",
        "absent",
        "absent_over_time",
        "present_over_time",
        "changes",
        "resets",
        "deriv",
        "holt_winters",
        "hour",
        "minute",
        "month",
        "year",
        "day_of_month",
        "day_of_week",
        "days_in_month",
        "clamp_max",
        "clamp_min",
        # 常见的 label 名称
        "le",
        "quantile",
        "job",
        "instance",
        "label",
    }

    # 匹配 by(...) / without(...) 子句（含括号内容），用于保护 label 名不被添加前缀
    PROMQL_BY_WITHOUT_PATTERN = re.compile(r"(?:by|without)\s*\([^)]*\)", re.IGNORECASE)

    def add_kaiwu_prefix_to_expr(expr: str) -> str:
        """
        对 PromQL 表达式添加 kaiwu_ 前缀，但排除：
        1. PromQL 关键字/函数名
        2. 已有 kaiwu_ 前缀的指标
        3. by(...) / without(...) 子句中的 label 名
        """
        # 先提取并保护 by(...) / without(...) 子句，用不可被指标正则匹配的占位符替换
        placeholders = []

        def save_by_clause(m: re.Match) -> str:
            idx = len(placeholders)
            placeholders.append(m.group(0))
            # 占位符以 \x00 开头，不会被 PROMQL_METRIC_PATTERN（要求 [a-zA-Z_] 开头）匹配
            return f"\x00BYPH{idx}\x00"

        protected_expr = PROMQL_BY_WITHOUT_PATTERN.sub(save_by_clause, expr)

        # 对保护后的表达式执行指标名前缀添加
        def add_prefix(match: re.Match) -> str:
            metric_name = match.group(1)
            if metric_name.lower() in PROMQL_KEYWORDS:
                return metric_name
            if metric_name.startswith("kaiwu_"):
                return metric_name
            return f"kaiwu_{metric_name}"

        result = PROMQL_METRIC_PATTERN.sub(add_prefix, protected_expr)

        # 还原占位符为原始 by(...) / without(...) 子句
        for i, clause in enumerate(placeholders):
            result = result.replace(f"\x00BYPH{i}\x00", clause)

        return result

    try:
        if not os.path.exists(user_file_path):
            logger.warning(
                add_log_prefix(
                    f"User custom monitor file does not exist: {user_file_path}, will skip loading", log_prefix
                )
            )
            return None

        file_dir, file_name = os.path.split(user_file_path)
        module_name = os.path.splitext(file_name)[0]
        if file_dir not in os.sys.path:
            os.sys.path.insert(0, file_dir)
            logger.info(add_log_prefix(f"User config directory added to Python path: {file_dir}", log_prefix))

        user_module = importlib.import_module(module_name)
        if not hasattr(user_module, "build_monitor"):
            logger.warning(
                add_log_prefix(
                    f"build_monitor method not found in user monitor file {user_file_path}, will skip loading",
                    log_prefix,
                )
            )
            return None

        user_build_func = getattr(user_module, "build_monitor")
        user_config = user_build_func()

        if not isinstance(user_config, dict):
            logger.warning(
                add_log_prefix(
                    f"User monitor config return value is invalid (must be dict), actual type: {type(user_config)}, will skip loading",
                    log_prefix,
                )
            )
            return None

        # 遍历并修正所有 metrics 的 expr 字段
        for group in user_config.get("groups", []):
            for panel in group.get("panels", []):
                for metric in panel.get("metrics", []):
                    expr = metric.get("expr", "")
                    if not isinstance(expr, str) or not expr.strip():
                        continue
                    # 执行替换（仅匹配指标名，保护 by/without 子句中的 label）
                    modified_expr = add_kaiwu_prefix_to_expr(expr)
                    if modified_expr != expr:
                        metric["expr"] = modified_expr
                        logger.info(
                            add_log_prefix(
                                f"Modified expr for metric [{metric.get('metrics_name')}]: {modified_expr}", log_prefix
                            )
                        )

        logger.info(
            add_log_prefix(
                f"User custom monitor config loaded successfully (kaiwu_ prefix added): {user_file_path}", log_prefix
            )
        )
        return user_config

    except ImportError as e:
        logger.warning(
            add_log_prefix(
                f"User monitor file import failed (path: {user_file_path}): {str(e)}, will skip loading", log_prefix
            )
        )
        return None
    except Exception as e:
        logger.error(
            add_log_prefix(
                f"Error occurred while loading user monitor config (path: {user_file_path}): {str(e)}, will skip loading",
                log_prefix,
            )
        )
        return None
    finally:
        if "file_dir" in locals() and file_dir in os.sys.path:
            os.sys.path.remove(file_dir)
