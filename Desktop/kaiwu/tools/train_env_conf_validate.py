#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Train environment configuration validator
训练环境配置校验器
"""


import toml
import json
import os


# Default valid hero IDs (used when no validate_rule.json is found)
# 默认有效的英雄ID（当没有 validate_rule.json 时使用）
DEFAULT_VALID_HERO_IDS = [112]

# Valid opponent agent types
# 有效的对手类型
VALID_OPPONENT_TYPES = ["selfplay", "common_ai"]


def load_valid_hero_ids():
    """
    Load VALID_HERO_IDS from validate_rule.json in the same directory as this script.
    If the file doesn't exist, return the default value.
    从本脚本同目录下的 validate_rule.json 加载 VALID_HERO_IDS。
    如果文件不存在，返回默认值。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rule_file = os.path.join(script_dir, "validate_rule.json")
    if os.path.exists(rule_file):
        with open(rule_file, "r") as f:
            rules = json.load(f)
        return rules.get("valid_hero_ids", DEFAULT_VALID_HERO_IDS)
    return DEFAULT_VALID_HERO_IDS


def validate_monitor_config(monitor_conf, logger):
    """
    Validate monitor configuration parameters
    校验监控配置参数

    Args:
        monitor_conf: Monitor configuration dictionary
                      监控配置字典
        logger: Logger instance
                日志实例

    Raises:
        ValueError: If validation fails
                    如果校验失败
    """
    # Validate monitor_side
    # 校验 monitor_side
    if "monitor_side" not in monitor_conf:
        raise ValueError("Missing required parameter: monitor.monitor_side")

    monitor_side = monitor_conf["monitor_side"]
    if type(monitor_side) != int:
        raise ValueError(f"monitor.monitor_side must be an integer, got {type(monitor_side).__name__}")

    if monitor_side not in [0, 1]:
        raise ValueError(f"monitor.monitor_side must be 0 or 1, got {monitor_side}")

    logger.info(f"✓ monitor.monitor_side validation passed: {monitor_side}")

    # Validate auto_switch_monitor_side
    # 校验 auto_switch_monitor_side
    if "auto_switch_monitor_side" not in monitor_conf:
        raise ValueError("Missing required parameter: monitor.auto_switch_monitor_side")

    auto_switch = monitor_conf["auto_switch_monitor_side"]
    if not isinstance(auto_switch, bool):
        raise ValueError(f"monitor.auto_switch_monitor_side must be a boolean, got {type(auto_switch).__name__}")

    logger.info(f"✓ monitor.auto_switch_monitor_side validation passed: {auto_switch}")


def validate_episode_config(episode_conf, logger):
    """
    Validate episode configuration parameters
    校验对局配置参数

    Args:
        episode_conf: Episode configuration dictionary
                      对局配置字典
        logger: Logger instance
                日志实例

    Raises:
        ValueError: If validation fails
                    如果校验失败
    """
    # Validate opponent_agent
    # 校验 opponent_agent
    if "opponent_agent" not in episode_conf:
        raise ValueError("Missing required parameter: episode.opponent_agent")

    opponent_agent = episode_conf["opponent_agent"]
    if not isinstance(opponent_agent, str):
        raise ValueError(f"episode.opponent_agent must be a string, got {type(opponent_agent).__name__}")

    # Check if it's a valid predefined type or a custom model ID
    # 检查是否是预定义类型或自定义模型ID
    if opponent_agent not in VALID_OPPONENT_TYPES and not opponent_agent.strip():
        raise ValueError(
            f"episode.opponent_agent must be one of {VALID_OPPONENT_TYPES} or a valid custom model ID, "
            f"got '{opponent_agent}'"
        )

    logger.info(f"✓ episode.opponent_agent validation passed: {opponent_agent}")

    # Validate eval_interval
    # 校验 eval_interval
    if "eval_interval" not in episode_conf:
        raise ValueError("Missing required parameter: episode.eval_interval")

    eval_interval = episode_conf["eval_interval"]
    if type(eval_interval) != int:
        raise ValueError(f"episode.eval_interval must be an integer, got {type(eval_interval).__name__}")

    if eval_interval < 2:
        raise ValueError(f"episode.eval_interval must be >= 2, got {eval_interval}")

    logger.info(f"✓ episode.eval_interval validation passed: {eval_interval}")

    # Validate eval_opponent_type
    # 校验 eval_opponent_type
    if "eval_opponent_type" not in episode_conf:
        raise ValueError("Missing required parameter: episode.eval_opponent_type")

    eval_opponent_type = episode_conf["eval_opponent_type"]
    if not isinstance(eval_opponent_type, str):
        raise ValueError(f"episode.eval_opponent_type must be a string, got {type(eval_opponent_type).__name__}")

    # Check if it's a valid predefined type or a custom model ID
    # 检查是否是预定义类型或自定义模型ID
    if eval_opponent_type not in VALID_OPPONENT_TYPES and not eval_opponent_type.strip():
        raise ValueError(
            f"episode.eval_opponent_type must be one of {VALID_OPPONENT_TYPES} or a valid custom model ID, "
            f"got '{eval_opponent_type}'"
        )

    logger.info(f"✓ episode.eval_opponent_type validation passed: {eval_opponent_type}")


def validate_lineups_config(lineups_conf, logger, valid_hero_ids=None):
    """
    Validate lineups configuration parameters
    校验阵容配置参数

    Args:
        lineups_conf: Lineups configuration dictionary
                      阵容配置字典
        logger: Logger instance
                日志实例
        valid_hero_ids: List of valid hero IDs (loaded from validate_rule.json or default)
                        有效的英雄ID列表（从 validate_rule.json 加载或使用默认值）

    Raises:
        ValueError: If validation fails
                    如果校验失败
    """
    if valid_hero_ids is None:
        valid_hero_ids = DEFAULT_VALID_HERO_IDS

    # Validate blue_camp lineup
    # 校验蓝方阵容
    if "blue_camp" not in lineups_conf:
        raise ValueError("Missing required parameter: lineups.blue_camp")

    blue_camp = lineups_conf["blue_camp"]
    if not isinstance(blue_camp, list) or len(blue_camp) == 0:
        raise ValueError("lineups.blue_camp must be a non-empty list")

    if "hero_id" not in blue_camp[0]:
        raise ValueError("Missing required parameter: lineups.blue_camp[0].hero_id")

    blue_hero_id = blue_camp[0]["hero_id"]
    if type(blue_hero_id) != int:
        raise ValueError(f"lineups.blue_camp[0].hero_id must be an integer, got {type(blue_hero_id).__name__}")

    if blue_hero_id not in valid_hero_ids:
        raise ValueError(f"lineups.blue_camp[0].hero_id must be one of {valid_hero_ids}, got {blue_hero_id}")

    logger.info(f"✓ lineups.blue_camp[0].hero_id validation passed: {blue_hero_id}")

    # Validate red_camp lineup
    # 校验红方阵容
    if "red_camp" not in lineups_conf:
        raise ValueError("Missing required parameter: lineups.red_camp")

    red_camp = lineups_conf["red_camp"]
    if not isinstance(red_camp, list) or len(red_camp) == 0:
        raise ValueError("lineups.red_camp must be a non-empty list")

    if "hero_id" not in red_camp[0]:
        raise ValueError("Missing required parameter: lineups.red_camp[0].hero_id")

    red_hero_id = red_camp[0]["hero_id"]
    if type(red_hero_id) != int:
        raise ValueError(f"lineups.red_camp[0].hero_id must be an integer, got {type(red_hero_id).__name__}")

    if red_hero_id not in valid_hero_ids:
        raise ValueError(f"lineups.red_camp[0].hero_id must be one of {valid_hero_ids}, got {red_hero_id}")

    logger.info(f"✓ lineups.red_camp[0].hero_id validation passed: {red_hero_id}")


def read_usr_conf(config_path, logger):
    """
    Read and validate user configuration file
    读取并校验用户配置文件

    Args:
        config_path: Path to the TOML configuration file
                     TOML配置文件路径
        logger: Logger instance
                日志实例

    Returns:
        dict: Validated configuration dictionary
              已校验的配置字典
        None: If validation fails
              如果校验失败

    Raises:
        ValueError: If any validation check fails
                    如果任何校验检查失败
    """
    try:
        # Load TOML configuration file
        # 加载TOML配置文件
        logger.info(f"Loading configuration from: {config_path}")
        usr_conf = toml.load(config_path)
        logger.info("Configuration file loaded successfully")

        # Validate monitor configuration
        # 校验监控配置
        logger.info("Validating monitor configuration...")
        if "monitor" not in usr_conf:
            raise ValueError("Missing required section: [monitor]")
        validate_monitor_config(usr_conf["monitor"], logger)

        # Validate episode configuration
        # 校验对局配置
        logger.info("Validating episode configuration...")
        if "episode" not in usr_conf:
            raise ValueError("Missing required section: [episode]")
        validate_episode_config(usr_conf["episode"], logger)

        # Validate lineups configuration
        # 校验阵容配置
        logger.info("Validating lineups configuration...")
        if "lineups" not in usr_conf:
            raise ValueError("Missing required section: [lineups]")
        valid_hero_ids = load_valid_hero_ids()
        logger.info(f"Valid hero IDs for this profile: {valid_hero_ids}")
        validate_lineups_config(usr_conf["lineups"], logger, valid_hero_ids)

        logger.info("=" * 60)
        logger.info("All configuration validations passed successfully!")
        logger.info("=" * 60)

        return usr_conf

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise ValueError(f"Configuration file not found: {config_path}")
    except toml.TomlDecodeError as e:
        logger.error(f"Failed to parse TOML file: {e}")
        raise ValueError(f"Invalid TOML format: {e}")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading configuration: {e}")
        raise ValueError(f"Failed to read configuration: {e}")
