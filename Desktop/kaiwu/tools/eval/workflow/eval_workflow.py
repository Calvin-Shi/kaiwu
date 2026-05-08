#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import signal
import sys
import json
from common_python.config.config_control import CONFIG
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

# 评估对局目录，通过环境变量 EVAL_DIR 注入，避免硬编码路径
EVAL_DIR = os.environ.get("EVAL_DIR", "/workspace/log/eval_game")
LOCAL_LINEUPS_DIR = "tools/eval/conf"


def _write_workflow_marker(file_id: int, filename: str, content: str = "") -> None:
    """写入 workflow 标记文件到对局目录。

    :param file_id: 对局序号
    :param filename: 标记文件名（workflow_done / workflow_error）
    :param content: 写入内容
    """
    if not file_id:
        return
    marker_dir = os.path.join(EVAL_DIR, "games", str(file_id))
    os.makedirs(marker_dir, exist_ok=True)
    marker_path = os.path.join(marker_dir, filename)
    with open(marker_path, "w") as f:
        f.write(content)


def write_workflow_done(file_id: int) -> None:
    """标记 eval_workflow 正常完成。"""
    _write_workflow_marker(file_id, "workflow_done", "done")


def write_workflow_error(file_id: int, reason: str) -> None:
    """标记 eval_workflow 异常退出。"""
    _write_workflow_marker(file_id, "workflow_error", reason)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]
    file_id = int(CONFIG.game_index)

    # 评估开始
    logger.info(".......... Evaluation Start ..........")

    try:
        run_episodes(env, agents, logger, monitor)
        write_workflow_done(file_id)
    except SystemExit:
        # load_game_info 中 sys.exit(1) 触发
        write_workflow_error(file_id, "对局配置加载失败")
        raise
    except Exception as e:
        write_workflow_error(file_id, f"评估流程异常: {e}")
        raise

    # 评估结束
    logger.info(".......... Evaluation End ..........")

    # 评估模式下 workflow 完成后，框架不会主动退出（多进程事件循环继续运行）
    # workflow 运行在子进程中，os._exit 只退出当前子进程，主进程和其他子进程不受影响
    # 需要先给主进程发 SIGTERM，让整个进程树退出
    logger.info("eval workflow finished, terminating aisrv main process")
    ppid = os.getppid()
    try:
        os.kill(ppid, signal.SIGTERM)
        logger.info(f"sent SIGTERM to parent process {ppid}")
    except ProcessLookupError:
        pass
    os._exit(0)


def load_game_info(file_id: int, logger) -> dict:
    """加载对局阵容配置。

    集群模式（file_id != 0）从 EVAL_DIR/games/<file_id>/lineup.json 读取，
    单机模式（file_id == 0）从本地 tools/eval/conf/0.json 读取。

    :param file_id: 对局序号，来自 CONFIG.game_index
    :param logger: 日志记录器
    :returns: 对局配置字典
    """
    if not file_id:
        json_file = f"{LOCAL_LINEUPS_DIR}/{file_id}.json"
    else:
        json_file = os.path.join(EVAL_DIR, "games", str(file_id), "lineup.json")

    try:
        with open(json_file, "r") as f:
            game_info = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {json_file}: {e}")
        sys.exit(1)

    logger.info(f"loaded lineup from {json_file}")
    return game_info


def run_episodes(env, agents, logger, monitor):
    # Default summoner skill ID (Flash / 闪现)
    # 默认召唤师技能ID（闪现）
    DEFAULT_SKILL = 80115

    AGENT_NUM = len(agents)
    episode_cnt = 1

    file_id = int(CONFIG.game_index)
    game_info = load_game_info(file_id, logger)

    heroes_config = game_info["lineups"]
    main_agent = game_info["main_agent"]

    # Call init_config on agents before constructing usr_conf
    # 在构造 usr_conf 前调用 agent.init_config 获取召唤师技能选择
    for camp_index in range(AGENT_NUM):
        agent_index = camp_index if main_agent == 0 else 1 - camp_index
        opponent_index = 1 - camp_index

        my_heroes = [h["hero_id"] for h in heroes_config[camp_index]["lineup"]]
        opponent_heroes = [h["hero_id"] for h in heroes_config[opponent_index]["lineup"]]

        config_data = {
            "my_camp": camp_index,
            "my_heroes": my_heroes,
            "opponent_heroes": opponent_heroes,
        }

        try:
            select_skills = agents[agent_index].init_config(config_data)
        except Exception as e:
            logger.warning(f"Agent[{agent_index}] init_config failed: {e}, using default skill")
            select_skills = None

        # Inject select_skill into lineups; fallback to default if select_skills is None
        # 将召唤师技能注入阵容配置，失败时回退到默认技能
        for hero in heroes_config[camp_index]["lineup"]:
            if select_skills:
                hero["select_skill"] = select_skills.get(hero["hero_id"], DEFAULT_SKILL)
            else:
                hero["select_skill"] = DEFAULT_SKILL

        logger.info(
            f"Agent[{agent_index}] init_config: camp_index={camp_index}, select_skills={select_skills}"
        )

    # 游戏启动配置
    usr_conf = {
        "monitor": {
            # 上报对局指标的阵营
            "monitor_side": main_agent,
        },
        "episode": {
            # 上报对局指标的标签： 自对弈 - "selfplay", common_ai - "common_ai", 对手模型 - model_id
            "opponent_agent": "battle",
        },
        # 表示双方使用的阵容
        "lineups": heroes_config,
        "game_id": game_info["game_id"],
    }

    done = False

    logger.info(f"Episode {episode_cnt} start , usr_conf is {usr_conf}, pid is {os.getpid()}")

    # 开始新对局
    env_obs = env.reset(usr_conf=usr_conf)
    if handle_disaster_recovery(env_obs, logger):
        logger.error(f"Error occurred at episode {episode_cnt} reset")
        sys.exit(1)

    observation = env_obs["observation"]
    extra_info = env_obs["extra_info"]

    # 重置agent
    try:
        for camp_index in range(AGENT_NUM):
            if main_agent == 0:
                agent_index = camp_index
            else:
                agent_index = 1 - camp_index
            agents[agent_index].reset(observation[str(camp_index)])
    except Exception as e:
        logger.error(f"Error occurred at episode {episode_cnt}, error is: {str(e)}")
        raise RuntimeError(f"模型重置异常: {e}")

    terminated = False
    truncated = False
    frame_no = 0
    step = 0
    agent_index = 0

    error_reason = ""

    while not done:
        actions = [
            None,
        ] * AGENT_NUM
        try:
            for camp_index in range(AGENT_NUM):
                if main_agent == 0:
                    agent_index = camp_index
                else:
                    agent_index = 1 - camp_index
                d_action = agents[agent_index].exploit(observation[str(camp_index)])
                actions[camp_index] = d_action
        except Exception as e:
            error_reason = f"模型推理异常: {e}"
            logger.error(f"Episode {episode_cnt} Step {step} Exploit Error: {str(e)}")
            break
        try:
            env_reward, env_obs = env.step(actions)
            if handle_disaster_recovery(env_obs, logger):
                error_reason = f"环境通信异常 (step {step})"
                logger.error(f"Episode {episode_cnt} Step {step} Env Error")
                break
        except Exception as e:
            error_reason = f"环境执行异常: {e}"
            logger.error(f"Episode {episode_cnt} Step {step} Step Error: {str(e)}")
            break
        step += 1

        frame_no = env_obs["frame_no"]
        observation = env_obs["observation"]
        extra_info = env_obs["extra_info"]
        terminated = env_obs["terminated"]
        truncated = env_obs["truncated"]

        # 正常结束或超时退出
        done = terminated or truncated

    if done:
        logger.info(
            f"Episode {episode_cnt} terminated, terminated is {terminated}, truncated is {truncated}, frame_no is {frame_no}."
        )
    elif error_reason:
        raise RuntimeError(error_reason)
