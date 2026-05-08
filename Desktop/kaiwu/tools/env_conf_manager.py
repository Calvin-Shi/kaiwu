#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import random
from tools.train_env_conf_validate import read_usr_conf


# Default summoner skill ID (Flash / 闪现)
# 默认召唤师技能ID（闪现）
DEFAULT_SKILL = 80115


class EnvConfManager:
    def __init__(self, config_path, logger):
        self.config_path = config_path
        self.logger = logger
        self.usr_conf = None
        self.episode_cnt = 0
        self.eval_interval = 0
        self.random_eval_start = 0
        self.default_opponent_agent = None
        self.auto_switch_monitor_side = False
        self.monitor_side = 0
        self.initialize()

    def initialize(self):
        # Load configuration file
        # 读取并验证配置文件
        self.usr_conf = read_usr_conf(self.config_path, self.logger)
        if self.usr_conf is None:
            raise ValueError("usr_conf is None, please check the configuration file")

        # Get evaluation interval and default opponent type
        # 获取评估间隔和默认对手类型
        self.eval_interval = self.usr_conf["episode"]["eval_interval"] + 1
        self.default_opponent_agent = self.usr_conf["episode"]["opponent_agent"]

        # Determine whether to automatically switch the training agent's side
        # 确定是否自动切换训练智能体的阵营
        self.auto_switch_monitor_side = self.usr_conf["monitor"]["auto_switch_monitor_side"]
        self.monitor_side = self.usr_conf["monitor"]["monitor_side"]

        # Randomly initialize the random evaluation start point
        # 初始化随机评估起始点
        if self.eval_interval != 0:
            self.random_eval_start = random.randint(0, self.eval_interval)
        else:
            self.random_eval_start = 0

    def get_current_config(self):
        # Get the current episode configuration
        # 获取当前对局的配置
        return self.usr_conf

    def get_monitor_side(self):
        # Get the current monitor side
        # 获取当前对局的上报阵营
        return self.monitor_side

    def get_opponent_agent(self):
        # Get the current opponent agent
        # 获取当前对局的对手类型
        return self.usr_conf["episode"]["opponent_agent"]

    def update_config(self, lineup=None):
        # Update the configuration
        # 更新对局配置，包括阵容和训练智能体的ID
        if lineup:
            # 确保阵容是有效的英雄ID列表
            if len(lineup) == 2 and all(type(hero_id) == int for hero_id in lineup[:2]):
                self.usr_conf["lineups"]["blue_camp"][0]["hero_id"] = lineup[0]
                self.usr_conf["lineups"]["red_camp"][0]["hero_id"] = lineup[1]
            else:
                raise ValueError("Invalid lineup format, expected list of 2 integers")

        # Determine whether to switch the monitor side
        # 确定是否自动切换上报监控阵营
        if self.auto_switch_monitor_side:
            self.monitor_side = 1 - self.monitor_side
        self.usr_conf["monitor"]["monitor_side"] = self.monitor_side

        # Determine whether to evaluate
        # 确定是否进行评估
        is_eval = (self.episode_cnt + self.random_eval_start) % self.eval_interval == 0
        opponent_agent = self.default_opponent_agent if not is_eval else self.usr_conf["episode"]["eval_opponent_type"]
        self.usr_conf["episode"]["opponent_agent"] = opponent_agent

        self.episode_cnt += 1

        return self.get_current_config(), is_eval, self.get_monitor_side()

    # ------------------------------------------------------------------
    # Hero ID extraction helpers
    # 英雄ID提取辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def extract_hero_ids_from_usr_conf(usr_conf):
        """Extract hero IDs from usr_conf for both camps (training format).
        从训练格式的 usr_conf 中提取双方阵营的英雄ID列表。

        :param usr_conf: The environment configuration dict with "lineups" containing
            "blue_camp" and "red_camp".
        :returns: (blue_hero_ids, red_hero_ids) - lists of hero IDs.
        """
        blue_camp = usr_conf["lineups"]["blue_camp"]
        red_camp = usr_conf["lineups"]["red_camp"]
        blue_hero_ids = [hero["hero_id"] for hero in blue_camp]
        red_hero_ids = [hero["hero_id"] for hero in red_camp]
        return blue_hero_ids, red_hero_ids

    # ------------------------------------------------------------------
    # Select skill injection helpers
    # 召唤师技能注入辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def inject_select_skills(usr_conf, camp_key, select_skills):
        """Inject select_skill into usr_conf lineup for a given camp (training format).
        将 init_config 返回的召唤师技能注入训练格式的 usr_conf 阵容配置中。

        If select_skills is None/empty, injects default skill (80115 Flash) for all heroes.
        如果 select_skills 为空，则为该阵营所有英雄注入默认技能（80115 闪现）。

        :param usr_conf: The environment configuration dict.
        :param camp_key: "blue_camp" or "red_camp".
        :param select_skills: dict mapping hero_id to skill ID, e.g. {112: 80115}.
            Can be None, in which case default skill 80115 is used.
        """
        camp_lineup = usr_conf["lineups"][camp_key]
        if not select_skills:
            # Use default skill for all heroes in this camp
            # 该阵营所有英雄使用默认技能
            for hero in camp_lineup:
                hero["select_skill"] = DEFAULT_SKILL
        else:
            for hero in camp_lineup:
                hero_id = hero["hero_id"]
                hero["select_skill"] = select_skills.get(hero_id, DEFAULT_SKILL)
