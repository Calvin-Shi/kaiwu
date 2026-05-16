#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

王者荣耀 1v1 奖励计算模块。

奖励项 (9 项):
  零和项: tower_hp_point, hp_point, money, exp, kill, death
  特殊项: forward, ep_rate, last_hit

时间衰减: 最终奖励 *= 0.6 ^ (frame_no / TIME_SCALE_ARG)
"""

import math
from agent_ppo.conf.conf import GameConfig


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight


def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1

        self.m_reward_value = {}
        self.m_last_frame_no = -1

        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()

        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}

        self.cached_main_tower_pos = None
        self.cached_enemy_tower_pos = None

        self._prev_enemy_tower_hp_ratio = 1.0

    # ------------------------------------------------------------------
    # 等级经验表
    # ------------------------------------------------------------------
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_get(d, key, default=0):
        if d is None:
            return default
        value = d.get(key, default)
        return default if value is None else value

    def _calc_total_exp(self, hero):
        """计算1级到当前等级的总经验 + 当前等级内经验。满级(>=15)返回0。"""
        if hero is None:
            return 0.0
        level = int(self._safe_get(hero, "level", 1))
        if level >= 15:
            return 0.0
        cur_level_exp = float(self._safe_get(hero, "exp", 0))
        total_exp = 0.0
        for lv in range(1, level):
            total_exp += self.m_each_level_max_exp.get(lv, 0)
        total_exp += cur_level_exp
        return total_exp

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def result(self, frame_data, action=None):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value, action=action)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(
                    0.6, 1.0 * frame_no / self.time_scale_arg
                )

        return self.m_reward_value

    # ------------------------------------------------------------------
    # 帧数据处理
    # ------------------------------------------------------------------
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]

        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # ------------------------------------------------------------------
    # 计算单个阵营的状态量
    # ------------------------------------------------------------------
    def set_cur_calc_frame_vec(self, calc_map, frame_data, camp):
        main_hero, enemy_hero, main_tower, enemy_tower = None, None, None, None

        for hero in frame_data["hero_states"]:
            if hero["camp"] == camp:
                main_hero = hero
            else:
                enemy_hero = hero

        for organ in frame_data["npc_states"]:
            if organ["sub_type"] == 21:
                if organ["camp"] == camp:
                    main_tower = organ
                else:
                    enemy_tower = organ

        hp = float(self._safe_get(main_hero, "hp", 0))
        max_hp = float(self._safe_get(main_hero, "max_hp", 1))
        ep = float(self._safe_get(main_hero, "ep", 0))
        max_ep = float(self._safe_get(main_hero, "max_ep", 1))
        money = float(self._safe_get(main_hero, "money", 0))
        kill_cnt = float(self._safe_get(main_hero, "kill_cnt", 0))
        dead_cnt = float(self._safe_get(main_hero, "dead_cnt", 0))
        total_exp = self._calc_total_exp(main_hero)

        if max_hp <= 0:
            max_hp = 1.0
        if max_ep <= 0:
            max_ep = 1.0

        is_main_side = (camp == self.main_hero_camp)

        for reward_name, reward_struct in calc_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            if reward_name == "tower_hp_point":
                if main_tower is not None and main_tower.get("max_hp", 0):
                    reward_struct.cur_frame_value = (
                        1.0 * main_tower["hp"] / main_tower["max_hp"]
                    )
                else:
                    reward_struct.cur_frame_value = 0.0

            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(
                    main_hero, main_tower, enemy_tower
                )

            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(
                    math.sqrt(hp / max_hp)
                )

            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = ep / max_ep

            elif reward_name == "money":
                reward_struct.cur_frame_value = money

            elif reward_name == "exp":
                reward_struct.cur_frame_value = total_exp

            elif reward_name == "kill":
                reward_struct.cur_frame_value = kill_cnt

            elif reward_name == "death":
                reward_struct.cur_frame_value = dead_cnt

            elif reward_name == "last_hit":
                if is_main_side:
                    reward_struct.cur_frame_value = (
                        self._calc_last_hit_from_dead_action(
                            frame_data, main_hero, enemy_hero
                        )
                    )
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

    # ------------------------------------------------------------------
    # 推进距离
    # ------------------------------------------------------------------
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_hero is None:
            return 0.0

        if main_tower is not None:
            self.cached_main_tower_pos = (
                main_tower["location"]["x"],
                main_tower["location"]["z"],
            )
        if enemy_tower is not None:
            self.cached_enemy_tower_pos = (
                enemy_tower["location"]["x"],
                enemy_tower["location"]["z"],
            )

        if self.cached_main_tower_pos is None or self.cached_enemy_tower_pos is None:
            return 0.0

        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        dist_hero2emy = math.dist(hero_pos, self.cached_enemy_tower_pos)
        dist_main2emy = math.dist(
            self.cached_main_tower_pos, self.cached_enemy_tower_pos
        )

        return (dist_main2emy - dist_hero2emy) / max(dist_main2emy, 1.0)

    # ------------------------------------------------------------------
    # 补刀检测
    # ------------------------------------------------------------------
    def _calc_last_hit_from_dead_action(self, frame_data, main_hero, enemy_hero):
        reward = 0.0
        frame_action = frame_data.get("frame_action", {}) or {}
        dead_actions = frame_action.get("dead_action", []) or []

        main_id = main_hero.get("runtime_id") if main_hero else None
        enemy_id = enemy_hero.get("runtime_id") if enemy_hero else None

        for da in dead_actions:
            killer = da.get("killer") or {}
            victim = da.get("death") or {}

            killer_id = killer.get("runtime_id")
            victim_type = str(victim.get("sub_type", ""))

            if victim_type not in ("ACTOR_SUB_SOLDIER", "11"):
                continue

            if killer_id == main_id:
                reward += 1.0
            elif killer_id == enemy_id:
                reward -= 1.0

        return reward

    # ------------------------------------------------------------------
    # 奖励汇总
    # ------------------------------------------------------------------
    def get_reward(self, frame_data, reward_dict, action=None):
        reward_dict.clear()
        reward_sum = 0.0

        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():

            main_hp = self.m_main_calc_frame_map["hp_point"].cur_frame_value
            enemy_hp = self.m_enemy_calc_frame_map["hp_point"].cur_frame_value

            # ----------------------------------------------------------
            # forward: HP>99% 且 英雄到敌塔距离 > 双方塔距
            # ----------------------------------------------------------
            if reward_name == "forward":
                cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last = self.m_main_calc_frame_map[reward_name].last_frame_value
                forward_delta = cur - last

                # 触发条件：英雄血量 > 99%
                main_hero_for_check = None
                for hero in frame_data.get("hero_states", []):
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_for_check = hero
                        break

                hp_ok = False
                dist_ok = False
                if main_hero_for_check is not None:
                    hp_val = float(self._safe_get(main_hero_for_check, "hp", 0))
                    max_hp_val = float(self._safe_get(main_hero_for_check, "max_hp", 1))
                    if max_hp_val > 0 and (hp_val / max_hp_val) > 0.99:
                        hp_ok = True

                    if self.cached_main_tower_pos and self.cached_enemy_tower_pos:
                        hero_pos = (
                            main_hero_for_check["location"]["x"],
                            main_hero_for_check["location"]["z"],
                        )
                        dist_hero2emy = math.dist(
                            hero_pos, self.cached_enemy_tower_pos
                        )
                        dist_main2emy = math.dist(
                            self.cached_main_tower_pos, self.cached_enemy_tower_pos
                        )
                        if dist_hero2emy > dist_main2emy:
                            dist_ok = True

                if hp_ok and dist_ok:
                    reward_struct.value = forward_delta
                else:
                    reward_struct.value = 0.0

                # 指定帧数后强制清零
                if GameConfig.REMOVE_FORWARD_AFTER is not None:
                    frame_no = frame_data.get("frame_no", 0)
                    reward_struct.value *= (
                        frame_no <= GameConfig.REMOVE_FORWARD_AFTER
                    )

            # ----------------------------------------------------------
            # ep_rate: 仅看己方，仅在法力值增加时给奖励
            # ----------------------------------------------------------
            elif reward_name == "ep_rate":
                cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last = self.m_main_calc_frame_map[reward_name].last_frame_value
                delta = cur - last
                reward_struct.value = delta if delta > 0 else 0.0

            # ----------------------------------------------------------
            # last_hit: 累计值差分
            # ----------------------------------------------------------
            elif reward_name == "last_hit":
                cur = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = cur - last

            # ----------------------------------------------------------
            # 零和差分项: tower_hp_point, hp_point, money, exp, kill, death
            # ----------------------------------------------------------
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = (
                    reward_struct.cur_frame_value - reward_struct.last_frame_value
                )

            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value

        # ================================================================
        # 拆塔即时反馈 + 拆塔里程碑
        # ================================================================
        enemy_tower_hp_ratio = (
            self.m_enemy_calc_frame_map["tower_hp_point"].cur_frame_value
        )
        if self._prev_enemy_tower_hp_ratio > 0:
            tower_damage_ratio = (
                self._prev_enemy_tower_hp_ratio - enemy_tower_hp_ratio
            )

            if tower_damage_ratio > 0:
                hero_near_enemy_tower = False
                main_hero_pos = None
                for hero in frame_data.get("hero_states", []):
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_pos = (
                            hero["location"]["x"],
                            hero["location"]["z"],
                        )
                        break
                if main_hero_pos and self.cached_enemy_tower_pos:
                    dist_to_tower = math.dist(
                        main_hero_pos, self.cached_enemy_tower_pos
                    )
                    hero_near_enemy_tower = dist_to_tower < 15000.0

                if hero_near_enemy_tower:
                    tower_damage_reward = tower_damage_ratio * 3.0
                    reward_dict["tower_damage"] = tower_damage_reward
                    reward_sum += tower_damage_reward

            if enemy_tower_hp_ratio <= 0:
                reward_dict["tower_destroy"] = 10.0
                reward_sum += 10.0

        self._prev_enemy_tower_hp_ratio = enemy_tower_hp_ratio
        reward_dict["reward_sum"] = reward_sum
