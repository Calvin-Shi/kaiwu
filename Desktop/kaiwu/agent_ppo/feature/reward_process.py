#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

王者荣耀 1v1 奖励计算模块（高阶战术强化版）。

设计要点：
1. 继承原版零和博弈 (Zero-Sum) 与时间差分逻辑；
2. 【降维打击】注入独立高阶奖励项：拉扯白嫖 (hp_trade)、补刀爆发 (last_hit)、防发呆惩罚 (anti_camp)。
"""

import math
from agent_ppo.conf.conf import GameConfig


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
        
    # =========================================================
    # 【战术升级】：强行注入高阶微操奖励项
    # =========================================================
    calc_frame_map["hp_trade"] = RewardStruct(3.0)  # 拉扯与白嫖奖励
    calc_frame_map["last_hit"] = RewardStruct(2.0)  # 补刀瞬时刺激
    calc_frame_map["anti_camp"] = RewardStruct(1.0) # 防发呆/站桩惩罚
    calc_frame_map["kiting"] = RewardStruct(1.0)    # 极限拉扯奖励
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1

        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        
        # 记录英雄最近轨迹，用于防发呆判定
        self.pos_window = []

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

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    @staticmethod
    def _safe_get(d, key, default=0):
        if d is None:
            return default
        value = d.get(key, default)
        return default if value is None else value

    def _calc_total_exp(self, hero):
        if hero is None:
            return 0.0
        level = int(self._safe_get(hero, "level", 1))
        cur_level_exp = float(self._safe_get(hero, "exp", 0))
        total_exp = 0.0
        for lv in range(1, max(level, 1)):
            total_exp += self.m_each_level_max_exp.get(lv, 0)
        total_exp += cur_level_exp
        return total_exp

    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        main_hero, main_tower, enemy_tower = None, None, None

        for hero in frame_data["hero_states"]:
            if hero["camp"] == camp:
                main_hero = hero

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

        if max_hp <= 0: max_hp = 1.0
        if max_ep <= 0: max_ep = 1.0

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            if reward_name == "tower_hp_point":
                if main_tower is not None and main_tower.get("max_hp", 0):
                    reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = hp / max_hp
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
            else:
                # 兼容新增的自定义键 (hp_trade等)，占位设为0即可
                reward_struct.cur_frame_value = 0.0

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return 0.0
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        
        if main_hero["max_hp"] > 0 and main_hero["hp"] / main_hero["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
                
                # 更新自身轨迹记录，用于防发呆检测
                hero_x = self._safe_get(hero["location"], "x", 0)
                hero_z = self._safe_get(hero["location"], "z", 0)
                self.pos_window.append((hero_x, hero_z))
                if len(self.pos_window) > 15:
                    self.pos_window.pop(0)
            else:
                enemy_camp = hero["camp"]
                
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():

            if reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "kill":
                cur_main = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last_main = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.cur_frame_value = cur_main
                reward_struct.last_frame_value = last_main
                reward_struct.value = max(cur_main - last_main, 0.0)

            elif reward_name == "death":
                cur_main = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last_main = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.cur_frame_value = cur_main
                reward_struct.last_frame_value = last_main
                reward_struct.value = max(cur_main - last_main, 0.0)

            # =========================================================
            # 【核心战术 1】：极限拉扯与白嫖奖励 (HP Trade Bonus)
            # =========================================================
            elif reward_name == "hp_trade":
                main_hp_delta = (self.m_main_calc_frame_map["hp_point"].cur_frame_value - 
                                 self.m_main_calc_frame_map["hp_point"].last_frame_value)
                enemy_hp_delta = (self.m_enemy_calc_frame_map["hp_point"].cur_frame_value - 
                                  self.m_enemy_calc_frame_map["hp_point"].last_frame_value)

                trade_bonus = 0.0
                if enemy_hp_delta < 0: # 只要打到了敌方
                    if main_hp_delta >= 0:
                        # 神级操作：完美拉扯，白嫖血量！给 1.5 倍爆发奖励
                        trade_bonus = -enemy_hp_delta * 1.5
                    elif enemy_hp_delta < main_hp_delta:
                        # 高阶操作：对点换血，但我方扣得比敌方少
                        trade_bonus = -(enemy_hp_delta - main_hp_delta) * 0.5
                reward_struct.value = trade_bonus

            # =========================================================
            # 【核心战术 2】：补刀/经济跃升瞬间奖励 (Last-hit Bonus)
            # =========================================================
            elif reward_name == "last_hit":
                money_delta = (self.m_main_calc_frame_map["money"].cur_frame_value - 
                               self.m_main_calc_frame_map["money"].last_frame_value)
                hit_bonus = 0.0
                # 自然跳钱大概在1~3，当金币单帧突增大于 15 时，必然是击杀了小兵/英雄/防御塔
                if money_delta > 15.0:
                    hit_bonus = money_delta / 100.0  # 给予强烈的瞬时正反馈
                reward_struct.value = hit_bonus

            # =========================================================
            # 【核心战术 3】：防站桩/怠惰惩罚 (Anti-Camping)
            # =========================================================
            elif reward_name == "anti_camp":
                camp_penalty = 0.0
                if len(self.pos_window) == 15:
                    start_x, start_z = self.pos_window[0]
                    end_x, end_z = self.pos_window[-1]
                    dist = math.sqrt((start_x - end_x)**2 + (start_z - end_z)**2)
                    main_hp_cur = self.m_main_calc_frame_map["hp_point"].cur_frame_value
                    
                    # 惩罚条件：过去15帧(近1秒)没怎么动(距离<1000)，且血量健康(>70%)
                    # 如果残血，可能是躲在塔下回城，予以宽容；如果满血还在发呆，重罚！
                    if dist < 1000 and main_hp_cur > 0.7:
                        camp_penalty = -0.05
                reward_struct.value = camp_penalty
            elif reward_name == "kiting":
                kiting_bonus = 0.0
                main_hero_pos, enemy_hero_pos = None, None
                enemy_hp = 0

                # 从当前帧数据中遍历获取双方位置和敌方血量
                for hero in frame_data["hero_states"]:
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                    else:
                        enemy_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                        enemy_hp = float(self._safe_get(hero, "hp", 0))

                # 必须确认拿到了坐标，且敌方存活才计算奖励
                if main_hero_pos and enemy_hero_pos and enemy_hp > 0:
                    dist_enemy = math.dist(main_hero_pos, enemy_hero_pos)
                    if 6000 <= dist_enemy <= 8500:
                        kiting_bonus = 0.02   # 处于极佳射程，持续给微小正反馈
                    elif dist_enemy < 4000:
                        kiting_bonus = -0.05  # 距离过近，面临被秒杀风险，重罚
                reward_struct.value = kiting_bonus

            # ---------- 基础状态零和差分 ----------
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
            
        reward_dict["reward_sum"] = reward_sum