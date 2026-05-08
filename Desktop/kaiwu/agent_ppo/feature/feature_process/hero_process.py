#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

王者荣耀 1v1 英雄特征处理模块（扩充版）。

本模块在基线代码基础上，新增下列英雄向量特征：
- hp_rate          生命值比例（hp / max_hp）
- ep_rate          法力/能量比例（ep / max_ep）
- level            英雄等级
- money            英雄金币
- skill_cooldown   5 个技能槽位的冷却比例（普攻 + 3 主动技 + 1 召唤师技能）

所有特征的归一化方式在 hero_feature_config.ini 里声明，本模块只负责
按 AIFrameState 协议解析原始数值。每个特征提取函数的签名保持
`(self, hero, vector_feature, feature_name)` 不变，由框架统一调度归一化。
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os


# 1 普攻 + 3 主动技 + 1 召唤师技能
# 注意：AIFrameState 不同版本里槽位可能从 0 或 1 开始编号，
# 这里只依赖 slot_states 的顺序，取前 SKILL_SLOT_NUM 个槽位
SKILL_SLOT_NUM = 5


class HeroProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000

        # 每个英雄输出的标量特征数量，必须与 ini 配置 + 各 extraction 函数吐出的值数对齐：
        #   is_hero_alive(1) + location_x(1) + location_z(1) +
        #   hp_rate(1) + ep_rate(1) + level(1) + money(1) +
        #   skill_cooldown(SKILL_SLOT_NUM)
        # = 7 + SKILL_SLOT_NUM
        self.one_unit_feature_num = 8 + SKILL_SLOT_NUM

        self.unit_buff_num = 1

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path, encoding="utf-8")

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):
        self.generate_hero_info_list(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")

        return main_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        # 当英雄缺失时，用 0 把剩余位置填平
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    # ------------------------------------------------------------------
    # 工具函数：安全取字段
    # AIFrameState 里某些字段在开局或单位死亡瞬间可能缺失/为 None
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_get(d, key, default=0):
        if d is None:
            return default
        value = d.get(key, default)
        return default if value is None else value

    # ==================================================================
    # 基础特征（保留原有实现）
    # ==================================================================
    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        # 让 PLAYERCAMP_2 的视角对齐 PLAYERCAMP_1，等价于坐标取反
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    # ==================================================================
    # 新增：生命值 / 法力值比例
    # ==================================================================
    def get_hp_rate(self, hero, vector_feature, feature_name):
        # hp_rate = hp / max_hp，归一到 [0,1]
        hp = float(self._safe_get(hero, "hp", 0))
        max_hp = float(self._safe_get(hero, "max_hp", 0))
        value = 0.0 if max_hp <= 0 else hp / max_hp
        # 兜底裁剪，避免协议异常导致越界
        value = max(0.0, min(1.0, value))
        vector_feature.append(value)

    def get_ep_rate(self, hero, vector_feature, feature_name):
        # ep_rate = ep / max_ep；无蓝英雄（战士/坦克）max_ep 可能为 0，直接记 0
        ep = float(self._safe_get(hero, "ep", 0))
        max_ep = float(self._safe_get(hero, "max_ep", 0))
        value = 0.0 if max_ep <= 0 else ep / max_ep
        value = max(0.0, min(1.0, value))
        vector_feature.append(value)

    # ==================================================================
    # 新增：等级 / 金币
    # ==================================================================
    def get_level(self, hero, vector_feature, feature_name):
        # 等级为 1~15 整数，送入 min_max:1:15 归一
        value = float(self._safe_get(hero, "level", 1))
        vector_feature.append(value)

    def get_money(self, hero, vector_feature, feature_name):
        # 金币为累计值，送入 min_max:0:20000 归一
        # （1v1 常规局经济上限大致在 2w 以内，超过即饱和为 1）
        value = float(self._safe_get(hero, "money", 0))
        vector_feature.append(value)

    # ==================================================================
    # 新增：技能冷却比例
    # ==================================================================
    def get_skill_cooldown(self, hero, vector_feature, feature_name):
        """
        提取 5 个技能槽的冷却比例：cooldown / cooldown_max ∈ [0,1]。
        语义：0 表示技能就绪，1 表示刚放完正处于满 CD。

        兼容两种协议布局：
        - hero["skill_state"]["slot_states"]   （常见）
        - hero["slot_states"]                  （部分旧版）

        若某个槽位缺失、cooldown_max 为 0，或解析失败，则填 0（视为就绪）。
        无论如何，一定 append 恰好 SKILL_SLOT_NUM 个值，保证维度稳定。
        """
        slot_states = self._extract_slot_states(hero)

        for i in range(SKILL_SLOT_NUM):
            value = 0.0
            if i < len(slot_states):
                slot = slot_states[i] or {}
                cooldown = float(self._safe_get(slot, "cooldown", 0))
                cooldown_max = float(self._safe_get(slot, "cooldown_max", 0))
                if cooldown_max > 0:
                    value = cooldown / cooldown_max
                    value = max(0.0, min(1.0, value))
            vector_feature.append(value)

    @staticmethod
    def _extract_slot_states(hero):
        """
        从 hero 中鲁棒地拿到技能槽列表。若找不到则返回空列表。
        """
        if hero is None:
            return []
        # 首选：skill_state.slot_states
        skill_state = hero.get("skill_state")
        if isinstance(skill_state, dict):
            slots = skill_state.get("slot_states")
            if isinstance(slots, list):
                return slots
        # 退路：hero 顶层 slot_states
        slots = hero.get("slot_states")
        if isinstance(slots, list):
            return slots
        return []
    def get_auto_attack_available(self, hero, vector_feature, feature_name):
        slot_states = self._extract_slot_states(hero)
        value = 0.0
        if len(slot_states) > 0:
            slot = slot_states[0] or {} # 0号槽位通常是普攻
            cooldown = float(self._safe_get(slot, "cooldown", 0))
            if cooldown <= 0:
                value = 1.0 # 普攻完全就绪
        vector_feature.append(value)
