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
import math
import os


# 1 普攻 + 3 主动技 + 1 召唤师技能
# 注意：AIFrameState 不同版本里槽位可能从 0 或 1 开始编号，
# 这里只依赖 slot_states 的顺序，取前 SKILL_SLOT_NUM 个槽位
SKILL_SLOT_NUM = 5

# Buff config_id 分组
BUFF_HEAL_IDS = {10000, 10010, 10014, 90015}       # 回复/治疗相关
BUFF_SPEED_IDS = {11001, 90015, 90025}              # 加速
BUFF_SLOW_IDS = {11002}                             # 减速/被控
BUFF_CLEANSE_IDS = {11010, 911290}                  # 净化/免控

def _hero_has_buff(hero, id_set):
    buffs = hero.get("buff_state", []) or []
    if not isinstance(buffs, list):
        return False
    for b in buffs:
        bid = b.get("config_id", 0) if isinstance(b, dict) else b
        if bid and int(bid) in id_set:
            return True
    return False

CAKE_LOCATIONS_BY_CAMP = {
    "PLAYERCAMP_1": {"main": (-15220, -15120), "enemy": (15340, 15100)},
    "PLAYERCAMP_2": {"main": (15340, 15100), "enemy": (-15220, -15120)},
}


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

        # 每个英雄输出的标量特征数量：
        #   is_hero_alive(1) + location_x(1) + location_z(1) +
        #   hp_rate(1) + ep_rate(1) + level(1) + money(1) +
        #   auto_attack_available(1) +
        #   skill_cooldown(5×4: cd_ratio, usable, hit_rate, use_rate_recent) +
        #   rel_to_opponent(3) + is_in_enemy_tower_range(1) +
        #   buff_features(5: heal, speed, debuff, cleanse, hero_passive) +
        #   cake_dist_features(2: dist_to_main_cake, dist_to_enemy_cake)
        # = 8 + 5*4 + 3 + 1 + 5 + 2 = 39
        self.one_unit_feature_num = 8 + 4 * SKILL_SLOT_NUM + 3 + 1 + 5 + 2

        # 缓存敌方/我方防御塔位置，用于越塔预警
        self.main_tower_pos = None
        self.enemy_tower_pos = None

        self.unit_buff_num = 1

        # 技能使用频率 EMA 跟踪
        self.skill_use_ema = {}
        self.skill_last_used = {}

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
        self._cache_tower_positions(frame_state)

        # Generate hero features for both camps
        # 生成双方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")

        return main_camp_hero_vector_feature + enemy_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

    def _cache_tower_positions(self, frame_state):
        """
        从 npc_states 中提取我方/敌方防御塔位置，并统一到 camp1 坐标系。
        防御塔的 sub_type == 21；若塔已被摧毁（hp<=0 或坐标为 100000）则置 None。
        """
        self.main_tower_pos = None
        self.enemy_tower_pos = None

        for npc in frame_state.get("npc_states", []) or []:
            if npc.get("sub_type") != 21:
                continue
            is_main = (npc["camp"] == self.main_camp)
            hp = self._safe_get(npc, "hp", 0)

            # 防御塔被推掉后坐标通常变为 100000，直接判定为无效
            loc = npc.get("location", {}) or {}
            x = self._safe_get(loc, "x", 0)
            z = self._safe_get(loc, "z", 0)
            if hp <= 0 or x > 90000 or z > 90000:
                continue

            # 镜像翻转：将 camp2 的坐标统一到 camp1 视角
            if self.transform_camp2_to_camp1 != (npc["camp"] == self.main_camp):
                x = -x
                z = -z

            pos = (float(x), float(z))
            if is_main:
                self.main_tower_pos = pos
            else:
                self.enemy_tower_pos = pos

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
        提取 5 个技能槽的 4 维特征（每槽共 20 维）：
          - cd_ratio:      冷却比例 cooldown / cooldown_max ∈ [0,1]
          - usable:        技能是否可用 {0,1}
          - hit_rate:      累计命中率 hitHeroTimes / usedTimes ∈ [0,1]
          - use_rate_recent: 近期使用频率 EMA (指数移动平均)
        """
        slot_states = self._extract_slot_states(hero)

        for i in range(SKILL_SLOT_NUM):
            if i < len(slot_states) and slot_states[i]:
                slot = slot_states[i]
                # 1) cd_ratio
                cd = float(self._safe_get(slot, "cooldown", 0))
                cd_max = float(self._safe_get(slot, "cooldown_max", 0))
                cd_ratio = (cd / cd_max) if cd_max > 0 else 0.0
                cd_ratio = max(0.0, min(1.0, cd_ratio))
                vector_feature.append(cd_ratio)

                # 2) usable
                usable = float(self._safe_get(slot, "usable", 0))
                vector_feature.append(usable)

                # 3) hit_rate
                used = int(self._safe_get(slot, "usedTimes", 0))
                hit = int(self._safe_get(slot, "hitHeroTimes", 0))
                hit_rate = (hit / used) if used > 0 else 0.0
                hit_rate = max(0.0, min(1.0, hit_rate))
                vector_feature.append(hit_rate)

                # 4) use_rate_recent (EMA)
                use_rate = self._update_use_rate_recent(i, slot, alpha=0.1)
                vector_feature.append(use_rate)
            else:
                vector_feature.extend([0.0, 0.0, 0.0, 0.0])

    @staticmethod
    def _extract_slot_states(hero):
        if hero is None:
            return []
        skill_state = hero.get("skill_state")
        if isinstance(skill_state, dict):
            slots = skill_state.get("slot_states")
            if isinstance(slots, list):
                return slots
        slots = hero.get("slot_states")
        if isinstance(slots, list):
            return slots
        return []

    def _update_use_rate_recent(self, raw_idx, slot, alpha=0.1):
        used_times = int(slot.get("usedTimes", 0) or 0)
        prev_used = self.skill_last_used.get(raw_idx, used_times)
        fired = 1.0 if used_times > prev_used else 0.0
        ema_prev = self.skill_use_ema.get(raw_idx, 0.0)
        ema_new = alpha * fired + (1 - alpha) * ema_prev
        self.skill_use_ema[raw_idx] = ema_new
        self.skill_last_used[raw_idx] = used_times
        return ema_new
    def get_auto_attack_available(self, hero, vector_feature, feature_name):
        slot_states = self._extract_slot_states(hero)
        value = 0.0
        if len(slot_states) > 0:
            slot = slot_states[0] or {} # 0号槽位通常是普攻
            cooldown = float(self._safe_get(slot, "cooldown", 0))
            if cooldown <= 0:
                value = 1.0 # 普攻完全就绪
        vector_feature.append(value)

    # ==================================================================
    # 拉扯特征：相对敌方英雄的位置 (dx, dz, r)，归一化到 [0,1]
    #
    # 作用：让 PPO 感知与对手的相对方位和距离，从而学习"极限射程拉扯"——
    #       呆在射手最大射程边缘输出，避免过度近身被秒。
    # 输出维度：3（dx_norm, dz_norm, r_norm）
    # ==================================================================
    def get_rel_to_opponent(self, hero, vector_feature, feature_name):
        # 根据英雄阵营选择对手来源：我方→找敌方，敌方→找我方
        hero_camp = hero.get("camp")
        if hero_camp == self.main_camp:
            opponent_dict = self.enemy_camp_hero_dict
        else:
            opponent_dict = self.main_camp_hero_dict

        enemy = None
        for e in opponent_dict.values():
            enemy = e
            break

        if enemy is None or hero is None:
            vector_feature.extend([0.0, 0.0, 0.0])
            return

        if self._safe_get(enemy, "hp", 0) <= 0:
            vector_feature.extend([0.0, 0.0, 0.0])
            return

        my_loc = hero.get("location", {}) or {}
        emy_loc = enemy.get("location", {}) or {}
        my_x = float(self._safe_get(my_loc, "x", 0))
        my_z = float(self._safe_get(my_loc, "z", 0))
        emy_x = float(self._safe_get(emy_loc, "x", 0))
        emy_z = float(self._safe_get(emy_loc, "z", 0))

        if self.transform_camp2_to_camp1:
            if my_x != 100000:
                my_x = -my_x
            if my_z != 100000:
                my_z = -my_z
        else:
            if emy_x != 100000:
                emy_x = -emy_x
            if emy_z != 100000:
                emy_z = -emy_z

        dx = emy_x - my_x
        dz = emy_z - my_z
        r = math.hypot(dx, dz)

        MAP_HALF = 15000.0
        MAP_DIAG = 42428.0
        dx_norm = max(0.0, min(1.0, (dx + MAP_HALF) / (MAP_HALF * 2)))
        dz_norm = max(0.0, min(1.0, (dz + MAP_HALF) / (MAP_HALF * 2)))
        r_norm = max(0.0, min(1.0, r / MAP_DIAG))

        vector_feature.extend([dx_norm, dz_norm, r_norm])

    # ==================================================================
    # 防越塔特征：是否处于敌方防御塔攻击范围内
    #
    # 作用：新手 AI 常因追击残血或走位失误冲入敌方塔下送人头。
    #       此特征在塔内时置 1.0，塔外置 0.0，让 PPO 学会"越塔需有把握"。
    # 输出维度：1
    # ==================================================================
    def get_is_in_enemy_tower_range(self, hero, vector_feature, feature_name):
        TOWER_ATK_RADIUS = 8800.0

        # 根据英雄阵营选择"敌方塔"：我方→敌方塔，敌方→我方塔
        hero_camp = hero.get("camp")
        if hero_camp == self.main_camp:
            target_tower = self.enemy_tower_pos
        else:
            target_tower = self.main_tower_pos

        if target_tower is None or hero is None:
            vector_feature.append(0.0)
            return

        my_loc = hero.get("location", {}) or {}
        my_x = float(self._safe_get(my_loc, "x", 0))
        my_z = float(self._safe_get(my_loc, "z", 0))

        if self.transform_camp2_to_camp1:
            if my_x != 100000:
                my_x = -my_x
            if my_z != 100000:
                my_z = -my_z

        if abs(my_x) > 90000 or abs(my_z) > 90000:
            vector_feature.append(0.0)
            return

        tower_x, tower_z = target_tower
        dist_to_tower = math.hypot(my_x - tower_x, my_z - tower_z)

        if dist_to_tower <= TOWER_ATK_RADIUS:
            vector_feature.append(1.0)
        else:
            vector_feature.append(0.0)

    # ==================================================================
    # 新增：Buff 状态特征（5 维）
    # ==================================================================
    def get_buff_features(self, hero, vector_feature, feature_name):
        has_heal = 1.0 if _hero_has_buff(hero, BUFF_HEAL_IDS) else 0.0
        has_speed = 1.0 if _hero_has_buff(hero, BUFF_SPEED_IDS) else 0.0
        has_slow = 1.0 if _hero_has_buff(hero, BUFF_SLOW_IDS) else 0.0
        has_cleanse = 1.0 if _hero_has_buff(hero, BUFF_CLEANSE_IDS) else 0.0

        # 英雄专属被动 buff：鲁班的强化普攻 / 狄仁杰的破甲窗口
        config_id = hero.get("actor_state", {}).get("config_id", 0) if hero else 0
        has_passive = 0.0
        if config_id == 112:   # 鲁班七号
            has_passive = 1.0 if _hero_has_buff(hero, {112001, 112015, 112025,
                112035, 112040, 112043, 112044, 112045, 112046, 112047,
                112048, 112100, 112200, 112201, 112300, 112301, 112320,
                112890, 112910, 112990, 112991}) else 0.0
        elif config_id == 133:  # 狄仁杰
            has_passive = 1.0 if _hero_has_buff(hero, {133000, 133001,
                133010, 133011, 133020, 133090, 133200, 133250, 133260,
                133950, 133951}) else 0.0

        vector_feature.extend([has_heal, has_speed, has_slow, has_cleanse, has_passive])

    # ==================================================================
    # 新增：蛋糕距离特征（2 维：到主方蛋糕距离、到敌方蛋糕距离）
    # ==================================================================
    def get_cake_dist_features(self, hero, vector_feature, feature_name):
        hero_camp = hero.get("camp") if hero else None
        cake_map = CAKE_LOCATIONS_BY_CAMP.get(hero_camp, {})
        main_cake = cake_map.get("main")
        enemy_cake = cake_map.get("enemy")

        my_loc = hero.get("location", {}) or {}
        hx = float(self._safe_get(my_loc, "x", 0))
        hz = float(self._safe_get(my_loc, "z", 0))
        MAP_DIAG = 42428.0

        if main_cake and hx != 100000 and hz != 100000:
            d_main = math.hypot(hx - main_cake[0], hz - main_cake[1]) / MAP_DIAG
            vector_feature.append(max(0.0, min(1.0, d_main)))
        else:
            vector_feature.append(0.0)

        if enemy_cake and hx != 100000 and hz != 100000:
            d_enemy = math.hypot(hx - enemy_cake[0], hz - enemy_cake[1]) / MAP_DIAG
            vector_feature.append(max(0.0, min(1.0, d_enemy)))
        else:
            vector_feature.append(0.0)
