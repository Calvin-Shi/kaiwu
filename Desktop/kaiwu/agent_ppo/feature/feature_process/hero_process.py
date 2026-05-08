#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from enum import Enum
from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math
from collections import OrderedDict




class HeroProcess:
    def __init__(self, camp,logger):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        # self.main_camp_hero_dict = {}
        # self.enemy_camp_hero_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000


        self.max_skill_slots =7
        

        self.one_unit_feature_num = 12 + 4 * self.max_skill_slots
        self.unit_buff_num = 1

        self.step_counter = 0  # 全局帧计数器

        self.logger = logger

        # 跟踪技能使用情况
        self.skill_use_ema = {}       # raw_idx -> ema 值
        self.skill_last_used = {}     # raw_idx -> 上一次的 usedTimes

        self._REL_HALF_RANGE = 15000.0
        self._REL_FULL_RANGE = 30000.0
        self._MAX_DIST = 30000.0

        self.tower_attack_radius = 8800.0
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}


    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

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

        self.generate_hero_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)
        self._generate_organ_info_dict(frame_state)

        self.step_counter += 1
        if self.step_counter % 5 == 0:
            self.log_hero_info()
            self.log_others_info(frame_state)



        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")

        

        self.logger.info(f"[main_camp_hero_vector_feature] :{main_camp_hero_vector_feature}")
        
        self.logger.info(f"[enemy_camp_hero_vector_feature] :{enemy_camp_hero_vector_feature}")

        return main_camp_hero_vector_feature  + enemy_camp_hero_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["actor_state"]["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["actor_state"]["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["actor_state"]["config_id"]] = hero

    def generate_hero_info_dict(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()

        # 找到我方英雄并按照顺序编号
        for hero in frame_state["npc_states"]:
            if hero["sub_type"] != "ACTOR_SUB_hero" or hero["hp"] <= 0:
                continue
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["runtime_id"]] = hero
        self.main_camp_hero_dict = OrderedDict(sorted(self.main_camp_hero_dict.items()))

        # Find enemy heroes and number them in order
        # 找到敌方英雄并按照顺序编号
        for hero in frame_state["npc_states"]:
            if hero["sub_type"] != "ACTOR_SUB_hero" or hero["hp"] <= 0:
                continue
            if hero["camp"] != self.main_camp:
                self.enemy_camp_hero_dict[hero["runtime_id"]] = hero
        self.enemy_camp_hero_dict = OrderedDict(sorted(self.enemy_camp_hero_dict.items()))

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break


            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)

                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    value_vec = []
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            self.logger.info(f"[{camp} hero {num_heros_considered}] :{vector_feature[-self.one_unit_feature_num:]}")

            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["actor_state"]["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["actor_state"]["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def skills_features_all(self, hero, vector_feature, feature_name):
        


        # 1) 拿到技能槽列表
        slots = []
        
        slots = hero["skill_state"].get("slot_states", [])
        self.logger.info(f"[hero skill slots] :{slots}")
        if not isinstance(slots, list):
            slots = []
            self.logger.warning(f"Invalid skill slots data: {slots}")



        actual = 0
        max_slots = int(getattr(self, "max_skill_slots", 0) or 0)
        # 3) 遍历槽位，依次 append 特征
        for raw_idx, slot in enumerate(slots[:max_slots]):
            if not slot:
                vector_feature.extend([0.0, 0.0, 0.0,0.0])
                self.logger.info(f"slot {raw_idx} is empty")
                actual += 1
                continue
            # cd_ratio
            mx = float(slot["cooldown_max"]) if slot["cooldown_max"] > 0 else 0.0
            cd = float(slot["cooldown"])
            cd_ratio = (cd / mx) if mx > 0 else 0.0
            cd_ratio = max(0.0, min(1.0, cd_ratio))
            vector_feature.append(cd_ratio)

            # usable → {0,1} 
            usable = 1.0 if slot["usable"] else 0.0
            vector_feature.append(usable)

            # hit_rate
            used = int(slot["usedTimes"])
            hit = int(slot["hitHeroTimes"])
            denom = used if used > 0 else 1
            hit_rate = hit / denom
            hit_rate = max(0.0, min(1.0, hit_rate))
            vector_feature.append(hit_rate)

            # 近期使用频率（指数移动平均）
            use_rate_recent = self._update_use_rate_recent(raw_idx, slot, alpha=0.1)
            vector_feature.append(use_rate_recent)

            self.logger.info(
                f"[slot {raw_idx}] cd_ratio={cd_ratio:.2f}, usable={usable}, "
                f"hit_rate={hit_rate:.2f}, use_rate_recent={use_rate_recent:.2f}"
            )
            actual += 1

        
        pad_slots = max(0, max_slots-actual)
        if pad_slots:
            vector_feature.extend([0.0, 0.0, 0.0, 0.0] * pad_slots)


        
    def get_hp_rate(self, hero, vector_feature, _):
        hp = float(hero["actor_state"]["hp"])
        max_hp = float(hero["actor_state"]["max_hp"])
        hp_rate = (hp / max_hp) if max_hp > 0 else 0.0
        # 限幅到[0,1]
        hp_rate = 0.0 if hp_rate < 0 else (1.0 if hp_rate > 1 else hp_rate)
        vector_feature.append(hp_rate)

    def get_ep_rate(self, hero, vector_feature, _):
        # ep = energy/mana，字段名以你的协议为准：ep / max_ep
        ep = float(hero["actor_state"]["values"]["ep"])
        max_ep = float(hero["actor_state"]["values"]["max_ep"])
        ep_rate = (ep / max_ep) if max_ep > 0 else 0.0
        ep_rate = 0.0 if ep_rate < 0 else (1.0 if ep_rate > 1 else ep_rate)
        vector_feature.append(ep_rate)
        
    def get_death(self,hero,vector_feature,feature_name):
        value=hero["deadCnt"]
        vector_feature.append(value)
    def get_kill(self,hero,vector_feature,feature_name):
        value=hero["killCnt"]
        vector_feature.append(value)


    def log_hero_info(self):
        # 主英雄
        if getattr(self, "main_hero_info", None) is not None:
            hero = self.main_hero_info
            a = hero.get("actor_state", {})
            is_inGrass = hero.get("isInGrass",{})
            self.logger.info(f"[MAIN] cfg={a.get('config_id','?')} camp={a.get('camp','?')} is_in_grass={is_inGrass}")
            behave = a.get("behav_mode",{})
            self.logger.info(f"[MAIN] behave={behave}")
            buff_state = hero.get("buff_state", {})
            self.logger.info(
                f"  [MAIN BUFF] skills={buff_state.get('buff_skills','-')} marks={buff_state.get('buff_marks','-')}"
            )

        # 敌方英雄
        for _, enemy in self.enemy_camp_hero_dict.items():
            a   = enemy.get("actor_state", {})
            loc = a.get("location", {})
            self.logger.info(
                f"[ENEMY] cfgId={a.get('config_id','?')} hp={a.get('hp','?') } is_in_grass={enemy.get('isInGrass', {})}"
                f"pos=({loc.get('x','?')},{loc.get('z','?')})"
            )
            behave = a.get("behav_mode",{})
            self.logger.info(f"[ENEMY] behave={behave}")
            buff_state = enemy.get("buff_state", {})
            self.logger.info(
                f"  [ENEMY BUFF] skills={buff_state.get('buff_skills','-')} marks={buff_state.get('buff_marks','-')}"
            )
    

    def log_others_info(self, frame_state):
        bullets = frame_state.get("bullets", [])
        npc_states = frame_state.get("npc_states", [])

        npcs = [npc for npc in npc_states if npc.get("actor_type") in ("ACTOR_TYPE_MONSTER", "ACTOR_MONSTER","ACTOR_TYPE_SHENFU","ACTOR_SHENFU")]

        cakes = frame_state.get("cakes", [])

        self.logger.info(f"[BULLETS] :{bullets}")
        self.logger.info(f"[CAKES] :{cakes}")

        for npc in npcs:
            self.logger.info(f"[NPC] sub_type:{npc.get('sub_type','?')} "
                             f"cfgId:{npc.get('config_id','?')} " 
                             f"camp:{npc.get('camp','?')} "
                             f"camp_visible:{npc.get('camp_visible','?')} "
                             f"sight_area:{npc.get('sight_area','?')} "
                             f"beahave_mode:{npc.get('behav_mode','?')}")   
            


    def _update_use_rate_recent(self, raw_idx, slot, alpha=0.1):
        usedTimes = int(slot.get("usedTimes", 0)) if slot else 0
        prev_used = self.skill_last_used.get(raw_idx, usedTimes)
        fired = 1.0 if usedTimes > prev_used else 0.0

        ema_prev = self.skill_use_ema.get(raw_idx, 0.0)
        ema_new = alpha * fired + (1 - alpha) * ema_prev

        self.skill_use_ema[raw_idx] = ema_new
        self.skill_last_used[raw_idx] = usedTimes
        return ema_new
    


    def _generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()
        for organ in frame_state.get("npc_states", []):
            organ_camp = organ.get("camp")
            organ_subtype = organ.get("sub_type")
            if organ_subtype != "ACTOR_SUB_TOWER":
                continue
            if organ_camp == self.main_camp:
                self.main_camp_organ_dict["tower"] = organ
            else:
                self.enemy_camp_organ_dict["tower"] = organ

    def is_in_tower_range(self, hero_loc: dict, tower: dict) -> float:
        """判断一个单位是否在指定塔的攻击范围内 (0/1)"""
        if not tower:
            return 0.0
        # 英雄位置
        hx, hz = float(hero_loc.get("x", 100000)), float(hero_loc.get("z", 100000))
        if 100000 in (hx, hz):
            return 0.0

        # 塔位置
        tloc = tower.get("location", {})
        tx, tz = float(tloc.get("x", 100000)), float(tloc.get("z", 100000))
        if 100000 in (tx, tz):
            return 0.0

        # 镜像到统一坐标系
        if self.transform_camp2_to_camp1:
            hx, hz = -hx, -hz
            tx, tz = -tx, -tz

        dx, dz = hx - tx, hz - tz
        return 1.0 if (dx * dx + dz * dz) <= (self.tower_attack_radius ** 2) else 0.0

    def is_friend(self, hero, vector_feature, feature_name):
        value = 1.0 if hero["actor_state"]["camp"] == self.main_camp else 0.0
        vector_feature.append(value)

    def is_in_tower_attack_range(self, hero, vector_feature, feature_name):
        """判断一个英雄是否在“其敌方塔”的攻击范围内 (0/1)"""
        # 英雄所属阵营
        camp = hero["actor_state"]["camp"]

        # 选该英雄的“敌方塔”
        if camp == self.main_camp:
            enemy_tower = self.enemy_camp_organ_dict.get("tower")
        else:
            enemy_tower = self.main_camp_organ_dict.get("tower")

        hero_loc = hero["actor_state"].get("location", {})
        in_range = self.is_in_tower_range(hero_loc, enemy_tower)
        vector_feature.append(in_range)


    def _select_opponent(self, hero):
        """
        为给定 hero 选择“对手”：
        - 若 hero 属于我方，则选“最近的敌方英雄”
        - 若 hero 属于敌方，则选“我方主英雄”（若无主英雄，则最近的我方英雄）
        """
        if hero["actor_state"]["camp"] == self.main_camp:
            pool = list(self.enemy_camp_hero_dict.values())
            if not pool:
                return None
            return min(
                pool,
                key=lambda e: math.hypot(
                    float(e["actor_state"]["location"]["x"]) - float(hero["actor_state"]["location"]["x"]),
                    float(e["actor_state"]["location"]["z"]) - float(hero["actor_state"]["location"]["z"]),
                )
            )
        else:
            # 敌方英雄相对“我方主英雄”；若主英雄不存在则相对最近我方英雄
            if getattr(self, "main_hero_info", None) is not None:
                return self.main_hero_info
            pool = list(self.main_camp_hero_dict.values())
            if not pool:
                return None
            return min(
                pool,
                key=lambda f: math.hypot(
                    float(f["actor_state"]["location"]["x"]) - float(hero["actor_state"]["location"]["x"]),
                    float(f["actor_state"]["location"]["z"]) - float(hero["actor_state"]["location"]["z"]),
                )
            )

    def _rel_encode_to_target(self, hero, target):
        """
        返回 hero 相对 target 的 (relx, relz, r)，都归一化到 [0,1]。
        无法计算则返回 (0,0,0)。
        """
        if target is None:
            return 0.0, 0.0, 0.0

        hx = float(hero["actor_state"]["location"]["x"])
        hz = float(hero["actor_state"]["location"]["z"])
        tx = float(target["actor_state"]["location"]["x"])
        tz = float(target["actor_state"]["location"]["z"])

        if 100000 in (hx, hz, tx, tz):
            return 0.0, 0.0, 0.0

        # 与其他坐标处理一致：阵营2镜像到阵营1视角
        if self.transform_camp2_to_camp1:
            hx, hz = -hx, -hz
            tx, tz = -tx, -tz

        dx, dz = (tx - hx), (tz - hz)

        relx = (dx + self._REL_HALF_RANGE) / self._REL_FULL_RANGE
        relz = (dz + self._REL_HALF_RANGE) / self._REL_FULL_RANGE
        # clamp
        relx = 0.0 if relx < 0.0 else (1.0 if relx > 1.0 else relx)
        relz = 0.0 if relz < 0.0 else (1.0 if relz > 1.0 else relz)

        dist = math.hypot(dx, dz) / self._MAX_DIST
        r = 0.0 if dist < 0.0 else (1.0 if dist > 1.0 else dist)
        return relx, relz, r

    # === 新增：供 INI 调用的“特征函数” ===
    def rel_to_opponent_x(self, hero, vector_feature, feature_name):
        opp = self._select_opponent(hero)
        relx, _, _ = self._rel_encode_to_target(hero, opp)
        vector_feature.append(relx)

    def rel_to_opponent_z(self, hero, vector_feature, feature_name):
        opp = self._select_opponent(hero)
        _, relz, _ = self._rel_encode_to_target(hero, opp)
        vector_feature.append(relz)

    def rel_to_opponent_r(self, hero, vector_feature, feature_name):
        opp = self._select_opponent(hero)
        _, _, r = self._rel_encode_to_target(hero, opp)
        vector_feature.append(r)
