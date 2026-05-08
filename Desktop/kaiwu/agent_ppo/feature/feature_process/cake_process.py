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

class CakeProcess:
    def __init__(self, camp, logger):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.logger = logger

        self.transform_camp2_to_camp1 = (camp == "PLAYERCAMP_2")

        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}
        self.main_hero_info = None

        self._REL_HALF_RANGE = 15000.0
        self._REL_FULL_RANGE = 30000.0
        self._MAX_DIST = 30000.0

    # --- 复制 Organ 的取塔逻辑 ---
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

    # --- 找到主英雄（用于相对坐标） ---
    def _update_main_hero(self, frame_state):
        self.main_hero_info = None
        for hero in frame_state.get("hero_states", []):
            if hero.get("actor_state", {}).get("camp") == self.main_camp:
                self.main_hero_info = hero
                break
        if self.main_hero_info is None and frame_state.get("hero_states"):
            self.main_hero_info = frame_state["hero_states"][0]

    # --- 按“一塔最近”分类（不兜底） ---
    def _classify_cakes_by_tower(self, cakes):
        ally_tower = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")
        if ally_tower is None or enemy_tower is None:
            return None, None

        ally_pos = ally_tower.get("location", {})
        enemy_pos = enemy_tower.get("location", {})

        def sqr_dist(p, q):
            return (float(p.get("x", 0)) - float(q.get("x", 0)))**2 + \
                   (float(p.get("z", 0)) - float(q.get("z", 0)))**2

        ally_cake, enemy_cake = None, None
        for c in cakes or []:
            loc = (c.get("collider") or {}).get("location") or {}
            da = sqr_dist(loc, ally_pos)
            de = sqr_dist(loc, enemy_pos)
            if da <= de:
                ally_cake = c
            else:
                enemy_cake = c
        return ally_cake, enemy_cake

    # --- 编码为固定 8 维 ---
    def _encode_one_cake(self, cake, is_ally_flag: float):
        if cake is None or self.main_hero_info is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        loc = (cake.get("collider") or {}).get("location") or {}
        x = float(loc.get("x", 100000.0))
        z = float(loc.get("z", 100000.0))
        if x == 100000.0 or z == 100000.0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        hx = float(self.main_hero_info["actor_state"]["location"]["x"])
        hz = float(self.main_hero_info["actor_state"]["location"]["z"])
        if self.transform_camp2_to_camp1:
            hx, hz = -hx, -hz
            x, z = -x, -z

        dx, dz = x - hx, z - hz
        relx = (dx + self._REL_HALF_RANGE) / self._REL_FULL_RANGE
        relz = (dz + self._REL_HALF_RANGE) / self._REL_FULL_RANGE
        relx = 0.0 if relx < 0.0 else (1.0 if relx > 1.0 else relx)
        relz = 0.0 if relz < 0.0 else (1.0 if relz > 1.0 else relz)

        dist = (dx*dx + dz*dz) ** 0.5
        dist_norm = dist / self._MAX_DIST
        dist_norm = 0.0 if dist_norm < 0.0 else (1.0 if dist_norm > 1.0 else dist_norm)

        ABS_MAX = 60000.0
        absx = (x + ABS_MAX) / (2.0 * ABS_MAX)
        absz = (z + ABS_MAX) / (2.0 * ABS_MAX)
        absx = 0.0 if absx < 0.0 else (1.0 if absx > 1.0 else absx)
        absz = 0.0 if absz < 0.0 else (1.0 if absz > 1.0 else absz)

        return [absx, absz, relx, relz, dist_norm, is_ally_flag]

    # 主入口
    def process_vec_cake(self, frame_state):
        self._generate_organ_info_dict(frame_state)
        self._update_main_hero(frame_state)

        cakes = frame_state.get("cakes", []) or []
        ally_cake, enemy_cake = self._classify_cakes_by_tower(cakes)

        vec = []
        vec.extend(self._encode_one_cake(ally_cake, 1.0))
        vec.extend(self._encode_one_cake(enemy_cake, 0.0))

        self.logger.info(f"[cakes] ally={vec[:6]} enemy={vec[6:12]} total_len={len(vec)}")
        return vec

        