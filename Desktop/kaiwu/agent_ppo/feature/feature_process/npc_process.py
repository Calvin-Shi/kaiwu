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



class NpcProcess:
    """
    生成 NPC 特征（4友兵、4敌兵、1野怪），并与 HeroProcess 的换边语义保持一致。
    每个 NPC 的特征为 6 维：
      [abs_x, abs_z, rel_x, rel_z, hp_rate, is_friend]
    输出顺序：
      我方士兵1-4 → 敌方士兵1-4 → 野怪(1)
    """

    def __init__(self, camp, logger=None):
        self.main_camp = camp
        self.logger = logger
        # 与 HeroProcess 保持一致的换边开关
        self.transform_camp2_to_camp1 = (camp == "PLAYERCAMP_2")

        self.max_friendly_soldiers = 4
        self.max_enemy_soldiers = 4
        self.monster_count = 1  

        self.tower_attack_radius = 8800.0
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}

    # ========== 工具 ==========
    def _mirror_pos(self, x, z):
        """ 阵营2镜像为阵营1视角；协议里 100000 表示无效坐标时不镜像 """
        if self.transform_camp2_to_camp1 and x != 100000 and z != 100000:
            return -x, -z
        return x, z

    def _is_friend(self, camp):
        return camp == self.main_camp

    def _hp_rate(self, hp, max_hp):
        max_hp = float(max_hp or 0.0)
        hp = float(hp or 0.0)
        r = (hp / max_hp) if max_hp > 0 else 0.0
        return max(0.0, min(1.0, r))

    def _npc_feat(self, npc, hero_loc_mirrored,ally_tower=None, enemy_tower=None):
        """
        单个 NPC 的 7 维特征（维度不变）：
        [abs_x, abs_z, rel_x, rel_z, hp_rate, is_friend,in_enemy_tower]
        其中坐标全部归一化到 [0,1]；100000 视为无效坐标 → 0。
        """
        if npc is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0]

        # ---------- 绝对坐标（镜像后） ----------
        loc = npc.get("location", {})
        abs_x, abs_z = loc.get("x", 100000), loc.get("z", 100000)
        abs_x, abs_z = self._mirror_pos(abs_x, abs_z)

        # [-60000, 60000] → [0,1]（与 Hero 对齐）
        ABS_MAX = 60000.0
        if abs_x == 100000 or abs_z == 100000:
            abs_x_n = abs_z_n = 0.0
        else:
            abs_x_n = (abs_x + ABS_MAX) / (2.0 * ABS_MAX)
            abs_z_n = (abs_z + ABS_MAX) / (2.0 * ABS_MAX)
            # 裁剪到 [0,1]
            abs_x_n = 0.0 if abs_x_n < 0.0 else (1.0 if abs_x_n > 1.0 else abs_x_n)
            abs_z_n = 0.0 if abs_z_n < 0.0 else (1.0 if abs_z_n > 1.0 else abs_z_n)

        # ---------- 相对坐标（对齐 Cake） ----------
        REL_HALF, REL_FULL = 15000.0, 30000.0
        if hero_loc_mirrored is not None and abs_x != 100000 and abs_z != 100000:
            hx, hz = hero_loc_mirrored
            rel_x = abs_x - hx
            rel_z = abs_z - hz
            rel_x_n = (rel_x + REL_HALF) / REL_FULL
            rel_z_n = (rel_z + REL_HALF) / REL_FULL
            # 裁剪到 [0,1]
            rel_x_n = 0.0 if rel_x_n < 0.0 else (1.0 if rel_x_n > 1.0 else rel_x_n)
            rel_z_n = 0.0 if rel_z_n < 0.0 else (1.0 if rel_z_n > 1.0 else rel_z_n)
        else:
            rel_x_n = rel_z_n = 0.0

        # ---------- 其余两维保持不变 ----------
        hp_rate = self._hp_rate(npc.get("hp", 0), npc.get("max_hp", 0))
        is_friend = 1.0 if self._is_friend(npc.get("camp")) else 0.0

        in_enemy_tower = 0.0
        if npc.get("sub_type") == "ACTOR_SUB_SOLDIER":
            # 小兵才计算是否在敌方塔内
            in_enemy_tower = self._is_in_tower_range(abs_x, abs_z, enemy_tower)
        # 野怪/其他单位恒 0

        feat = [abs_x_n, abs_z_n, rel_x_n, rel_z_n, hp_rate, is_friend,in_enemy_tower]
        if self.logger:
            self.logger.info(f"[NPC FEAT] npc_id={npc.get('runtime_id', -1)} feat={feat}")
        return feat


    # ========== 收集 ==========
    def _friend_enemy_soldiers(self, frame_state):
        friends, enemies = [], []
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") != "ACTOR_SUB_SOLDIER":
                continue
            if npc.get("hp", 0) <= 0:
                continue
            (friends if self._is_friend(npc.get("camp")) else enemies).append(npc)

        # 按 runtime_id 升序并各取前 4 个
        friends.sort(key=lambda n: n.get("runtime_id", 0))
        enemies.sort(key=lambda n: n.get("runtime_id", 0))
        return friends[:self.max_friendly_soldiers], enemies[:self.max_enemy_soldiers]

    def _pick_monsters(self, frame_state, hero_loc_mirrored):
        """
        取与我方英雄最近的 1 个野怪（若不足则补零）。
        兼容 actor_type 的两种命名：ACTOR_MONSTER / ACTOR_TYPE_MONSTER
        """
        candidates = []
        for npc in frame_state.get("npc_states", []):
            at = npc.get("actor_type")
            if at not in ("ACTOR_MONSTER", "ACTOR_TYPE_MONSTER"):
                continue
            if npc.get("hp", 0) <= 0:
                continue
            # 计算镜像后的 npc 坐标与英雄的距离
            loc = npc.get("location", {})
            nx, nz = self._mirror_pos(loc.get("x", 100000), loc.get("z", 100000))
            if hero_loc_mirrored is not None and nx != 100000 and nz != 100000:
                hx, hz = hero_loc_mirrored
                d = math.hypot(nx - hx, nz - hz)
            else:
                d = float("inf")
            candidates.append((d, npc))

        candidates.sort(key=lambda x: x[0])  # 距离近的在前
        picked = [c[1] for c in candidates[:self.monster_count]]
        # 补齐到 monster_count
        while len(picked) < self.monster_count:
            picked.append(None)
        return picked

    def _my_hero_loc_mirrored(self, frame_state):
        """找到我方英雄位置并镜像；用于计算相对坐标"""
        for h in frame_state.get("hero_states", []):
            a = h.get("actor_state", {})
            if self._is_friend(a.get("camp")):
                loc = a.get("location", None)
                if loc is None:
                    return None
                return self._mirror_pos(loc.get("x", 100000), loc.get("z", 100000))
        return None

    # ========== 主入口 ==========
    def generate_npc_feature(self, frame_state):
        """
        输出顺序：
          我方士兵1-4 → 敌方士兵1-4 → 野怪(1)
        每项 7维,共 63 维。
        """
        self._generate_organ_info_dict(frame_state)
        ally_tower  = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")

        hero_loc_m = self._my_hero_loc_mirrored(frame_state)

        my_soldiers, enemy_soldiers = self._friend_enemy_soldiers(frame_state)
        # 补齐 4 个
        while len(my_soldiers) < self.max_friendly_soldiers:
            my_soldiers.append(None)
        while len(enemy_soldiers) < self.max_enemy_soldiers:
            enemy_soldiers.append(None)

        monsters = self._pick_monsters(frame_state, hero_loc_m)

        feats = []
        for npc in my_soldiers:
            feats.extend(self._npc_feat(npc, hero_loc_m,ally_tower, enemy_tower))
        for npc in enemy_soldiers:
            feats.extend(self._npc_feat(npc, hero_loc_m,enemy_tower, ally_tower))
        for npc in monsters:
            feats.extend(self._npc_feat(npc, hero_loc_m,ally_tower, enemy_tower))

        if self.logger:
            self.logger.debug(f"[NPC FEAT] len={len(feats)} (expect 63)")
        return feats

    def _generate_organ_info_dict(self, frame_state):
        """按阵营缓存本帧的防御塔（只取 ACTOR_SUB_TOWER）"""
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()
        for organ in frame_state.get("npc_states", []):
            if organ.get("sub_type") != "ACTOR_SUB_TOWER":
                continue
            if organ.get("camp") == self.main_camp:
                self.main_camp_organ_dict["tower"] = organ
            else:
                self.enemy_camp_organ_dict["tower"] = organ

    def _is_in_tower_range(self, abs_x: float, abs_z: float, tower: dict) -> float:
        """使用镜像后的绝对坐标与塔坐标计算是否在塔范围内(0/1)"""
        if not tower:
            return 0.0
        if abs_x == 100000 or abs_z == 100000:
            return 0.0
        loc = tower.get("location", {})
        tx, tz = loc.get("x", 100000), loc.get("z", 100000)
        if tx == 100000 or tz == 100000:
            return 0.0
        # 与全局视角对齐：塔坐标也镜像
        tx_m, tz_m = self._mirror_pos(tx, tz)
        dx, dz = abs_x - tx_m, abs_z - tz_m
        return 1.0 if (dx*dx + dz*dz) <= (self.tower_attack_radius ** 2) else 0.0


       