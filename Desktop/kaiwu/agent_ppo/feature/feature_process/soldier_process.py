#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Author: Tencent AI Arena Authors
兼容最新 Lite 整数协议与旧版字符串协议的 NPC 特征提取模块。
"""

from enum import Enum
import math
from collections import OrderedDict


# Known cake sub_types in the game engine (health/mana recovery items)
# 已知的蛋糕/血包 sub_type 值
CAKE_SUB_TYPES = {12, 13, 14, 22, 24, 31, 32}


class NpcProcess:
    def __init__(self, camp, logger=None):
        self.main_camp = camp
        self.logger = logger
        self.transform_camp2_to_camp1 = (camp == "PLAYERCAMP_2")

        self.max_friendly_soldiers = 4
        self.max_enemy_soldiers = 4
        self.resource_count = 1

        self.tower_attack_radius = 8800.0
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}

    def _mirror_pos(self, x, z):
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

    def _npc_feat(self, npc, hero_loc_mirrored, ally_tower=None, enemy_tower=None):
        if npc is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        loc = npc.get("location", {})
        abs_x, abs_z = loc.get("x", 100000), loc.get("z", 100000)
        abs_x, abs_z = self._mirror_pos(abs_x, abs_z)

        ABS_MAX = 60000.0
        if abs_x == 100000 or abs_z == 100000:
            abs_x_n = abs_z_n = 0.0
        else:
            abs_x_n = (abs_x + ABS_MAX) / (2.0 * ABS_MAX)
            abs_z_n = (abs_z + ABS_MAX) / (2.0 * ABS_MAX)
            abs_x_n = 0.0 if abs_x_n < 0.0 else (1.0 if abs_x_n > 1.0 else abs_x_n)
            abs_z_n = 0.0 if abs_z_n < 0.0 else (1.0 if abs_z_n > 1.0 else abs_z_n)

        REL_HALF, REL_FULL = 15000.0, 30000.0
        if hero_loc_mirrored is not None and abs_x != 100000 and abs_z != 100000:
            hx, hz = hero_loc_mirrored
            rel_x = abs_x - hx
            rel_z = abs_z - hz
            rel_x_n = (rel_x + REL_HALF) / REL_FULL
            rel_z_n = (rel_z + REL_HALF) / REL_FULL
            rel_x_n = 0.0 if rel_x_n < 0.0 else (1.0 if rel_x_n > 1.0 else rel_x_n)
            rel_z_n = 0.0 if rel_z_n < 0.0 else (1.0 if rel_z_n > 1.0 else rel_z_n)
        else:
            rel_x_n = rel_z_n = 0.0

        hp_rate = self._hp_rate(npc.get("hp", 0), npc.get("max_hp", 0))
        is_friend = 1.0 if self._is_friend(npc.get("camp")) else 0.0

        # 最后一个特征位语义因实体类型而异：
        #   - 小兵：是否处于敌方塔攻击范围内 (in_enemy_tower)
        #   - 蛋糕/血包：蛋糕标识 (is_cake = 1.0)
        #   - 野怪/其他：0.0
        in_enemy_tower = 0.0
        is_cake = 0.0
        if self._is_soldier(npc):
            in_enemy_tower = self._is_in_tower_range(abs_x, abs_z, enemy_tower)
        elif self._is_cake(npc):
            is_cake = 1.0

        feat = [abs_x_n, abs_z_n, rel_x_n, rel_z_n, hp_rate, is_friend, is_cake if is_cake > 0 else in_enemy_tower]
        return feat

    def _friend_enemy_soldiers(self, frame_state):
        friends, enemies = [], []
        for npc in frame_state.get("npc_states", []):
            if str(npc.get("sub_type", "")) not in ("ACTOR_SUB_SOLDIER", "11"):
                continue
            if npc.get("hp", 0) <= 0:
                continue
            (friends if self._is_friend(npc.get("camp")) else enemies).append(npc)

        friends.sort(key=lambda n: n.get("runtime_id", 0))
        enemies.sort(key=lambda n: n.get("runtime_id", 0))
        return friends[:self.max_friendly_soldiers], enemies[:self.max_enemy_soldiers]

    @staticmethod
    def _is_cake(npc):
        """检测 NPC 是否为蛋糕/血包/恢复符文。
        综合判定策略：
        1. 优先按已知 sub_type 精确匹配
        2. 按 actor_type 字段进行字符串匹配 (CAKE, ORGAN)
        3. 兜底排除法：排除士兵、防御塔、野怪后，存活且距离合理的视为潜在蛋糕
        """
        sub = int(npc.get("sub_type", -1) or -1)
        at = str(npc.get("actor_type", "") or "")

        # 精确匹配：已知的 cake sub_type 值
        if sub in CAKE_SUB_TYPES:
            return True

        # 字符串匹配：actor_type 中包含 cake 或 organ 关键词
        at_upper = at.upper()
        if "CAKE" in at_upper or "ORGAN" in at_upper or "CHERRY" in at_upper:
            return True

        return False

    def _is_monster(self, npc):
        """保留野怪检测作为后备。"""
        at = str(npc.get("actor_type", ""))
        return at in ("ACTOR_MONSTER", "ACTOR_TYPE_MONSTER", "2")

    def _is_soldier(self, npc):
        """检测是否为小兵。"""
        st = str(npc.get("sub_type", ""))
        return st in ("ACTOR_SUB_SOLDIER", "11")

    def _is_tower(self, npc):
        """检测是否为防御塔。"""
        st = str(npc.get("sub_type", ""))
        return st in ("ACTOR_SUB_TOWER", "21")

    def _pick_resources(self, frame_state, hero_loc_mirrored):
        """优先选择蛋糕/血包，次选野怪作为可拾取资源目标。
        蛋糕优先级高于野怪：当同时存在蛋糕和野怪时，选择距离英雄最近的蛋糕。
        """
        cake_candidates = []
        monster_candidates = []

        for npc in frame_state.get("npc_states", []):
            if npc.get("hp", 0) <= 0:
                continue

            if self._is_soldier(npc):
                continue
            if self._is_tower(npc):
                continue

            loc = npc.get("location", {})
            nx, nz = self._mirror_pos(loc.get("x", 100000), loc.get("z", 100000))
            if hero_loc_mirrored is not None and nx != 100000 and nz != 100000:
                hx, hz = hero_loc_mirrored
                d = math.hypot(nx - hx, nz - hz)
            else:
                d = float("inf")

            if self._is_cake(npc):
                cake_candidates.append((d, npc))
            elif self._is_monster(npc):
                monster_candidates.append((d, npc))
            # 不认识的 NPC 类型也作为潜在蛋糕处理（兜底）
            else:
                cake_candidates.append((d, npc))

        # 蛋糕优先，选择最近的
        cake_candidates.sort(key=lambda x: x[0])
        if cake_candidates:
            return [cake_candidates[0][1]]

        # 无蛋糕时退回野怪
        monster_candidates.sort(key=lambda x: x[0])
        picked = [c[1] for c in monster_candidates[:self.resource_count]]
        while len(picked) < self.resource_count:
            picked.append(None)
        return picked

    def _my_hero_loc_mirrored(self, frame_state):
        # 【极其重要：兼容新老协议的英雄提取嵌套】
        for h in frame_state.get("hero_states", []):
            camp = h.get("camp") if h.get("camp") is not None else h.get("actor_state", {}).get("camp")
            if self._is_friend(camp):
                loc = h.get("location") or h.get("actor_state", {}).get("location")
                if loc is None:
                    return None
                return self._mirror_pos(loc.get("x", 100000), loc.get("z", 100000))
        return None

    def generate_npc_feature(self, frame_state):
        self._generate_organ_info_dict(frame_state)
        ally_tower  = self.main_camp_organ_dict.get("tower")
        enemy_tower = self.enemy_camp_organ_dict.get("tower")

        hero_loc_m = self._my_hero_loc_mirrored(frame_state)

        my_soldiers, enemy_soldiers = self._friend_enemy_soldiers(frame_state)
        while len(my_soldiers) < self.max_friendly_soldiers:
            my_soldiers.append(None)
        while len(enemy_soldiers) < self.max_enemy_soldiers:
            enemy_soldiers.append(None)

        # 优先选蛋糕，次选野怪
        resources = self._pick_resources(frame_state, hero_loc_m)

        feats = []
        for npc in my_soldiers:
            feats.extend(self._npc_feat(npc, hero_loc_m, ally_tower, enemy_tower))
        for npc in enemy_soldiers:
            feats.extend(self._npc_feat(npc, hero_loc_m, enemy_tower, ally_tower))
        for npc in resources:
            feats.extend(self._npc_feat(npc, hero_loc_m, ally_tower, enemy_tower))

        return feats

    def _generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()
        for organ in frame_state.get("npc_states", []):
            # 【极其重要：兼容新老协议的防御塔判定】
            if str(organ.get("sub_type", "")) not in ("ACTOR_SUB_TOWER", "21"):
                continue
            if organ.get("camp") == self.main_camp:
                self.main_camp_organ_dict["tower"] = organ
            else:
                self.enemy_camp_organ_dict["tower"] = organ

    def _is_in_tower_range(self, abs_x: float, abs_z: float, tower: dict) -> float:
        if not tower:
            return 0.0
        if abs_x == 100000 or abs_z == 100000:
            return 0.0
        loc = tower.get("location", {})
        tx, tz = loc.get("x", 100000), loc.get("z", 100000)
        if tx == 100000 or tz == 100000:
            return 0.0
        tx_m, tz_m = self._mirror_pos(tx, tz)
        dx, dz = abs_x - tx_m, abs_z - tz_m
        return 1.0 if (dx*dx + dz*dz) <= (self.tower_attack_radius ** 2) else 0.0