#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

小兵（Soldier）特征提取模块。

从 AIFrameState 的 npc_states 中筛选存活小兵（sub_type == 1 且 hp > 0），
按与我方英雄的欧式距离排序，取最近的 MAX_SOLDIER_NUM 个，提取固定维度特征。
不足时用 0 填充，保证输出维度恒为 MAX_SOLDIER_NUM * FEATURE_PER_SOLDIER。

集成方式：
    在 feature_process/__init__.py 中：
    1. from agent_ppo.feature.feature_process.soldier_process import SoldierProcess
    2. __init__ 中添加 self.soldier_process = SoldierProcess(camp)
    3. reset 中添加 self.soldier_process = SoldierProcess(camp)
    4. 新增方法：
         def process_soldier_feature(self, frame_state):
             return self.soldier_process.process_vec_soldier(frame_state)
    5. process_feature 中拼接：
         soldier_feature = self.process_soldier_feature(frame_state)
         feature = main_camp_hero_vector_feature + organ_feature + soldier_feature
    6. 更新 conf.py 中 DimConfig.DIM_OF_FEATURE 增加 20 维
"""

import math


# 最大观察小兵数量
MAX_SOLDIER_NUM = 4
# 每个小兵的特征维度
FEATURE_PER_SOLDIER = 5
# 小兵 sub_type 标识（ACTOR_SUB_SOLDIER）
SOLDIER_SUB_TYPE = 1
# 相对坐标归一化用的视野半径（与 hero_process / organ_process 保持一致）
VIEW_RANGE = 15000.0


class SoldierProcess:
    """
    小兵特征提取器。

    输出维度：MAX_SOLDIER_NUM * FEATURE_PER_SOLDIER = 4 * 5 = 20
    每个小兵槽位输出 5 维：
        [is_exist, is_enemy, hp_rate, relative_loc_x, relative_loc_z]
    """

    def __init__(self, camp):
        # 我方阵营标识，如 "PLAYERCAMP_1" / "PLAYERCAMP_2"
        self.main_camp = camp
        # 如果我方出生在右上角（PLAYERCAMP_2），需要对坐标做镜像翻转
        self.transform_camp2_to_camp1 = (camp == "PLAYERCAMP_2")

    # ------------------------------------------------------------------
    # 公开接口：输入 frame_state，输出固定长度的特征向量
    # ------------------------------------------------------------------
    def process_vec_soldier(self, frame_state):
        """
        主入口。返回长度为 MAX_SOLDIER_NUM * FEATURE_PER_SOLDIER 的 list[float]。
        """
        # 1. 获取我方英雄位置
        main_hero = self._get_main_hero(frame_state)
        if main_hero is None:
            return [0.0] * (MAX_SOLDIER_NUM * FEATURE_PER_SOLDIER)

        hero_x = main_hero["location"]["x"]
        hero_z = main_hero["location"]["z"]

        # 2. 从 npc_states 中筛选存活小兵
        soldiers = self._filter_soldiers(frame_state)

        # 3. 按与英雄的欧式距离排序
        soldiers_with_dist = []
        for soldier in soldiers:
            sx = soldier["location"]["x"]
            sz = soldier["location"]["z"]
            dist = math.hypot(sx - hero_x, sz - hero_z)
            soldiers_with_dist.append((dist, soldier))
        soldiers_with_dist.sort(key=lambda x: x[0])

        # 4. 截断：只取最近的 MAX_SOLDIER_NUM 个
        nearest = soldiers_with_dist[:MAX_SOLDIER_NUM]

        # 5. 提取特征 + 填充
        vector_feature = []
        for i in range(MAX_SOLDIER_NUM):
            if i < len(nearest):
                _, soldier = nearest[i]
                self._extract_soldier_feature(soldier, hero_x, hero_z, vector_feature)
            else:
                # Padding：该槽位无小兵，全部填 0
                vector_feature.extend([0.0] * FEATURE_PER_SOLDIER)

        return vector_feature

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _get_main_hero(self, frame_state):
        """从 hero_states 中找到我方英雄。"""
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == self.main_camp:
                return hero
        return None

    def _filter_soldiers(self, frame_state):
        """
        筛选条件：
        - sub_type == SOLDIER_SUB_TYPE (1)
        - hp > 0（存活）
        """
        soldiers = []
        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == SOLDIER_SUB_TYPE and npc.get("hp", 0) > 0:
                soldiers.append(npc)
        return soldiers

    def _extract_soldier_feature(self, soldier, hero_x, hero_z, vector_feature):
        """
        提取单个小兵的 5 维特征并 append 到 vector_feature：
            [is_exist, is_enemy, hp_rate, relative_loc_x, relative_loc_z]
        """
        # is_exist：该槽位有小兵
        vector_feature.append(1.0)

        # is_enemy：小兵阵营与我方不同则为敌方
        is_enemy = 1.0 if soldier.get("camp") != self.main_camp else 0.0
        vector_feature.append(is_enemy)

        # hp_rate：生命值比例，clamp 到 [0, 1]
        hp = float(soldier.get("hp", 0))
        max_hp = float(soldier.get("max_hp", 0))
        hp_rate = 0.0 if max_hp <= 0 else hp / max_hp
        hp_rate = max(0.0, min(1.0, hp_rate))
        vector_feature.append(hp_rate)

        # relative_loc_x / relative_loc_z：相对英雄的坐标差，归一化到 [-1, 1]
        sx = soldier["location"]["x"]
        sz = soldier["location"]["z"]
        rel_x = sx - hero_x
        rel_z = sz - hero_z

        # 镜像翻转：PLAYERCAMP_2 出生在右上角，坐标系需要取反以对齐 PLAYERCAMP_1 视角
        if self.transform_camp2_to_camp1:
            rel_x = -rel_x
            rel_z = -rel_z

        # 归一化到 [-1, 1]，超出视野范围的 clamp 到边界
        norm_x = max(-1.0, min(1.0, rel_x / VIEW_RANGE))
        norm_z = max(-1.0, min(1.0, rel_z / VIEW_RANGE))
        vector_feature.append(norm_x)
        vector_feature.append(norm_z)
