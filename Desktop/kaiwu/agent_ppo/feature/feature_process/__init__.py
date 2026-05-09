#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import math
from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
# 【修改点 1】：从旧文件名中，导入新的 NpcProcess 类
from agent_ppo.feature.feature_process.soldier_process import NpcProcess 

class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        # 【修改点 2】：用新的引擎，替换掉旧的实例
        self.npc_process = NpcProcess(camp)       

        self.last_hero_x = 0.0
        self.last_hero_z = 0.0
        self.last_dx_norm = 0.0
        self.last_dz_norm = 0.0

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.npc_process = NpcProcess(camp)
        
        self.last_hero_x = 0.0
        self.last_hero_z = 0.0
        self.last_dx_norm = 0.0
        self.last_dz_norm = 0.0

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_soldier_feature(self, frame_state):
        return self.soldier_process.process_vec_soldier(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        advanced_tactical_feature = self.extract_advanced_features(frame_state)
        
        # 【修改点 3】：调用新引擎的 generate_npc_feature 方法 (输出63维)
        npc_feature = self.npc_process.generate_npc_feature(frame_state) 

        # 总特征拼接
        feature = main_camp_hero_vector_feature + organ_feature + advanced_tactical_feature + npc_feature

        return feature

    def extract_advanced_features(self, frame_state):
        hero_x, hero_z = 0.0, 0.0
        enemy_x, enemy_z, enemy_alive, enemy_hp_rate = 0.0, 0.0, 0.0, 0.0
        tower_x, tower_z, tower_alive = 0.0, 0.0, 0.0

        for hero in frame_state.get("hero_states", []):
            if hero["camp"] == self.camp:
                hero_x = hero["location"]["x"]
                hero_z = hero["location"]["z"]
            else:
                enemy_alive = 1.0 if hero["hp"] > 0 else 0.0
                enemy_x = hero["location"]["x"]
                enemy_z = hero["location"]["z"]
                enemy_hp_rate = hero["hp"] / max(hero["max_hp"], 1)

        for organ in frame_state.get("npc_states", []):
            if organ["camp"] != self.camp and organ["sub_type"] == 21:
                tower_alive = 1.0 if organ["hp"] > 0 else 0.0
                tower_x = organ["location"]["x"]
                tower_z = organ["location"]["z"]

        if self.camp == "PLAYERCAMP_2":
            hero_x, hero_z = -hero_x, -hero_z
            if enemy_x != 100000: enemy_x = -enemy_x
            if enemy_z != 100000: enemy_z = -enemy_z
            if tower_x != 100000: tower_x = -tower_x
            if tower_z != 100000: tower_z = -tower_z

        dist_enemy = math.sqrt((hero_x - enemy_x)**2 + (hero_z - enemy_z)**2) if enemy_alive else 30000.0
        dist_enemy_norm = min(1.0, dist_enemy / 30000.0)

        dist_tower = math.sqrt((hero_x - tower_x)**2 + (hero_z - tower_z)**2) if tower_alive else 30000.0
        dist_tower_norm = min(1.0, dist_tower / 30000.0)

        in_tower_danger = 1.0 if (tower_alive and dist_tower < 8500) else 0.0

        dx = hero_x - self.last_hero_x
        dz = hero_z - self.last_hero_z

        dx_norm, dz_norm = 0.0, 0.0
        displacement = math.sqrt(dx**2 + dz**2)
        
        if displacement > 0.1:
            dx_norm = dx / displacement
            dz_norm = dz / displacement

        cos_sim = 1.0 
        if displacement > 0.1 and (self.last_dx_norm != 0 or self.last_dz_norm != 0):
            cos_sim = dx_norm * self.last_dx_norm + dz_norm * self.last_dz_norm

        self.last_hero_x, self.last_hero_z = hero_x, hero_z
        self.last_dx_norm, self.last_dz_norm = dx_norm, dz_norm

        adv_feats = [
            enemy_alive,
            (enemy_x / 15000.0) if enemy_x != 100000 else 0.0,
            (enemy_z / 15000.0) if enemy_z != 100000 else 0.0,
            enemy_hp_rate,
            dist_enemy_norm,
            dist_tower_norm,
            in_tower_danger,
            dx_norm,
            dz_norm,
            cos_sim
        ]
        return adv_feats