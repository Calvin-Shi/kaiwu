#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.npc_process import NpcProcess
from agent_ppo.feature.feature_process.cake_process import CakeProcess

class FeatureProcess:
    def __init__(self, camp,logger):
        self.camp = camp
        self.logger=logger
        self.hero_process = HeroProcess(camp,self.logger)
        self.organ_process = OrganProcess(camp,self.logger)

        self.npc_process = NpcProcess(camp,self.logger)
        self.cake_process = CakeProcess(camp,self.logger)

    def reset(self, camp,logger):
        self.camp = camp
        self.logger=logger
        self.hero_process = HeroProcess(camp,self.logger)
        self.organ_process = OrganProcess(camp,self.logger)
        self.npc_process = NpcProcess(camp,self.logger)
        self.cake_process = CakeProcess(camp,self.logger)

    def process_organ_feature(self, frame_state):
        return self.organ_process.process_vec_organ(frame_state)

    def process_hero_feature(self, frame_state):
        return self.hero_process.process_vec_hero(frame_state)

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature = self.process_hero_feature(frame_state)
        organ_feature = self.process_organ_feature(frame_state)
        npc_feature = self.npc_process.generate_npc_feature(frame_state)
        cake_feature = self.cake_process.process_vec_cake(frame_state)

        feature = main_camp_hero_vector_feature + organ_feature + cake_feature + npc_feature

        return feature
