#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # ==================================================================
    # 奖励权重配置（9 项）
    # ==================================================================
    #   零和项 (tower_hp_point / hp_point / money / exp / kill / death):
    #     单帧奖励 = (己方当前帧差值 - 敌方当前帧差值)
    #   特殊项 (forward / ep_rate / last_hit): 见各自计算逻辑
    # ==================================================================
    REWARD_WEIGHT_DICT = {
        "tower_hp_point": 10.0,   # 防御塔血量（零和，核心胜负条件）
        "hp_point": 2.0,          # 英雄血量比例（零和，双开方）
        "ep_rate": 0.75,          # 法力值比例（仅己方、仅增加时给奖励）
        "last_hit": 0.5,          # 补刀小兵（己方+1，敌方-1）
        "kill": -0.6,             # 击杀（零和，单次击杀净奖励=+0.4）
        "death": -1.0,            # 死亡（零和，配合kill形成+0.4净奖励）
        "forward": 0.01,          # 推进进度（HP>99%且英雄在塔后方时触发）
        "money": 0.004,           # 金钱（零和）
        "exp": 0.004,             # 经验（零和，满级后=0）
    }

    # ==================================================================
    # 时间衰减
    #   公式: 最终奖励 = 基础奖励 * 0.6 ^ (frame_no / TIME_SCALE_ARG)
    TIME_SCALE_ARG = 8000
    # 指定帧数后移除 forward 奖励
    REMOVE_FORWARD_AFTER = 1000
    # 模型保存间隔
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
# 特征维度 177 = 42(self_hero) + 42(emy_hero) + 7(organ) + 23(tactical)
#              + 28(fri_soldiers_4x7) + 28(emy_soldiers_4x7) + 7(resource)
#   英雄(42): is_alive(1) + loc_x(1) + loc_z(1) + hp_rate(1) + ep_rate(1)
#            + level(1) + money(1) + auto_attack(1)
#            + skill_cd(5x4=20) + rel_to_opp(3) + in_tower_range(1)
#            + buffs(5) + cake_dist(2) + attack_speed(1) + move_speed(1)
#            + is_under_tower_atk(1)
#   战术(23): 敌方状态(7) + 移动方向(3) + 兵线推进(1) + 子弹追踪(3)
#            + 技能射程(4) + 游戏时间(5)
class DimConfig:
    DIM_OF_FEATURE = [177]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 576
    DATA_SPLIT_SHAPE = [
        177 + 85,  # 262  feature + legal
        1, 1, 1, 1, 1, 1, 1, 1,      # 8   reward, advantage, action[6]
        12, 16, 16, 16, 16, 9,        # 6   old label probabilities
        1, 1, 1, 1, 1, 1, 1,          # 7   sub_action[6], is_train
        LSTM_UNIT_SIZE,                # lstm cell
        LSTM_UNIT_SIZE,                # lstm hidden
    ]
    SERI_VEC_SPLIT_SHAPE = [(177,), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [True] * 6

    # P0: 伪自回归 Target Head — Button Embedding 维度
    BUTTON_EMBED_DIM = 64

    CLIP_PARAM = 0.15
    MIN_POLICY = 0.00001

    BETA_START = 0.025
    BETA_END = 0.001
    BETA_DECAY_STEPS = 50000

    # data_shapes: per-frame sample element sizes (last 2 = lstm hidden/cell)
    LS = LABEL_SIZE_LIST
    data_shapes = [
        [(177 + 85) * 16],
        [16], [16], [16], [16], [16], [16], [16], [16],
        [LS[0] * 16], [LS[1] * 16], [LS[2] * 16], [LS[3] * 16], [LS[4] * 16], [LS[5] * 16],
        [16], [16], [16], [16], [16], [16], [16],
        [LSTM_UNIT_SIZE],
        [LSTM_UNIT_SIZE],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.997
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
