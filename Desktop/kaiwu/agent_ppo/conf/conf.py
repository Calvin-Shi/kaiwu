#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        # 推塔 / 守塔：防御塔血量比例
        "tower_hp_point": 5.0,
        # 前进：鼓励靠近敌方防御塔
        "forward": 0.01,
        "recall": 1.0,  # <--- 【新增】注册回城奖励及其初始权重
        # 生命值比例差（零和）
        "hp_point": 2.0,
        # 金币差（零和）
        "money": 0.002,
        # 经验差（零和）
        "exp": 0.005,
        # 击杀敌方英雄
        "kill": 3.0,
        # 自身被击杀
        "death": -3.0,
        # 法力值比例（零和）
        "ep_rate": 0.5,
        "hero_combo_window": 1.0,  # 【新增】英雄专属连招与状态窗口奖励
        "kill_gold_consistency": 1.0,   # 【新增】击杀与经济一致性纠偏
        "kill_tower_consistency": 1.0,  # 【新增】击杀与推塔一致性纠偏
        "skill5_flash": 1.0,           # 【新增】闪现边沿检测与事件关联奖励
        "cake_hunt": 2.0,             # 【新增】蛋糕/血包趋向奖励
        "cake_pickup": 5.0,           # 【新增】蛋糕/血包拾取瞬间奖励
    }
    # 动作空间宏定义
    RECALL_BUTTON_INDEX = 9  # <--- 【新增】12维离散空间中第9位为回城
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
# 特征维度 = 12(英雄) + 7(防御塔) = 19
#   英雄: is_alive(1) + location_x(1) + location_z(1) + hp_rate(1)
#        + ep_rate(1) + level(1) + money(1) + skill_cooldown(5) = 12
#   防御塔: is_alive(1) + belong_to_main_camp(1) + location_x(1) + location_z(1)
#         + relative_location_x(1) + relative_location_z(1) + hp_rate(1) = 7
class DimConfig:
    DIM_OF_FEATURE = [97]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        97 + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(97,), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.15

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    BETA_START = 0.025
    BETA_END = 0.001
    BETA_DECAY_STEPS = 50000

    data_shapes = [
        [(97 + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.997
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
