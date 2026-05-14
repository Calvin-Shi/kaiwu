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
        # 推塔：胜负核心，最高权重
        "tower_hp_point": 8.0,
        # 前进：适度加强位置引导
        "forward": 0.02,
        "recall": 1.0,
        # 生命值比例差（零和）
        "hp_point": 2.0,
        # 金币差（零和）
        "money": 0.002,
        # 经验差（零和）
        "exp": 0.005,
        # 击杀：对局关键事件
        "kill": 4.0,
        # 被击杀：对称于击杀
        "death": -4.0,
        # 法力值比例（零和）
        "ep_rate": 0.5,
        "hero_combo_window": 1.5,
        # 降低一致性惩罚权重，防止 AI 因怕惩罚而不敢打架
        "kill_gold_consistency": 0.3,
        "kill_tower_consistency": 0.3,
        "skill5_flash": 0.5,
        # 降低蛋糕权重，防止 AI 过度关注吃蛋糕而忽视推塔
        "cake_hunt": 0.5,
        "cake_pickup": 2.0,
    }
    # 动作空间宏定义
    RECALL_BUTTON_INDEX = 9  # <--- 【新增】12维离散空间中第9位为回城

    # ==================================================================
    # 攻击/施法动作索引，Kiting 危险区惩罚时对这些动作豁免
    ATTACK_BUTTON_INDICES = [1, 2, 3, 4, 5, 6]

    # ==================================================================
    # Kiting 拉扯弹性距离参数
    KITING_OPTIMAL_MIN = 6000
    KITING_OPTIMAL_MAX = 8500
    KITING_DANGER_DIST = 4000
    KITING_DIST_COEFF = 0.02
    # 怠惰惩罚：距离敌人超过此值且血量健康时每帧惩罚
    IDLE_PENALTY_DIST = 9500
    IDLE_PENALTY_VALUE = -0.02
    IDLE_PENALTY_HP_THRESHOLD = 0.5
    # 追逐缩放：敌方血量低于此比例时，危险边界线性收缩
    KITING_CHASE_HP_THRESHOLD = 0.30

    # ==================================================================
    # Anti-camp 连续平滑参数
    ANTI_CAMP_MIN_DIST = 1000
    ANTI_CAMP_HP_UPPER = 0.9
    ANTI_CAMP_HP_LOWER = 0.3
    ANTI_CAMP_MAX_PENALTY = -0.05

    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用；值越大衰减越慢
    TIME_SCALE_ARG = 8000
    # 指定帧数后移除 forward 奖励，防止后期无效前压
    REMOVE_FORWARD_AFTER = 1000
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
# 特征维度 159 = 39(self_hero) + 39(emy_hero) + 7(organ) + 11(tactical)
#              + 28(fri_soldiers_4x7) + 28(emy_soldiers_4x7) + 7(resource)
#   英雄(39): is_alive(1) + loc_x(1) + loc_z(1) + hp_rate(1) + ep_rate(1)
#            + level(1) + money(1) + auto_attack(1)
#            + skill_cd(5x4=20) + rel_to_opp(3) + in_tower_range(1)
#            + buffs(5) + cake_dist(2)
class DimConfig:
    DIM_OF_FEATURE = [159]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 576
    DATA_SPLIT_SHAPE = [
        159 + 85,  # 244  feature + legal
        1, 1, 1, 1, 1, 1, 1, 1,      # 8   reward, advantage, action[6]
        12, 16, 16, 16, 16, 9,        # 6   old label probabilities
        1, 1, 1, 1, 1, 1, 1,          # 7   sub_action[6], is_train
        LSTM_UNIT_SIZE,                # lstm cell
        LSTM_UNIT_SIZE,                # lstm hidden
    ]
    SERI_VEC_SPLIT_SHAPE = [(159,), (85,)]
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
        [(159 + 85) * 16],
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
