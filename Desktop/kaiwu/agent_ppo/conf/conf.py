#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT = {
        "hp_point":2.0,
        "tower_hp_point": 11.7,
        "money":0.008,
        "exp":0.008,
        "ep_rate":0.5,
        "death":-1.2,
        "kill":0.2,
        "last_hit":0.35,
        # "forward": 0.03,
        "passive":0.9,
        "skill1":0.2,
        "skill2":1.3,
        "skill3":1.0,
        "time_decay": 0.5,
        "skill5_flash": 0.8,
        "kill_gold_consistency": 0.4,     # 击杀领先但经济不领先 → 轻罚
        "kill_tower_consistency": 0.8,    # 击杀领先但“净塔压”不领先 → 轻罚
        "sxx_armorbreak_window": 1.6,    # 破甲窗口奖励（核心）
        "sxx_s1_enh_aa_timing": 1.2,

    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800

    # 训练步长换算（你的环境是 6 帧=1 step）
    STEP_LEN_FRAMES = 6

    # 破甲窗口（按 step），你要严格→用 20 步（≈120帧）
    SXX2_WIN_STEPS  = 20

    # 连招内的“最佳跟进”窗口（按 step）
    SXX2_S1_MAX_STEPS  = 5     # 2 后 1 的最佳窗口（~30帧）；快跟更香
    SXX2_S3_MAX_STEPS  = 8     # 2 后 3 的最佳窗口（~48帧）

    # 3 之后用于“收割加成”的窗口（按 step）
    SXX2_FINISH_WINDOW_STEPS = 8

    # 奖惩幅度（保持与你现有奖励量级一致）
    SXX2_MISS_PENALTY   = -0.05   # 2 未命中轻罚
    SXX2_DMG_SCALE      = 1.0     # 窗口内按敌方掉血%累计奖励的比例
    SXX2_S1_HIT_BONUS   = 0.15    # 窗口内 1 命中离散加分
    SXX2_S3_HIT_BONUS   = 0.30    # 窗口内 3 命中离散加分
    SXX2_KILL_BONUS     = 0.8     # 窗口内击杀加分（叠加在 kill 项之上）
    SXX2_NO_FOLLOW_PEN  = -0.02   # 窗口结束若几乎无有效跟进的小罚
    SXX2_MIN_DMG_EPS    = 0.01    # 判定“有造成伤害”的最小 HP％阈值

    # 延迟衰减（鼓励“快跟”）：奖励 *= exp( - (step延迟) / TAU )
    SXX2_LATENCY_TAU    = 4

    # —— S1 强化普攻（卡→释放）参数（按 step；1 step = STEP_LEN_FRAMES 帧）——
    SXX1_CD_READY_RATIO = 0.20        # S1 冷却比例 cd/max_cd ≤ 该阈值 → 进入“应当释放期”
    SXX1_RELEASE_GRACE_STEPS = 2      # 进入应当释放期后，宽限 GRACE 步内释放：越快越香
    SXX1_HOLD_MIN_STEPS = 3           # 至少持有这么久（避免一放就打；仅非对抗态才有意义）

    SXX1_COMBAT_MAX_HOLD_STEPS = 2    # 对抗态下最多“憋”这么多步；超过则每步轻罚

    SXX1_RELEASE_BASE = 0.25          # 成功打出强化普攻的基础奖励
    SXX1_IN_AB_MULT = 1.3             # 若在 S2 破甲窗口内释放 → 乘子
    SXX1_LATENCY_TAU = 2.5            # 延迟衰减常数：越早释放越香（e^(-latency/TAU)）

    SXX1_CHAIN_S1_STEPS = 3           # 成功释放后，短窗口内再次 S1（滚） → 额外奖励
    SXX1_CHAIN_BONUS = 0.20

    SXX1_SKIP_PENALTY = -0.04         # 持有强化却又放了下一次 S1（浪费强化）的小罚
    SXX1_COMBAT_HOLD_PEN = -0.01      # 对抗态下超时还在“硬卡”的每步轻罚
    SXX1_LATE_RELEASE_PEN = -0.01     # 进入应当释放期后，过宽限还不打的每步轻罚

    # —— 不用 behav_mode 的“对抗态”判定（纯事件/数值）——
    SXX_COMBAT_RECENT_STEPS = 2       # 近 N 步内有技能事件 or 任一方掉血 → 视为对抗
    SXX_COMBAT_PROX_DIST = None       # 可选：双方距离 ≤ 阈值 也视为“接战环境”（不用就 None）
    SXX_COMBAT_PROX_STEPS = 2

    # —— 不用 behav_mode 的“普攻已打出”判定（保守）——
    AA_PROX_DIST = None               # 可选：双方距离 ≤ 阈值 才允许判为普攻（防塔伤/远程噪声）
    AA_MIN_DMG_PCT = 0.005            # 本帧敌人 HP% 下降至少达到该阈值才认为“打出了一下”


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    DIM_OF_FEATURE = [40+40+7+7+12+63]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        40+40+7+7+12+63 + 85,
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
    SERI_VEC_SPLIT_SHAPE = [(40+40+7+7+12+63,), (85,)]
    INIT_LEARNING_RATE_START = 1e-3
    TARGET_LR = 1e-4
    TARGET_STEP = 5000
    BETA_START = 0.025
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

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(40+40+7+7+12+63 + 85) * 16],
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

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
