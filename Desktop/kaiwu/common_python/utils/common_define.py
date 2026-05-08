#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class CommonDefine(object):
    # 进程名字, 其中main是为了便于在七彩石上管理, 不是实际存在的进程名
    SERVER_AISRV = "aisrv"
    SERVER_ACTOR = "actor"
    SERVER_LEARNER = "learner"
    SERVER_MAIN = "main"
    SERVER_MODELPOOL = "modelpool"
    SERVER_MODELPOOL_PROXY = "modelpool_proxy"
    SERVER_PYTHON = "python3"
    SERVER_CLIENT = "client"
    SERVER_BATTLE_SRV = "battlesrv"
    TRAIN_TEST_CMDLINE = "train_test.py"
    SERVER_ARENA = "arena"
    SERVER_GAMECORE = "gamecore"
    SERVER_ENV = "env"

    # 普罗米修斯的push/pull模式
    USE_PROMETHEUS_WAY_PULL = "pull"
    USE_PROMETHEUS_WAY_PUSH = "push"

    # 监控相关 | Monitor related
    MONITOR_VERSION_V2 = "v2"  # 监控版本 v2 | Monitor version v2
    MONITOR_VERSION_V1 = "v1"  # 监控版本 v1 | Monitor version v1
    MONITOR_METRICS_PREFIX = "kaiwu_"  # 监控指标前缀 | Monitor metrics prefix

    # checkpoint文件
    CHECK_POINT_FILE = "checkpoint"
    KAIWU_CHECK_POINT_FILE = "kaiwu_checkpoint"
    KAIWU_MODEL_CKPT = "model.ckpt"
    KAIWU_MODEDL_WIGHT = "model.wight"
    KAIWU_ONNX_FILE = "onnx"
    KAIWU_PB_FILE = "model.pb"

    # KaiwuDRL支持的GPU机器类型
    GPU_MACHINE_A100 = "A100"
    GPU_MACHINE_V100 = "V100"
    GPU_MACHINE_T4 = "T4"
    GPU_MACHINE_P100 = "P100"
    GPU_MACHINE_CPU = "CPU"

    # configparser的默认配置
    CONFIG_DEFAULT_INT = 0
    CONFIG_DEFAULT_FLOAT = 0.0
    CONFIG_DEFAULT_BOOL = False
    CONFIG_DEFAULT_STRING = ""

    # COS桶下的key名字
    COS_BUCKET_KEY = "kaiwu_drl_models/"
