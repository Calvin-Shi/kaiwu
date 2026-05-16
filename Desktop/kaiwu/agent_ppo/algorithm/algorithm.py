#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

王者荣耀 1v1 算法核心模块（PPO 进阶优化版）。

优化点：
1. 引入全局优势函数归一化 (Advantage Normalization)，极大平滑由击杀、补刀带来的大幅度 Reward 波动。
2. 引入指数衰减+强制保底的熵系数 (Entropy Coefficient) 更新逻辑，防止 MOBA 训练后期因熵过小导致策略坍塌（停止探索）。
"""

import torch
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.last_report_monitor_time = 0
        
        # 新增：记录当前熵系数，初始值为 Config.BETA_START
        self.current_entropy_beta = Config.BETA_START

    def learn(self, list_sample_data):
        """
        list_sample_data: list[SampleData]
        SampleData对象列表
        """
        
        # =========================================================
        # 【高阶经验 1】：熵正则化系数 (Entropy Beta) 指数衰减与强制保底
        # 放弃基线中直接归零的线性衰减。MOBA 游戏状态空间极大，必须保证终身探索。
        # =========================================================
        decay_rate = 0.99995  # 指数平滑衰减率
        min_beta = max(0.005, Config.BETA_END) # 强制保持至少 0.5% 的探索底线
        self.current_entropy_beta = max(min_beta, self.current_entropy_beta * decay_rate)
        self.model.var_beta = self.current_entropy_beta

        # 从 SampleData 对象中提取 sample 字段并 stack 成 tensor
        _input_datas = torch.stack([sample.sample for sample in list_sample_data]).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        # =========================================================
        # 【高阶经验 2】：Batch-level 优势函数归一化 (Advantage Normalization)
        # 在送入 PPO 计算之前，对整个 Batch 的 Advantage 进行归一化。
        # 此举能化解 last_hit 等累积奖励项导致的不同批次间回报尺度不一致的问题。
        # =========================================================
        # data_list[2] 根据 Config.DATA_SPLIT_SHAPE 对齐的是 Advantage
        adv = data_list[2]
        if adv.shape[0] > 1:
            # 加上 1e-8 防止除 0 崩溃
            adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
            data_list[2] = adv_normalized

        # 序列化特征切分
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        # 模型前向传播计算 loss
        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # =========================================================
        # 梯度裁剪 (Gradient Clipping)
        # 配合上面的 Advantage 归一化，构成 PPO 稳定训练的双保险
        # =========================================================
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        # 更新学习率
        self.scheduler.step(self.train_step)

        # 指标处理与上报
        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now