#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
import torch
import torch.nn as nn
from torch.nn import ModuleDict
import torch.nn.functional as F

import numpy as np
from math import ceil, floor
from collections import OrderedDict
from typing import Dict, List, Tuple

from agent_ppo.conf.conf import DimConfig, Config

HERO_DIM = 80
ORGAN_DIM = 26
NPC_DIM = 63
ATTN_D = 32  # attention中间维度

PER_HERO_DIM = 40
PER_ORGAN_DIM = 7
PER_NPC_DIM = 7
PER_CAKE_DIM = 6

HERO_COUNT = 1
ORGAN_COUNT = 1
NPC_COUNT = 4
MONSTER_COUNT = 1

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.head_in_dim = self.lstm_unit_size + self.target_embed_dim

        # NETWORK DIM
        # 网络维度
        self.hero_data_len = sum(Config.data_shapes[0])
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        # fc_concat_dim_list = [self.feature_dim, 256, 256]
        # self.concat_mlp = MLP(fc_concat_dim_list, "concat_mlp", non_linearity_last=True)


        # ---------- HERO: 40 -> trunk(256) -> {frd, emy} -> 32 ----------
        self.hero_trunk = MLP([PER_HERO_DIM, 512, 256], "hero_trunk", non_linearity_last=True)
        self.hero_frd_fc = make_fc_layer(256, ATTN_D)
        self.hero_emy_fc = make_fc_layer(256, ATTN_D)

        # ---------- SOLDIER (NPC 7维): 7 -> trunk(64) -> {frd, emy} -> 32 ----------
        self.soldier_trunk = MLP([PER_NPC_DIM, 64, 64], "soldier_trunk", non_linearity_last=True)
        self.soldier_frd_fc = make_fc_layer(64, ATTN_D)
        self.soldier_emy_fc = make_fc_layer(64, ATTN_D)

        # ---------- TOWER (ORGAN 7维): 7 -> trunk(64) -> {frd, emy} -> 32 ----------
        self.tower_trunk = MLP([PER_ORGAN_DIM, 64, 64], "tower_trunk", non_linearity_last=True)
        self.tower_frd_fc = make_fc_layer(64, ATTN_D)
        self.tower_emy_fc = make_fc_layer(64, ATTN_D)

        # ---------- CAKE / HP (6维): 6 -> trunk(32) -> {frd, emy} -> 32 ----------
        self.cake_trunk = MLP([PER_CAKE_DIM, 32, 32], "cake_trunk", non_linearity_last=True)
        self.cake_frd_fc = make_fc_layer(32, ATTN_D)
        self.cake_emy_fc = make_fc_layer(32, ATTN_D)

        # ---------- MONSTER (中立 7维): 7 -> trunk(64) -> fc -> 32 ----------
        # 复用 soldier 的干路，单独一个 neutral fc
        self.monster_fc = make_fc_layer(64, ATTN_D)


        self.lstm = torch.nn.LSTM(
            input_size=9*ATTN_D, # 288
            hidden_size=self.lstm_unit_size,#512
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.lstm_proj = make_fc_layer(self.lstm_unit_size, 256)

        self.label_mlp = ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP(
                    [self.head_in_dim, self.label_size_list[label_index]],
                    f"hero_label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list) - 1)  # 非 target 头
            }
        )
        # 新增：target 专用头，输出维与最后一个 label_size 对齐（例如 9）
        self.target_head = MLP(
            [self.head_in_dim, self.label_size_list[-1]],
            "target_head"
        )

        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)

        self.value_mlp = MLP([256, 256, 1], "hero_value_mlp")

        self.target_embed_mlp = make_fc_layer(ATTN_D, self.target_embed_dim, use_bias=False)

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        device = feature_vec.device


        SERI_DIM =Config.SERI_VEC_SPLIT_SHAPE[0][0]  
        LEGAL_DIM = Config.SERI_VEC_SPLIT_SHAPE[1][0]  # 85
        ALL_DIM  = SERI_DIM + LEGAL_DIM  

        T_cfg = self.lstm_time_steps                # 训练: >1，推理: 1（set_eval_mode 会改）

        if feature_vec.dim() != 2:
            raise RuntimeError(f"feature_vec must be 2D, got {feature_vec.shape}")

        N, D = feature_vec.shape  # N 是“当前张量的行数”，可能是 B 或 B*T
            
        # 每帧一行 -> (B*T, D)
        if D in (SERI_DIM, ALL_DIM) and (N % T_cfg == 0):
            B = N // T_cfg
            T = T_cfg
            if D == ALL_DIM:
                feature_seq = feature_vec.reshape(B, T, ALL_DIM)
                seri_seq    = feature_seq[:, :, :SERI_DIM]
            else:
                seri_seq = feature_vec.reshape(B, T, SERI_DIM)
        else:
            raise RuntimeError(
                f"Unsupported feature_vec shape {feature_vec.shape}. "
                f"Each row must be SERI_DIM({SERI_DIM}) or ALL_DIM({ALL_DIM}), "
                f"or a multiple thereof."
            )

        
        start = 0
        # hero 40友+40敌
        hero_friend = seri_seq[:, :, start : start + PER_HERO_DIM * HERO_COUNT] ;start += PER_HERO_DIM * HERO_COUNT
        hero_enemy  = seri_seq[:, :, start : start + PER_HERO_DIM * HERO_COUNT] ;start += PER_HERO_DIM * HERO_COUNT

        # organ 7友+7敌塔 + 6友包 + 6敌包
        tower_friend = seri_seq[:, :, start : start + PER_ORGAN_DIM * ORGAN_COUNT] ;start += PER_ORGAN_DIM * ORGAN_COUNT
        tower_enemy  = seri_seq[:, :, start : start + PER_ORGAN_DIM * ORGAN_COUNT] ;start += PER_ORGAN_DIM * ORGAN_COUNT
        cake_friend  = seri_seq[:, :, start : start + PER_CAKE_DIM * ORGAN_COUNT]  ;start += PER_CAKE_DIM * ORGAN_COUNT
        cake_enemy   = seri_seq[:, :, start : start + PER_CAKE_DIM * ORGAN_COUNT]  ;start += PER_CAKE_DIM * ORGAN_COUNT

        # npc（友兵4×7 + 敌兵4×7 + 野怪1×7）
        npc_friend   = seri_seq[:, :, start : start + PER_NPC_DIM * NPC_COUNT] ;start += PER_NPC_DIM * NPC_COUNT
        npc_enemy    = seri_seq[:, :, start : start + PER_NPC_DIM * NPC_COUNT] ;start += PER_NPC_DIM * NPC_COUNT
        npc_monster  = seri_seq[:, :, start : start + PER_NPC_DIM * MONSTER_COUNT] ;start += PER_NPC_DIM * MONSTER_COUNT

        B_eff, T_eff = seri_seq.size(0), seri_seq.size(1)

        # 共享编码 + 类内池化
        def enc_pool_split(x_btcd, trunk_mlp, fc_layer, feat_dim):
            # x_btcd: [B,T,C*feat_dim] or [B,T,C,feat_dim]
            if x_btcd.dim() == 3:
                C = x_btcd.size(-1) // feat_dim
                x_btcd = x_btcd.reshape(B_eff, T_eff, C, feat_dim)
            B, T, C, D = x_btcd.shape
            trunk = trunk_mlp(x_btcd.reshape(B*T*C, D))              # [B*T*C, trunk_dim]
            emb   = fc_layer(trunk).reshape(B, T, C, ATTN_D)          # [B,T,C,32]
            pooled, _ = emb.max(dim=2)                                # [B,T,32]
            return emb, pooled

        # HERO
        h_emb_f, h_pool_f = enc_pool_split(hero_friend, self.hero_trunk,   self.hero_frd_fc,  PER_HERO_DIM)
        h_emb_e, h_pool_e = enc_pool_split(hero_enemy,  self.hero_trunk,   self.hero_emy_fc,  PER_HERO_DIM)

        # TOWER
        t_emb_f, t_pool_f = enc_pool_split(tower_friend, self.tower_trunk,  self.tower_frd_fc, PER_ORGAN_DIM)
        t_emb_e, t_pool_e = enc_pool_split(tower_enemy,  self.tower_trunk,  self.tower_emy_fc, PER_ORGAN_DIM)

        # CAKE (HP)
        c_emb_f, c_pool_f = enc_pool_split(cake_friend,  self.cake_trunk,   self.cake_frd_fc,  PER_CAKE_DIM)
        c_emb_e, c_pool_e = enc_pool_split(cake_enemy,   self.cake_trunk,   self.cake_emy_fc,  PER_CAKE_DIM)

        # SOLDIER
        s_emb_f, s_pool_f = enc_pool_split(npc_friend,   self.soldier_trunk, self.soldier_frd_fc, PER_NPC_DIM)  # C=4
        s_emb_e, s_pool_e = enc_pool_split(npc_enemy,    self.soldier_trunk, self.soldier_emy_fc, PER_NPC_DIM)  # C=4

        # MONSTER（中立，无阵营）
        # 先过 soldier_trunk，再过 monster_fc
        def enc_pool_neutral(x_btcd, trunk_mlp, fc_layer, feat_dim):
            if x_btcd.dim() == 3:
                C = x_btcd.size(-1) // feat_dim
                x_btcd = x_btcd.reshape(B_eff, T_eff, C, feat_dim)
            B, T, C, D = x_btcd.shape
            trunk = trunk_mlp(x_btcd.reshape(B*T*C, D))
            emb   = fc_layer(trunk).reshape(B, T, C, ATTN_D)
            pooled, _ = emb.max(dim=2)
            return emb, pooled

        m_emb, m_pool = enc_pool_neutral(npc_monster, self.soldier_trunk, self.monster_fc, PER_NPC_DIM)
       


        # LSTM 输入：9 路池化拼接（每路32维 = 288）
        frame_feat = torch.cat([
            h_pool_f, h_pool_e,
            t_pool_f, t_pool_e,
            c_pool_f, c_pool_e,
            s_pool_f, s_pool_e,
            m_pool
        ], dim=-1)  # [B,T,288]



        # —— 初态 batch 大小校验，避免静默错位 —— 
        if lstm_hidden_init.size() != (B_eff, self.lstm_unit_size) or \
        lstm_cell_init.size()  != (B_eff, self.lstm_unit_size):
            raise RuntimeError(
                f"LSTM initial state shape mismatch. "
                f"Expect (B={B_eff}, H={self.lstm_unit_size}), "
                f"got h={tuple(lstm_hidden_init.size())}, c={tuple(lstm_cell_init.size())}"
            )
        
        # 把外部传来的 (hidden, cell) 当作初态喂给 LSTM 
        h0 = lstm_hidden_init.unsqueeze(0)   # [1, B, H]
        c0 = lstm_cell_init.unsqueeze(0)     # [1, B, H]

        # 带初态调用 LSTM
        lstm_out, state = self.lstm(frame_feat, (h0, c0))    # lstm_out: [B,T,H], state=(h_T,c_T)

        # 保存终态，供推理返回
        self.lstm_hidden_output = state[0]   # h_T: [1,B,H]
        self.lstm_cell_output   = state[1]   # c_T: [1,B,H]

        # 逐时间步输出：把时间维展开
        lstm_bt = lstm_out.reshape(-1, self.lstm_unit_size)                    # [B*T,512]
        fc_public_result = self.lstm_proj(lstm_bt)                             # [B*T,256]

        result_list = []

        BT = lstm_bt.size(0)

        # 组装target候选 敌英雄1 + 敌兵4 + 敌塔1 + 我血包1 + 敌血包1 + 野怪1 = 9
        cands = [h_emb_e, # 敌英雄 [B,T,1,32]
                 s_emb_e, # 敌兵 [B,T,4,32]
                 t_emb_e, # 敌塔 [B,T,1,32]
                 c_emb_f, # 我血包 [B,T,1,32]
                 c_emb_e, # 敌血包 [B,T,1,32]
                 m_emb,   # 野怪 [B,T,1,32]
        ]  # 6类

        #展平为 [B*T, K, 32]
        keys = torch.cat([x.reshape(B_eff*T_eff, -1, ATTN_D) for x in cands], dim=1)  # [B*T, K, 32]
        K = keys.size(1)

        # query：LSTM投到32维
        query = self.lstm_tar_embed_mlp(lstm_bt) # [B*T, 32]

        # target logits 
        atten_logits = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1) / (math.sqrt(ATTN_D))  # [B*T, K]
        atten = F.softmax(atten_logits, dim=1)  # [B*T, K]
        context = torch.bmm(atten.unsqueeze(1), keys).squeeze(1)  # [B*T, 32]

        # 4) 断言配置一致（很重要）
        assert self.label_size_list[-1] == K, \
            f"Config.LABEL_SIZE_LIST[-1] ({self.label_size_list[-1]}) must equal K={K}"


        # 非 target 的所有动作头：输入 = [h, context]
        head_in = torch.cat([lstm_bt, context], dim=1)                 # [BT, H+D]
        for label_index, label_dim in enumerate(self.label_size_list[:-1]):  # 非 target 头
            head_logits = self.label_mlp[f"hero_label{label_index}_mlp"](head_in)
            result_list.append(head_logits)

        # target 头：直接用注意力打分作为 logits 
        result_list.append(atten_logits)

        # 输出价值
        value_result = self.value_mlp(fc_public_result)
        result_list.append(value_result)


        # 准备推理图
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:

            B = logits.size(0)

            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list


    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]

        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # loss of value net
        # 值网络的损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        # for entropy loss calculate
        # 用于熵损失计算
        label_logits_subtract_max_list = []
        label_sum_exp_logits_list = []
        label_probability_list = []

        epsilon = 1e-5


        # 策略损失：PPO剪辑损失
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0)
                boundary = torch.pow(torch.tensor(10.0), torch.tensor(20.0))
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_logits_subtract_max_list.append(label_logits_subtract_max)

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)
                label_sum_exp_logits_list.append(label_sum_exp_logits)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)
                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * 1
                ) / torch.maximum(torch.sum((weight_list[task_index].float()) * 1), torch.tensor(1.0))

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * 1)
                ) / torch.maximum(torch.sum(weight_list[task_index].float() * 1), torch.tensor(1.0))

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Wrapper function to create and initialize a linear layer

    Args:
        in_features (int): ``in_features``
        out_features (int): ``out_features``

    Returns:
        nn.Linear: the initialized linear layer
    """
    """ 创建并初始化线性层的包装函数

    参数:
        in_features (int): 输入特征数
        out_features (int): 输出特征数

    返回:
        nn.Linear: 初始化的线性层
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        """Create a MLP object

        Args:
            fc_feat_dim_list (List[int]): ``in_features`` of the first linear layer followed by
                ``out_features`` of each linear layer
            name (str): human-friendly name, serving as prefix of each comprising layers
            non_linearity (nn.Module, optional): the activation function to use. Defaults to nn.ReLU.
            non_linearity_last (bool, optional): whether to append a activation function in the end.
                Defaults to False.
        """
        """ 创建一个MLP对象

        参数:
            fc_feat_dim_list (List[int]): 第一个线性层的输入特征数，后续每个线性层的输出特征数
            name (str): 人类友好的名称，作为每个组成层的前缀
            non_linearity (nn.Module, optional): 要使用的激活函数。默认为 nn.ReLU。
            non_linearity_last (bool, optional): 是否在最后附加一个激活函数。默认为 False。
        """
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
