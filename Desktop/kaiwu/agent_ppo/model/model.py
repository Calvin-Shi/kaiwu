#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
import torch.nn as nn
from torch.nn import ModuleDict

import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal actions only. mask: 1=legal, 0=illegal."""
    logits = logits + (mask.float() - 1.0) * 1e9
    return torch.log_softmax(logits, dim=-1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over legal actions only. mask: 1=legal, 0=illegal."""
    logits = logits + (mask.float() - 1.0) * 1e9
    return torch.softmax(logits, dim=-1)


def masked_categorical_sample(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sample from categorical distribution over legal actions."""
    probs = masked_softmax(logits, mask)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# === 【高阶经验 1】：增加 gain 参数，控制不同输出层的初始分布 ===
def make_fc_layer(in_features: int, out_features: int, use_bias=True, gain=np.sqrt(2)):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    # 使用带有 gain 的正交初始化
    nn.init.orthogonal_(fc_layer.weight, gain=gain)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


# === 【高阶经验 4 (预留)】：实体注意力机制 ===
# 用于后续特征工程升级时，让英雄动态"盯防"敌方英雄、残血小兵或防御塔
class EntityAttention(nn.Module):
    def __init__(self, hero_dim, entity_dim, embed_dim=64, num_heads=4):
        super().__init__()
        self.hero_mlp = make_fc_layer(hero_dim, embed_dim, gain=np.sqrt(2))
        self.entity_mlp = make_fc_layer(entity_dim, embed_dim, gain=np.sqrt(2))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, hero_feat, entity_seq):
        # hero_feat: [Batch, hero_dim] 
        # entity_seq: [Batch, N, entity_dim] 
        q = self.hero_mlp(hero_feat).unsqueeze(1) # [Batch, 1, embed_dim]
        k = self.entity_mlp(entity_seq)           # [Batch, N, embed_dim]
        # 注意力计算：用英雄的特征去 Query 所有小兵的特征
        attn_out, _ = self.attn(q, k, k)
        return attn_out.squeeze(1)                # [Batch, embed_dim]
# === 【高阶经验 3 的究极进化】：真正的目标锁定注意力机制 ===
class TargetAttentionHead(nn.Module):
    def __init__(self, query_dim, entity_dim, embed_dim=64):
        super().__init__()
        # Query 映射：把英雄的战斗意图映射到注意力空间
        self.query_mlp = make_fc_layer(query_dim, embed_dim, gain=np.sqrt(2))
        # Key 映射：共享权重！把 9 个候选目标的特征映射到注意力空间
        self.key_mlp = make_fc_layer(entity_dim, embed_dim, gain=np.sqrt(2))

    def forward(self, query_feat, target_seq):
        # query_feat: [Batch, query_dim]   -> 通常是 combat_branch 的输出 (意图)
        # target_seq: [Batch, 9, entity_dim] -> 9个候选目标的特征序列
        
        q = self.query_mlp(query_feat).unsqueeze(1)    # [Batch, 1, embed_dim]
        k = self.key_mlp(target_seq)                   # [Batch, 9, embed_dim]

        # 内积计算注意力打分 (Dot-product Attention)
        # 意图(q) 去匹配 每一个目标的特征(k)
        # [Batch, 1, embed_dim] x [Batch, embed_dim, 9] -> [Batch, 1, 9]
        attn_logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1)
        
        # 缩放因子，防止梯度消失/爆炸
        attn_logits = attn_logits / np.sqrt(k.size(-1))
        
        # 输出的就是 9 个目标的 logits 分数
        return attn_logits


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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

        # 网络维度
        self.hero_data_len = sum(Config.data_shapes[0])
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])
        
        # ========================================================
        # 【高阶经验 2】：Actor 与 Critic 彻底解耦
        # ========================================================
        
        # ========================================================
        # 【注意力改造 1】：初始化 Attention 模块与维度
        # ========================================================
        # 假设基础特征：12(我方) + 7(塔) + 10(战术雷达) = 29维
        self.base_feat_dim = 30
        # 小兵特征：每个小兵5维
        self.entity_dim = 7
        self.attn_embed_dim = 64
        
        self.entity_attention = EntityAttention(
            hero_dim=self.base_feat_dim, 
            entity_dim=self.entity_dim, 
            embed_dim=self.attn_embed_dim,
            num_heads=4
        )

        # 融合后的总特征维度 = 基础29维 + Attention提取的64维 = 93维
        self.fused_dim = self.base_feat_dim + self.attn_embed_dim

        # ========================================================
        # 【注意力改造 2】：将 Actor 和 Critic 的输入维度改为 LSTM 输出维度
        # ========================================================
        # 注意：这里我们让网络真正利用上 LSTM 的时序记忆！
        self.critic_backbone = nn.Sequential(
            make_fc_layer(self.lstm_unit_size, 256), nn.ReLU(),  # <-- 改为 lstm_unit_size (512)
            make_fc_layer(256, 256), nn.ReLU()
        )
        self.value_head = make_fc_layer(256, 1, gain=1.0)

        self.actor_shared = nn.Sequential(
            make_fc_layer(self.lstm_unit_size, 256), nn.ReLU()   # <-- 改为 lstm_unit_size (512)
        )

        # 3.1 决策分支 (选择按哪个键)
        self.button_branch = nn.Sequential(make_fc_layer(256, 128), nn.ReLU())
        self.head_button = make_fc_layer(128, self.label_size_list[0], gain=0.01)

        # 3.2 走位分支 (Move X, Move Z) -> 负责连续的拉扯
        self.move_branch = nn.Sequential(make_fc_layer(256, 128), nn.ReLU())
        self.head_move_x = make_fc_layer(128, self.label_size_list[1], gain=0.01)
        self.head_move_z = make_fc_layer(128, self.label_size_list[2], gain=0.01)

        # 3.3 战斗与施法分支 (Skill X, Skill Z, Target)
        self.combat_branch = nn.Sequential(make_fc_layer(256, 128), nn.ReLU())
        self.head_skill_x = make_fc_layer(128, self.label_size_list[3], gain=0.01)
        self.head_skill_z = make_fc_layer(128, self.label_size_list[4], gain=0.01)
        
        # 【究极进化】：Target Attention 的 Query 维度改为 LSTM 输出维度！
        self.target_attention = TargetAttentionHead(
            query_dim=self.lstm_unit_size,   # <--- 直接使用 LSTM 记忆作为 Query！
            entity_dim=self.entity_dim,      # 实体特征维度 (7维)
            embed_dim=64
        )

        # 修复 LSTM 定义：输入维度应该是融合后的特征维度 94 (30维基础 + 64维Entity Attention)
        self.lstm = torch.nn.LSTM(
            input_size=self.fused_dim, hidden_size=self.lstm_unit_size, num_layers=1,
            bias=True, batch_first=True, dropout=0, bidirectional=False,
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)
        self.target_embed_mlp = make_fc_layer(self.target_embed_dim, self.target_embed_dim, use_bias=False)

    def forward(self, data_list, inference=False, legal_action=None):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        # ========================================================
        # 【注意力改造 3】：动态特征切片与 Attention 前向传播
        # ========================================================
        # 1. 拆分基础特征与实体特征
        base_feat = feature_vec[:, :self.base_feat_dim]             # [Batch*T, 30]
        npc_flat = feature_vec[:, self.base_feat_dim:]              # [Batch*T, 63]
        
        npc_seq = npc_flat.view(-1, 9, self.entity_dim)             # [Batch*T, 9, 7]

        # 2. 提取注意力融合特征
        attn_out = self.entity_attention(base_feat, npc_seq)        # [Batch*T, 64]
        fused_feat = torch.cat([base_feat, attn_out], dim=1)        # [Batch*T, 94]

        # ========================================================
        # 【修复与升级】：彻底激活 LSTM 时序网络！
        # ========================================================
        N = fused_feat.size(0)
        T = self.lstm_time_steps
        B = N // T
        
        # 将展平的帧序列 reshape 成 LSTM 需要的 (Batch, Time, Dim)
        fused_seq = fused_feat.view(B, T, -1)
        
        h0 = lstm_hidden_init.unsqueeze(0) # [1, B, H]
        c0 = lstm_cell_init.unsqueeze(0)   # [1, B, H]
        
        # 通过 LSTM，获取具有时序记忆的特征
        lstm_out, (h_n, c_n) = self.lstm(fused_seq, (h0, c0))
        
        # 必须正确保存隐藏状态，供下一步推理时使用！(修复了你原本代码直接返回 init 状态的 bug)
        self.lstm_hidden_output = h_n  
        self.lstm_cell_output = c_n
        
        # 展平 LSTM 输出，交给全连接层 [Batch*T, 512]
        lstm_out_flat = lstm_out.contiguous().view(N, -1)  

        # --- Critic 评估分支 ---
        v_feat = self.critic_backbone(lstm_out_flat)       # <-- 传入 LSTM 记忆
        value_result = self.value_head(v_feat)

        # --- Actor 决策分支 ---
        a_feat = self.actor_shared(lstm_out_flat)          # <-- 传入 LSTM 记忆

        # 决策
        b_feat = self.button_branch(a_feat)
        logit_button = self.head_button(b_feat)

        # 走位
        m_feat = self.move_branch(a_feat)
        logit_move_x = self.head_move_x(m_feat)
        logit_move_z = self.head_move_z(m_feat)

        # 战斗
        c_feat = self.combat_branch(a_feat)
        logit_skill_x = self.head_skill_x(c_feat)
        logit_skill_z = self.head_skill_z(c_feat)
        
        # 【神级走A与锁敌】：用 LSTM 的输出 (lstm_out_flat) 直接作为 Target Attention 的 Query！
        # 你的射手 AI 现在能根据上一秒敌人的移动轨迹，以及刚被打掉的血量，稳健地锁定同一个目标。
        logit_target = self.target_attention(lstm_out_flat, npc_seq)

        # 组装返回列表 (必须与 LABEL_SIZE_LIST 的顺序严格一致)
        result_list = [
            logit_button,
            logit_move_x,
            logit_move_z,
            logit_skill_x,
            logit_skill_z,
            logit_target,
            value_result
        ]

        # 下面的 Inference 推理掩码逻辑保持原样，完美兼容框架
        if inference:
            if legal_action is not None:
                la_splits = torch.split(legal_action, self.legal_action_size, dim=1)
            else:
                la_splits = [torch.ones_like(result_list[i]) for i in range(len(self.label_size_list))]

            action_list, d_action_list, prob_list, d_prob_list = [], [], [], []

            for i in range(len(self.label_size_list) - 1):
                mask = la_splits[i]
                logit = result_list[i]
                probs = masked_softmax(logit, mask)
                action = masked_categorical_sample(logit, mask)
                d_action = torch.argmax(probs, dim=-1)
                
                action_list.append(action)
                d_action_list.append(d_action)
                prob_list.append(probs)
                d_prob_list.append(probs)

            # Last head (target): mask filtered by chosen button action
            last_logit = result_list[-2]  
            n_button = self.label_size_list[0]
            n_target = self.label_size_list[-1]

            if legal_action is not None:
                full_target_mask = legal_action[:, sum(self.label_size_list[:-1]):]
                full_target_mask = full_target_mask.reshape(-1, n_button, n_target)

                btn_idx = action_list[0]  
                btn_onehot = torch.zeros(btn_idx.shape[0], n_button, device=btn_idx.device)
                btn_onehot.scatter_(1, btn_idx.unsqueeze(1), 1.0)
                target_mask = (full_target_mask * btn_onehot.unsqueeze(-1)).sum(dim=1)

                d_btn_idx = d_action_list[0]
                d_btn_onehot = torch.zeros(d_btn_idx.shape[0], n_button, device=d_btn_idx.device)
                d_btn_onehot.scatter_(1, d_btn_idx.unsqueeze(1), 1.0)
                d_target_mask = (full_target_mask * d_btn_onehot.unsqueeze(-1)).sum(dim=1)
            else:
                target_mask = torch.ones(feature_vec.shape[0], n_target, device=feature_vec.device)
                d_target_mask = target_mask

            target_probs = masked_softmax(last_logit, target_mask)
            target_action = masked_categorical_sample(last_logit, target_mask)
            d_target_probs = masked_softmax(last_logit, d_target_mask)
            d_target_action = torch.argmax(d_target_probs, dim=-1)

            action_list.append(target_action)
            d_action_list.append(d_target_action)
            prob_list.append(target_probs)
            d_prob_list.append(d_target_probs)

            flat_prob = torch.cat(prob_list, dim=1)
            flat_d_prob = torch.cat(d_prob_list, dim=1)
            logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
            value = result_list[-1]

            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output,
                    action_list, d_action_list, flat_prob, flat_d_prob]
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

        legal_action_flag_list = list(torch.split(feature_legal_action, self.label_size_list, dim=1))

        # Value loss
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        label_probability_list = []
        epsilon = 1e-5

        # Policy loss
        self.policy_cost = torch.tensor(0.0)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                mask = legal_action_flag_list[task_index]
                logit = label_result[task_index]
                one_hot_actions = nn.functional.one_hot(
                    label_list[task_index].long(), self.label_size_list[task_index]
                ).float()

                label_probability = masked_softmax(logit, mask)
                label_probability = label_probability * mask + self.min_policy * mask
                label_probability = label_probability / label_probability.sum(1, keepdim=True).clamp(min=epsilon)
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)

                surr1 = ratio.clamp(0.0, 3.0) * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0)
                )
                self.policy_cost = self.policy_cost + temp_policy_loss

        # Entropy loss
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                prob = label_probability_list[current_entropy_loss_index]
                mask = legal_action_flag_list[task_index]
                temp_entropy_loss = -torch.sum(
                    prob * mask * torch.log(prob + epsilon),
                    dim=1,
                )
                temp_entropy_loss = -torch.sum(
                    temp_entropy_loss * weight_list[task_index].float() * frame_is_train
                ) / torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0)
                )
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0))

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

# 保留 MLP 类以防你的其他外围脚本调用
class MLP(nn.Module):
    def __init__(
        self, fc_feat_dim_list: List[int], name: str,
        non_linearity: nn.Module = nn.ReLU, non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)