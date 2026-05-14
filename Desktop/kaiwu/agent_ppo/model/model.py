#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

精简重构版网络 + Target 注意力机制 + 伪自回归 Target Head + Twin Critics。
Feature(145) → FC(256→256) → 2-layer LSTM(256→256) → Actor/Critic split
Target head: Query-Key Attention + Button-Conditioned Head (P0修复)。
Twin Critics: 双价值网络减小价值估计方差 (Double-Q style)。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config


# ---------------------------------------------------------------------------
# Action sampling utilities
# ---------------------------------------------------------------------------

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = logits + (mask.float() - 1.0) * 1e9
    return torch.softmax(logits, dim=-1)


def masked_categorical_sample(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = masked_softmax(logits, mask)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Weighted FC layer
# ---------------------------------------------------------------------------

def make_fc_layer(in_features: int, out_features: int, use_bias=True, gain=np.sqrt(2)):
    fc = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc.weight, gain=gain)
    if use_bias:
        nn.init.zeros_(fc.bias)
    return fc


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- config ----
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE      # 512
        self.label_size_list = Config.LABEL_SIZE_LIST      # [12, 16, 16, 16, 16, 9]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = Config.BETA_START
        self.learning_rate = Config.INIT_LEARNING_RATE_START
        self.lstm_time_steps = Config.LSTM_TIME_STEPS

        FEAT_DIM = 145
        HIDDEN = 256
        ATTN_D = 64  # attention embedding dimension
        self.button_embed_dim = Config.BUTTON_EMBED_DIM  # 64

        # ---- 1. Feature embedding (145 → 256 → 256) ----
        self.feature_embed = nn.Sequential(
            make_fc_layer(FEAT_DIM, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )

        # ---- 2. LSTM (256 → 256), 2 layers ----
        self.lstm_num_layers = 2
        self.lstm_per_layer = HIDDEN
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, num_layers=self.lstm_num_layers, batch_first=True)

        # ---- 3. Actor shared backbone (256 → 256 → 256) ----
        self.actor_shared = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )

        # ---- 4. Twin Critics (Double-Q style, reduces value overestimation) ----
        # Critic 1
        self.critic_backbone_1 = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.critic_context_fusion_1 = make_fc_layer(HIDDEN + ATTN_D, HIDDEN)
        self.value_head_1 = make_fc_layer(HIDDEN, 1, gain=1.0)
        # Critic 2
        self.critic_backbone_2 = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.critic_context_fusion_2 = make_fc_layer(HIDDEN + ATTN_D, HIDDEN)
        self.value_head_2 = make_fc_layer(HIDDEN, 1, gain=1.0)

        # ---- 5. Action heads (256 → label_size) ----
        self.head_button  = make_fc_layer(HIDDEN, self.label_size_list[0], gain=0.01)
        self.head_move_x  = make_fc_layer(HIDDEN, self.label_size_list[1], gain=0.01)
        self.head_move_z  = make_fc_layer(HIDDEN, self.label_size_list[2], gain=0.01)
        self.head_skill_x = make_fc_layer(HIDDEN, self.label_size_list[3], gain=0.01)
        self.head_skill_z = make_fc_layer(HIDDEN, self.label_size_list[4], gain=0.01)

        # ---- 6. Target attention: Key embeddings for 9 candidate targets ----
        # Targets (per action space spec):
        #   0=None  1=EnemyHero  2=Self  3-6=Soldiers×4  7=Tower  8=Resource(cake)
        # P1: none_key 用正交随机初始化替代全零，避免初始注意力度偏向
        self.none_key = nn.Parameter(torch.randn(1, ATTN_D) * 0.01)
        self.emy_hero_key = make_fc_layer(32, ATTN_D)     # enemy hero: 32 dims
        self.self_key = make_fc_layer(32, ATTN_D)          # self: 32 dims
        self.soldier_key = make_fc_layer(7, ATTN_D)        # shared for 4 soldiers
        self.tower_key = make_fc_layer(7, ATTN_D)          # tower: 7 dims
        # P2: resource_key 替代 monster_key，明确特征来源是蛋糕/血包
        self.resource_key = make_fc_layer(7, ATTN_D)       # resource(cake): 7 dims

        # ---- 7. Query projection: LSTM output → query embedding ----
        self.target_query = make_fc_layer(HIDDEN, ATTN_D)

        # ---- 8. Context fusion: inject attention context into actor features ----
        self.context_fusion = make_fc_layer(HIDDEN + ATTN_D, HIDDEN)

        # ---- 9. P0: 伪自回归 Target Head ----
        # button_embed_mlp: 将 Button one-hot 编码为条件向量
        # head_target: 条件化的目标选择 (a_feat_fused ⊕ button_embed → target_logits)
        self.button_embed_mlp = nn.Sequential(
            make_fc_layer(self.label_size_list[0], self.button_embed_dim),
            nn.ReLU(),
        )
        self.head_target = make_fc_layer(
            HIDDEN + self.button_embed_dim, self.label_size_list[5], gain=0.01
        )

        # 缓存 a_feat_fused 供 compute_loss 用真实 button 重算 target
        self.cached_a_feat_fused = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data_list, inference=False, legal_action=None):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        B_flat = feature_vec.shape[0]
        N = self.lstm_num_layers
        H = self.lstm_per_layer
        ATTN_D = 64

        # ---- 1. Feature embedding ----
        embed = self.feature_embed(feature_vec)          # [B_flat, 256]

        # ---- 2. LSTM step (2 layers) ----
        if not inference:
            T = self.lstm_time_steps                     # e.g. 16
            B_real = B_flat // T
            feat_3d = embed.reshape(B_real, T, H)       # [B, T, 256]
            h0 = lstm_hidden_init.reshape(N, B_real, H) # [2, B, 256]
            c0 = lstm_cell_init.reshape(N, B_real, H)   # [2, B, 256]
            lstm_out, (hn, cn) = self.lstm(feat_3d, (h0, c0))
            lstm_feat = lstm_out.reshape(B_flat, H)     # [B*T, 256]
            lstm_cell_out = cn.reshape(B_real, N * H)   # [B, 512]
            lstm_hidden_out = hn.reshape(B_real, N * H) # [B, 512]
        else:
            lstm_in = embed.unsqueeze(1)                 # [B, 1, 256]
            h0 = lstm_hidden_init.reshape(N, B_flat, H) # [2, B, 256]
            c0 = lstm_cell_init.reshape(N, B_flat, H)   # [2, B, 256]
            lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))
            lstm_feat = lstm_out.squeeze(1)              # [B, 256]
            lstm_cell_out = cn.reshape(B_flat, N * H)   # [B, 512]
            lstm_hidden_out = hn.reshape(B_flat, N * H) # [B, 512]

        # ---- 3. Actor shared ----
        a_feat = self.actor_shared(lstm_feat)            # [B_flat, 256]

        # ---- 4. Target attention ----
        # Feature layout [B_flat, 145]:
        #   [0:32]   = Self (friendly hero, 32)
        #   [32:64]  = Enemy hero (32)
        #   [64:71]  = Organ (enemy tower, 7)
        #   [71:82]  = Tactical (11)
        #   [82:110] = Friendly soldiers (4×7)
        #   [110:138]= Enemy soldiers (4×7)
        #   [138:145]= Cake/resource (1×7)

        self_feat        = feature_vec[:, 0:32]            # [B_flat, 32]
        emy_hero_feat    = feature_vec[:, 32:64]           # [B_flat, 32]
        emy_tower_feat   = feature_vec[:, 64:71]           # [B_flat,  7]
        emy_soldier_feat = feature_vec[:, 110:138].reshape(B_flat, 4, 7)  # [B_flat, 4, 7]
        resource_feat    = feature_vec[:, 138:145]         # [B_flat,  7]

        # Build 9 key tensors
        none_k      = self.none_key.expand(B_flat, 1, ATTN_D)                    # [B_flat, 1, 64] Target 0
        emy_hero_k  = self.emy_hero_key(emy_hero_feat).unsqueeze(1)               # [B_flat, 1, 64] Target 1
        self_k      = self.self_key(self_feat).unsqueeze(1)                       # [B_flat, 1, 64] Target 2

        # 4 soldiers share embedding weight (Target 3-6)
        emy_soldier_k = self.soldier_key(
            emy_soldier_feat.reshape(B_flat * 4, 7)
        ).reshape(B_flat, 4, ATTN_D)                                              # [B_flat, 4, 64]

        tower_k     = self.tower_key(emy_tower_feat).unsqueeze(1)                 # [B_flat, 1, 64] Target 7
        resource_k  = self.resource_key(resource_feat).unsqueeze(1)               # [B_flat, 1, 64] Target 8

        keys = torch.cat([
            none_k,          # Target 0: None
            emy_hero_k,      # Target 1: Enemy hero
            self_k,          # Target 2: Self
            emy_soldier_k,   # Target 3-6: Soldiers ×4
            tower_k,         # Target 7: Tower
            resource_k,      # Target 8: Resource (cake/blood)
        ], dim=1)                                                                 # [B_flat, 9, 64]

        # Query: LSTM output → query embedding
        query = self.target_query(lstm_feat)                                      # [B_flat, 64]

        # Dot-product attention → target logits
        attn_logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)           # [B_flat, 9]
        attn_logits = attn_logits / math.sqrt(ATTN_D)                             # [B_flat, 9]

        # Context vector: softmax-weighted sum of keys
        attn_weights = torch.softmax(attn_logits, dim=-1)                         # [B_flat, 9]
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)           # [B_flat, 64]

        # ---- 5. Inject context into actor features ----
        a_feat_aug = torch.cat([a_feat, context], dim=1)                          # [B_flat, 320]
        a_feat_fused = self.context_fusion(a_feat_aug)                            # [B_flat, 256]

        # ---- 6. Twin Critics (Double-Q: both receive attention context) ----
        v_feat_aug = torch.cat([lstm_feat, context], dim=1)                       # [B_flat, 320]
        # Critic 1
        v_feat_1 = self.critic_backbone_1(self.critic_context_fusion_1(v_feat_aug))
        value_1 = self.value_head_1(v_feat_1)                                     # [B_flat, 1]
        # Critic 2
        v_feat_2 = self.critic_backbone_2(self.critic_context_fusion_2(v_feat_aug))
        value_2 = self.value_head_2(v_feat_2)                                     # [B_flat, 1]

        # ---- 7. Action logits (all use context-enhanced actor features) ----
        logit_button  = self.head_button(a_feat_fused)        # [B_flat, 12]
        logit_move_x  = self.head_move_x(a_feat_fused)        # [B_flat, 16]
        logit_move_z  = self.head_move_z(a_feat_fused)        # [B_flat, 16]
        logit_skill_x = self.head_skill_x(a_feat_fused)       # [B_flat, 16]
        logit_skill_z = self.head_skill_z(a_feat_fused)       # [B_flat, 16]

        # ---- 8. P0: pseudo-autoregressive Target Head ----
        # button_embed: [12] → [64], 将 button 选择编码为条件向量
        # head_target: [256+64] → [9], 条件化的目标 logits
        # 训练时 detach 防止 target loss 污染 button head 的梯度
        button_embed = self.button_embed_mlp(logit_button.detach())
        logit_target = self.head_target(
            torch.cat([a_feat_fused, button_embed], dim=1)
        )

        result_list = [
            logit_button, logit_move_x, logit_move_z,
            logit_skill_x, logit_skill_z, logit_target, value_1, value_2,
        ]

        if not inference:
            self.cached_a_feat_fused = a_feat_fused
            return result_list

        # ================================================================
        # Inference path: masked sampling
        # ================================================================
        if legal_action is not None:
            la_splits = list(torch.split(legal_action, self.legal_action_size, dim=1))
        else:
            la_splits = [torch.ones_like(r) for r in result_list]

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

        # ---- Target Head: button-conditional (P0) ----
        # 随机路径：用采样的 button 计算 target
        stoch_onehot = nn.functional.one_hot(action_list[0], self.label_size_list[0]).float()
        stoch_embed = self.button_embed_mlp(stoch_onehot)
        stoch_target_logit = self.head_target(
            torch.cat([a_feat_fused, stoch_embed], dim=1)
        )

        # 确定性路径：用 argmax button 计算 target
        det_onehot = nn.functional.one_hot(d_action_list[0], self.label_size_list[0]).float()
        det_embed = self.button_embed_mlp(det_onehot)
        det_target_logit = self.head_target(
            torch.cat([a_feat_fused, det_embed], dim=1)
        )

        n_button = self.label_size_list[0]
        n_target = self.label_size_list[-1]

        if legal_action is not None:
            full_target_mask = legal_action[:, sum(self.label_size_list[:-1]):]
            full_target_mask = full_target_mask.reshape(-1, n_button, n_target)

            batch_indices = torch.arange(B_flat, device=feature_vec.device)
            target_mask = full_target_mask[batch_indices, action_list[0], :]
            d_target_mask = full_target_mask[batch_indices, d_action_list[0], :]
        else:
            target_mask = torch.ones(B_flat, n_target, device=feature_vec.device)
            d_target_mask = target_mask

        target_probs = masked_softmax(stoch_target_logit, target_mask)
        target_action = masked_categorical_sample(stoch_target_logit, target_mask)
        d_target_probs = masked_softmax(det_target_logit, d_target_mask)
        d_target_action = torch.argmax(d_target_probs, dim=-1)

        action_list.append(target_action)
        d_action_list.append(d_target_action)
        prob_list.append(target_probs)
        d_prob_list.append(d_target_probs)

        flat_prob = torch.cat(prob_list, dim=1)
        flat_d_prob = torch.cat(d_prob_list, dim=1)
        all_logits = [result_list[0], result_list[1], result_list[2],
                      result_list[3], result_list[4], stoch_target_logit]
        logits = torch.flatten(torch.cat(all_logits, dim=1), start_dim=1)
        # Inference: use average of twin critics as value estimate
        value_out = (result_list[-2] + result_list[-1]) / 2.0

        return [
            logits, value_out,
            lstm_cell_out.unsqueeze(0),   # [1, B, 512]
            lstm_hidden_out.unsqueeze(0), # [1, B, 512]
            action_list, d_action_list,
            flat_prob, flat_d_prob,
        ]

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------
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
                -1, self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [w.squeeze(dim=1) for w in usq_weight_list]
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-2]
        value_result_1 = rst_list[-2]
        value_result_2 = rst_list[-1]

        # ================================================================
        # P0 修复: 用 rollout 时的真实 button 标签重算 target logits
        #
        # forward 训练路径中 button_embed 来自 logit_button.detach()，
        # 反映的是当前策略的按钮倾向，不是 rollout 时实际按下的按钮。
        # 此处用 label_list[0]（rollout 真实 button）构造 one-hot 重算。
        # ================================================================
        if self.cached_a_feat_fused is not None:
            gt_button = label_list[0]
            gt_button_onehot = nn.functional.one_hot(
                gt_button.long(), self.label_size_list[0]
            ).float()
            gt_button_embed = self.button_embed_mlp(gt_button_onehot)
            label_result[5] = self.head_target(
                torch.cat([self.cached_a_feat_fused, gt_button_embed], dim=1)
            )

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        fla_shape = list(self.seri_vec_split_shape[1])
        fla_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(fla_shape)
        legal_action_flag_list = list(torch.split(feature_legal_action, self.label_size_list, dim=1))

        # ---- Twin Value loss with cross-clipping ----
        # Each critic clips relative to the other's prediction,
        # preventing either from drifting too far from consensus.
        v1 = value_result_1.squeeze(dim=1)
        v2 = value_result_2.squeeze(dim=1)
        v1_clipped = v2.detach() + (v1 - v2.detach()).clamp(-0.2, 0.2)
        v2_clipped = v1.detach() + (v2 - v1.detach()).clamp(-0.2, 0.2)
        self.value_cost = 0.5 * torch.mean(torch.max(
            torch.square(v1 - reward), torch.square(v1_clipped - reward)
        ), dim=0) + 0.5 * torch.mean(torch.max(
            torch.square(v2 - reward), torch.square(v2_clipped - reward)
        ), dim=0)

        # ---- Policy loss ----
        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0)

        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue

            mask = legal_action_flag_list[task_index]
            logit = label_result[task_index]
            one_hot = nn.functional.one_hot(
                label_list[task_index].long(), self.label_size_list[task_index]
            ).float()

            label_probability = masked_softmax(logit, mask)
            label_probability = label_probability * mask + self.min_policy * mask
            label_probability = label_probability / label_probability.sum(1, keepdim=True).clamp(min=epsilon)
            label_probability_list.append(label_probability)

            policy_p = (one_hot * label_probability).sum(1)
            policy_log_p = torch.log(policy_p + epsilon)
            old_policy_p = (one_hot * old_label_probability_list[task_index] + epsilon).sum(1)
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

        # ---- Entropy loss ----
        current_entropy_loss_index = 0
        entropy_loss_list = []

        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                entropy_loss_list.append(torch.tensor(0.0))
                continue

            prob = label_probability_list[current_entropy_loss_index]
            mask = legal_action_flag_list[task_index]
            temp_entropy_loss = -torch.sum(
                prob * mask * torch.log(prob + epsilon), dim=1,
            )
            temp_entropy_loss = -torch.sum(
                temp_entropy_loss * weight_list[task_index].float() * frame_is_train
            ) / torch.maximum(
                torch.sum(weight_list[task_index].float() * frame_is_train), torch.tensor(1.0)
            )
            entropy_loss_list.append(temp_entropy_loss)
            current_entropy_loss_index += 1

        self.entropy_cost = torch.tensor(0.0)
        for e in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + e

        self.entropy_cost_list = entropy_loss_list
        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


# Legacy compatibility
class MLP(nn.Module):
    def __init__(
        self, fc_feat_dim_list: List[int], name: str,
        non_linearity: nn.Module = nn.ReLU, non_linearity_last: bool = False,
    ):
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
