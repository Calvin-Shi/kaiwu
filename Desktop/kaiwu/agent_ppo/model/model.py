#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

精简重构版网络 + Target 注意力机制。
Feature(114) → FC(128) → LSTM(128→128) → Actor/Critic split
Target head: Query-Key Attention 替换简单 Linear 头。
Context 向量注入 Actor 特征，让 Button/Move/Skill 头感知"当前盯着谁"。
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
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE      # 128
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

        FEAT_DIM = 144
        HIDDEN = 128
        ATTN_D = 32  # attention embedding dimension

        # ---- 1. Feature embedding (114 → 128) ----
        self.feature_embed = nn.Sequential(
            make_fc_layer(FEAT_DIM, HIDDEN), nn.ReLU()
        )

        # ---- 2. LSTM (128 → 128) ----
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, num_layers=1, batch_first=True)

        # ---- 3. Actor shared backbone (128 → 128) ----
        self.actor_shared = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU()
        )

        # ---- 4. Critic (128 → 128 → 1) ----
        self.critic_backbone = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU()
        )
        self.value_head = make_fc_layer(HIDDEN, 1, gain=1.0)

        # ---- 5. Action heads (128 → label_size) ----
        self.head_button  = make_fc_layer(HIDDEN, self.label_size_list[0], gain=0.01)
        self.head_move_x  = make_fc_layer(HIDDEN, self.label_size_list[1], gain=0.01)
        self.head_move_z  = make_fc_layer(HIDDEN, self.label_size_list[2], gain=0.01)
        self.head_skill_x = make_fc_layer(HIDDEN, self.label_size_list[3], gain=0.01)
        self.head_skill_z = make_fc_layer(HIDDEN, self.label_size_list[4], gain=0.01)

        # ---- 6. Target attention: Key embeddings for 9 candidate targets ----
        # Targets (per action space spec):
        #   0=None  1=EnemyHero  2=Self  3-6=Soldiers×4  7=Tower  8=Monster
        self.none_key = nn.Parameter(torch.zeros(1, ATTN_D))
        self.emy_hero_key = make_fc_layer(32, ATTN_D)     # enemy hero: 32 dims
        self.self_key = make_fc_layer(32, ATTN_D)          # self: 32 dims
        self.soldier_key = make_fc_layer(7, ATTN_D)        # shared for 4 soldiers
        self.tower_key = make_fc_layer(7, ATTN_D)          # tower: 7 dims
        self.monster_key = make_fc_layer(7, ATTN_D)        # monster: 7 dims

        # ---- 7. Query projection: LSTM output → query embedding ----
        self.target_query = make_fc_layer(HIDDEN, ATTN_D)

        # ---- 8. Context fusion: inject attention context into actor features ----
        self.context_fusion = make_fc_layer(HIDDEN + ATTN_D, HIDDEN)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data_list, inference=False, legal_action=None):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        B_flat = feature_vec.shape[0]
        H = self.lstm_hidden_dim
        ATTN_D = 32

        # ---- 1. Feature embedding ----
        embed = self.feature_embed(feature_vec)          # [B_flat, 128]

        # ---- 2. LSTM step ----
        if not inference:
            T = self.lstm_time_steps                     # e.g. 16
            B_real = B_flat // T
            feat_3d = embed.reshape(B_real, T, H)       # [B, T, 128]
            h0 = lstm_hidden_init.reshape(1, B_real, H) # [1, B, 128]
            c0 = lstm_cell_init.reshape(1, B_real, H)   # [1, B, 128]
            lstm_out, (hn, cn) = self.lstm(feat_3d, (h0, c0))
            lstm_feat = lstm_out.reshape(B_flat, H)     # [B*T, 128]
            lstm_cell_out = cn.reshape(B_real, H)       # [B, 128]
            lstm_hidden_out = hn.reshape(B_real, H)     # [B, 128]
        else:
            lstm_in = embed.unsqueeze(1)                 # [B, 1, 128]
            h0 = lstm_hidden_init.reshape(1, B_flat, H)
            c0 = lstm_cell_init.reshape(1, B_flat, H)
            lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))
            lstm_feat = lstm_out.squeeze(1)              # [B, 128]
            lstm_cell_out = cn.reshape(B_flat, H)
            lstm_hidden_out = hn.reshape(B_flat, H)

        # ---- 3. Actor shared ----
        a_feat = self.actor_shared(lstm_feat)            # [B_flat, 128]

        # ---- 4. Critic ----
        v_feat = self.critic_backbone(lstm_feat)         # [B_flat, 128]
        value = self.value_head(v_feat)                  # [B_flat, 1]

        # ---- 5. Target attention ----
        # Feature layout [B_flat, 144]:
        #   [0:32]   = Self (friendly hero, 32)
        #   [32:64]  = Enemy hero (32)
        #   [64:71]  = Organ (enemy tower, 7)
        #   [71:81]  = Tactical (10)
        #   [81:109] = Friendly soldiers (4×7)
        #   [109:137]= Enemy soldiers (4×7)
        #   [137:144]= Cake/resource (1×7)

        self_feat        = feature_vec[:, 0:32]            # [B_flat, 32]
        emy_hero_feat    = feature_vec[:, 32:64]           # [B_flat, 32]
        emy_tower_feat   = feature_vec[:, 64:71]           # [B_flat,  7]
        emy_soldier_feat = feature_vec[:, 109:137].reshape(B_flat, 4, 7)  # [B_flat, 4, 7]
        resource_feat    = feature_vec[:, 137:144]         # [B_flat,  7]

        # Build 9 key tensors → correct target order per action spec
        none_k      = self.none_key.expand(B_flat, 1, ATTN_D)                    # [B_flat, 1, 32] Target 0
        emy_hero_k  = self.emy_hero_key(emy_hero_feat).unsqueeze(1)               # [B_flat, 1, 32] Target 1
        self_k      = self.self_key(self_feat).unsqueeze(1)                       # [B_flat, 1, 32] Target 2

        # 4 soldiers share embedding weight (Target 3-6)
        emy_soldier_k = self.soldier_key(
            emy_soldier_feat.reshape(B_flat * 4, 7)
        ).reshape(B_flat, 4, ATTN_D)                                              # [B_flat, 4, 32]

        tower_k     = self.tower_key(emy_tower_feat).unsqueeze(1)                 # [B_flat, 1, 32] Target 7
        monster_k   = self.monster_key(resource_feat).unsqueeze(1)                # [B_flat, 1, 32] Target 8

        keys = torch.cat([
            none_k,          # Target 0: None
            emy_hero_k,      # Target 1: Enemy hero
            self_k,          # Target 2: Self
            emy_soldier_k,   # Target 3-6: Soldiers ×4
            tower_k,         # Target 7: Tower
            monster_k,       # Target 8: Monster
        ], dim=1)                                                                 # [B_flat, 9, 32]

        # Query: LSTM output → query embedding
        query = self.target_query(lstm_feat)                                      # [B_flat, 32]

        # Dot-product attention → target logits
        attn_logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)           # [B_flat, 9]
        attn_logits = attn_logits / math.sqrt(ATTN_D)                             # [B_flat, 9]
        logit_target = attn_logits                                                # [B_flat, 9]

        # Context vector: softmax-weighted sum of keys
        attn_weights = torch.softmax(attn_logits, dim=-1)                         # [B_flat, 9]
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)           # [B_flat, 32]

        # ---- 6. Inject context into actor features ----
        a_feat_aug = torch.cat([a_feat, context], dim=1)                          # [B_flat, 160]
        a_feat_fused = self.context_fusion(a_feat_aug)                            # [B_flat, 128]

        # ---- 7. Action logits (all use context-enhanced actor features) ----
        logit_button  = self.head_button(a_feat_fused)        # [B_flat, 12]
        logit_move_x  = self.head_move_x(a_feat_fused)        # [B_flat, 16]
        logit_move_z  = self.head_move_z(a_feat_fused)        # [B_flat, 16]
        logit_skill_x = self.head_skill_x(a_feat_fused)       # [B_flat, 16]
        logit_skill_z = self.head_skill_z(a_feat_fused)       # [B_flat, 16]

        result_list = [
            logit_button, logit_move_x, logit_move_z,
            logit_skill_x, logit_skill_z, logit_target, value,
        ]

        if not inference:
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

        # ---- Last head (target): mask filtered by chosen button ----
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
            target_mask = torch.ones(B_flat, n_target, device=feature_vec.device)
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
        value_out = result_list[-1]

        return [
            logits, value_out,
            lstm_cell_out.unsqueeze(0),   # [1, B, H]
            lstm_hidden_out.unsqueeze(0), # [1, B, H]
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

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        fla_shape = list(self.seri_vec_split_shape[1])
        fla_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(fla_shape)
        legal_action_flag_list = list(torch.split(feature_legal_action, self.label_size_list, dim=1))

        # ---- Value loss ----
        v_sq = value_result.squeeze(dim=1)
        new_advantage = reward - v_sq
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

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
