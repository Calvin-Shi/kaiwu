#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

模块化分实体编码 + LSTM/Bypass 双路径 + Target 注意力 + Twin Critics。
Hero(39)→Enc(64+32key) | Soldier(7)→Enc(32) max-pool×4
Organ(7)→Enc(32+32key) | Resource(7)→Enc(32) | Tactical(11)→MLP(32)
Concat(288)→[LSTM(2层) | Bypass MLP]→Merge(576→288)→Actor/Critic+Attention(32-dim)
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
# Entity encoder: input → hidden → (feat + optional key)
# ---------------------------------------------------------------------------

class EntityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, feat_dim: int, key_dim: int = 0):
        super().__init__()
        self.has_key = key_dim > 0
        self.mlp = nn.Sequential(
            make_fc_layer(in_dim, hidden), nn.ReLU(),
            make_fc_layer(hidden, feat_dim + key_dim),
        )
        self.feat_dim = feat_dim
        self.key_dim = key_dim

    def forward(self, x: torch.Tensor):
        out = self.mlp(x)
        if self.has_key:
            return out[:, :self.feat_dim], out[:, self.feat_dim:]
        return out, None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- config ----
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE      # 576
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

        FEAT_DIM = 159
        HIDDEN = 288
        ATTN_D = 32
        ENC_H = 64
        self.button_embed_dim = Config.BUTTON_EMBED_DIM  # 64

        # ---- Entity encoders ----
        # Hero: 39→64→(64 feat + 32 key)
        self.hero_encoder = EntityEncoder(39, ENC_H, 64, ATTN_D)
        # Soldier: 7→32 (shared for all 8, no separate key)
        self.soldier_encoder = EntityEncoder(7, 32, ATTN_D, 0)
        # Organ: 7→64→(32 feat + 32 key)
        self.organ_encoder = EntityEncoder(7, ENC_H, ATTN_D, ATTN_D)
        # Resource: 7→32 (no separate key — use feat directly)
        self.resource_encoder = EntityEncoder(7, 32, ATTN_D, 0)
        # Tactical: 11→64→32
        self.tactical_mlp = nn.Sequential(
            make_fc_layer(11, ENC_H), nn.ReLU(),
            make_fc_layer(ENC_H, ATTN_D),
        )

        # Concat dim: 64+64+32+32+32+32+32 = 288 = HIDDEN
        self.concat_proj = make_fc_layer(HIDDEN, HIDDEN)

        # ---- LSTM (288→288), 2 layers ----
        self.lstm_num_layers = 2
        self.lstm_per_layer = HIDDEN
        self.lstm = nn.LSTM(HIDDEN, HIDDEN, num_layers=self.lstm_num_layers, batch_first=True)

        # ---- Bypass MLP: 纯前馈旁路，保留 LSTM 可能遗忘的即时信息 ----
        self.bypass_mlp = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )

        # ---- Merge: cat(LSTM out, bypass out) = 576 → 288 ----
        self.merge_mlp = nn.Sequential(
            make_fc_layer(HIDDEN * 2, HIDDEN), nn.ReLU(),
        )

        # ---- Actor shared backbone (288→288→288) ----
        self.actor_shared = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )

        # ---- Twin Critics ----
        self.critic_backbone_1 = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.critic_context_fusion_1 = make_fc_layer(HIDDEN + ATTN_D, HIDDEN)
        self.value_head_1 = make_fc_layer(HIDDEN, 1, gain=1.0)
        self.critic_backbone_2 = nn.Sequential(
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
            make_fc_layer(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.critic_context_fusion_2 = make_fc_layer(HIDDEN + ATTN_D, HIDDEN)
        self.value_head_2 = make_fc_layer(HIDDEN, 1, gain=1.0)

        # ---- Action heads (288 → label_size) ----
        self.head_button  = make_fc_layer(HIDDEN, self.label_size_list[0], gain=0.01)
        self.head_move_x  = make_fc_layer(HIDDEN, self.label_size_list[1], gain=0.01)
        self.head_move_z  = make_fc_layer(HIDDEN, self.label_size_list[2], gain=0.01)
        self.head_skill_x = make_fc_layer(HIDDEN, self.label_size_list[3], gain=0.01)
        self.head_skill_z = make_fc_layer(HIDDEN, self.label_size_list[4], gain=0.01)

        # ---- Target attention ----
        self.none_key = nn.Parameter(torch.randn(1, ATTN_D) * 0.01)
        self.target_query = make_fc_layer(HIDDEN, ATTN_D)

        # ---- Context fusion ----
        self.context_fusion = make_fc_layer(HIDDEN + ATTN_D, HIDDEN)

        # ---- P0: 伪自回归 Target Head ----
        self.button_embed_mlp = nn.Sequential(
            make_fc_layer(self.label_size_list[0], self.button_embed_dim),
            nn.ReLU(),
        )
        self.head_target = make_fc_layer(
            HIDDEN + self.button_embed_dim, self.label_size_list[5], gain=0.01
        )

        self.cached_a_feat_fused = None

    # ------------------------------------------------------------------
    # Entity splitting and encoding
    # ------------------------------------------------------------------
    def _encode_entities(self, feature_vec, B_flat):
        """Split 159-dim feature into entity slices, return concat+keys.

        Feature layout [B_flat, 159]:
          [0:39]   = Self hero
          [39:78]  = Enemy hero
          [78:85]  = Organ (enemy tower)
          [85:96]  = Tactical
          [96:124] = Friendly soldiers (4×7)
          [124:152]= Enemy soldiers (4×7)
          [152:159]= Resource
        """
        ATTN_D = 32

        # ---- Split ----
        self_feat     = feature_vec[:, 0:39]
        emy_feat      = feature_vec[:, 39:78]
        tower_feat    = feature_vec[:, 78:85]
        tactical_feat = feature_vec[:, 85:96]
        fri_soldiers  = feature_vec[:, 96:124].reshape(B_flat, 4, 7)
        emy_soldiers  = feature_vec[:, 124:152].reshape(B_flat, 4, 7)
        resource_feat = feature_vec[:, 152:159]

        # ---- Encode ----
        self_hero_f, self_hero_k = self.hero_encoder(self_feat)       # (B,64), (B,32)
        emy_hero_f, emy_hero_k   = self.hero_encoder(emy_feat)        # (B,64), (B,32)
        tower_f, tower_k         = self.organ_encoder(tower_feat)     # (B,32), (B,32)
        tactical_f               = self.tactical_mlp(tactical_feat)   # (B,32)
        resource_f, _            = self.resource_encoder(resource_feat)  # (B,32)

        # Soldiers: encode each → max-pool
        def _enc_soldiers(soldiers_4d):
            flat = soldiers_4d.reshape(B_flat * 4, 7)
            enc, _ = self.soldier_encoder(flat)         # (4B, 32)
            return enc.reshape(B_flat, 4, ATTN_D)       # (B, 4, 32)

        fri_enc = _enc_soldiers(fri_soldiers)
        emy_enc = _enc_soldiers(emy_soldiers)
        fri_pooled, _ = fri_enc.max(dim=1)              # (B, 32)
        emy_pooled, _ = emy_enc.max(dim=1)              # (B, 32)

        # ---- Concat: 64+64+32+32+32+32+32 = 288 ----
        concat_feat = torch.cat([
            self_hero_f,   # 64
            emy_hero_f,    # 64
            fri_pooled,    # 32
            emy_pooled,    # 32
            tower_f,       # 32
            resource_f,    # 32
            tactical_f,    # 32
        ], dim=1)

        # ---- Attention keys (9 targets, 32-dim each) ----
        keys = torch.cat([
            self.none_key.expand(B_flat, 1, ATTN_D),   # 0: None
            emy_hero_k.unsqueeze(1),                    # 1: Enemy hero
            self_hero_k.unsqueeze(1),                   # 2: Self
            emy_enc,                                     # 3-6: Soldiers ×4
            tower_k.unsqueeze(1),                        # 7: Tower
            resource_f.unsqueeze(1),                     # 8: Resource
        ], dim=1)  # (B_flat, 9, 32)

        return concat_feat, keys

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, data_list, inference=False, legal_action=None):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        B_flat = feature_vec.shape[0]
        N = self.lstm_num_layers
        H = self.lstm_per_layer
        ATTN_D = 32

        # ---- 1. Entity encoding ----
        concat_feat, keys = self._encode_entities(feature_vec, B_flat)
        embed = self.concat_proj(concat_feat)            # [B_flat, 288]

        # ---- 2. LSTM step ----
        if not inference:
            T = self.lstm_time_steps
            B_real = B_flat // T
            feat_3d = embed.reshape(B_real, T, H)
            h0 = lstm_hidden_init.reshape(N, B_real, H)
            c0 = lstm_cell_init.reshape(N, B_real, H)
            lstm_out, (hn, cn) = self.lstm(feat_3d, (h0, c0))
            lstm_feat = lstm_out.reshape(B_flat, H)
            lstm_cell_out = cn.reshape(B_real, N * H)
            lstm_hidden_out = hn.reshape(B_real, N * H)
        else:
            lstm_in = embed.unsqueeze(1)
            h0 = lstm_hidden_init.reshape(N, B_flat, H)
            c0 = lstm_cell_init.reshape(N, B_flat, H)
            lstm_out, (hn, cn) = self.lstm(lstm_in, (h0, c0))
            lstm_feat = lstm_out.squeeze(1)
            lstm_cell_out = cn.reshape(B_flat, N * H)
            lstm_hidden_out = hn.reshape(B_flat, N * H)

        # ---- 3. Bypass + Merge: 纯前馈旁路保留即时信息 ----
        bypass_feat = self.bypass_mlp(concat_feat)       # [B_flat, 288]
        merged = self.merge_mlp(torch.cat([lstm_feat, bypass_feat], dim=1))  # [B_flat, 288]

        # ---- 4. Actor shared ----
        a_feat = self.actor_shared(merged)

        # ---- 5. Target attention (section renumbered) ----
        query = self.target_query(merged)                # [B_flat, 32]
        attn_logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        attn_logits = attn_logits / math.sqrt(ATTN_D)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)

        # ---- 6. Inject context into actor ----
        a_feat_aug = torch.cat([a_feat, context], dim=1)  # [B_flat, 320]
        a_feat_fused = self.context_fusion(a_feat_aug)    # [B_flat, 288]

        # ---- 7. Twin Critics (use merged features) ----
        v_feat_aug = torch.cat([merged, context], dim=1)
        v_feat_1 = self.critic_backbone_1(self.critic_context_fusion_1(v_feat_aug))
        value_1 = self.value_head_1(v_feat_1)
        v_feat_2 = self.critic_backbone_2(self.critic_context_fusion_2(v_feat_aug))
        value_2 = self.value_head_2(v_feat_2)

        # ---- 8. Action logits ----
        logit_button  = self.head_button(a_feat_fused)
        logit_move_x  = self.head_move_x(a_feat_fused)
        logit_move_z  = self.head_move_z(a_feat_fused)
        logit_skill_x = self.head_skill_x(a_feat_fused)
        logit_skill_z = self.head_skill_z(a_feat_fused)

        # ---- 8. Target Head ----
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
        # Inference path
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

        # Button-conditional target
        stoch_onehot = F.one_hot(action_list[0], self.label_size_list[0]).float()
        stoch_embed = self.button_embed_mlp(stoch_onehot)
        stoch_target_logit = self.head_target(
            torch.cat([a_feat_fused, stoch_embed], dim=1)
        )

        det_onehot = F.one_hot(d_action_list[0], self.label_size_list[0]).float()
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
        value_out = (result_list[-2] + result_list[-1]) / 2.0

        return [
            logits, value_out,
            lstm_cell_out.unsqueeze(0),
            lstm_hidden_out.unsqueeze(0),
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

        # P0 fix: recompute target logits with ground-truth button
        if self.cached_a_feat_fused is not None:
            gt_button = label_list[0]
            gt_button_onehot = F.one_hot(
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
            one_hot = F.one_hot(
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
