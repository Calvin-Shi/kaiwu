#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

王者荣耀 1v1 奖励计算模块（高阶战术强化版）。

设计要点：
1. 继承原版零和博弈 (Zero-Sum) 与时间差分逻辑；
2. 【降维打击】注入独立高阶奖励项：拉扯白嫖 (hp_trade)、补刀爆发 (last_hit)、防发呆惩罚 (anti_camp)。
"""

import math
from agent_ppo.conf.conf import GameConfig


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        

def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
        
    # =========================================================
    # 【战术升级】：追加 REWARD_WEIGHT_DICT 之外独立的高阶微操奖励项
    # （已在 REWARD_WEIGHT_DICT 中的键不在此重复定义，统一由 GameConfig 管理）
    # =========================================================
    calc_frame_map["hp_trade"] = RewardStruct(3.0)  # 拉扯与白嫖奖励
    calc_frame_map["last_hit"] = RewardStruct(2.0)  # 补刀瞬时刺激
    calc_frame_map["anti_camp"] = RewardStruct(1.0) # 防发呆/站桩惩罚
    calc_frame_map["kiting"] = RewardStruct(1.0)    # 极限拉扯奖励
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1

        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        
        # 记录英雄最近轨迹，用于防发呆判定
        self.pos_window = []
        # <--- 【新增】用于防刷分的连续回城帧计数器
        self.consecutive_recall_frames = 0
        # P0/P1: 追踪上一帧敌方塔血量比例，用于检测塔伤害和拆塔事件
        self._prev_enemy_tower_hp_ratio = 1.0
        self.cached_main_tower_pos = None
        self.cached_enemy_tower_pos = None
        # 狄仁杰 S3 黄牌破甲追击窗口
        self._dj_s3_active = False              # 窗口是否激活
        self._dj_s3_end_step = -10**9           # 窗口结束 step
        self._dj_s3_enemy_hp_open = None        # 窗口开启时敌方血量（跟踪峰值）
        self._dj_s3_enemy_hp_max = 1.0          # 窗口开启时敌方最大血量
        self._dj_s3_dmg_dealt = 0.0             # 窗口内累计伤害比例（检测是否跟进）
        # 狄仁杰 S2 防滥用与极限反杀
        self._dj_prev_frame_hp = -1.0           # 上一帧我方血量（检测瞬降）
        self._dj_prev_frame_hp_max = 1.0        # 上一帧我方最大血量
        self._dj_prev_kill_cnt = 0              # 上一帧击杀数（检测反杀）
        self._dj_s2_used_frame = -1              # S2 最近使用的帧号（-1 = 未使用）
        # 鲁班七号被动连招状态机（独立于狄仁杰的 _dj_* 通道）
        self._luban_passive_ready = False           # 是否处于"强化普攻等待期"
        self._luban_passive_open_frame = -10**9     # 窗口开启时的帧号
        self._luban_passive_skills_used = 0         # 窗口内累计使用技能数（检测吞被动）
        self._luban_passive_enemy_hp_open = None    # 窗口开启时敌方血量（检测扫射命中）
        self._luban_passive_enemy_hp_max_open = 1   # 窗口开启时敌方最大血量
        # 用于记录上一帧的技能使用情况，防止增量计算报错
        self._skill_prev_used = [0]*7
        self._skill_prev_hit  = [0]*7
        # 闪现边沿检测：记录上一帧的 CD 比例，用于检测"交闪"瞬间
        self._prev_flash_cd_ratio = 0.0
        self._prev_main_kill_cnt = 0
        self._prev_main_dead_cnt = 0

        # 蛋糕/血包追踪状态
        self._prev_cake_npc_id = None    # 上一帧最近蛋糕的 runtime_id
        self._prev_main_hp_abs = 0.0     # 上一帧我方绝对血量
        self._prev_main_hp_max = 1.0     # 上一帧我方最大血量
        self._cake_nearby_this_frame = False  # 本帧是否附近有蛋糕
        self._cake_dist_this_frame = 999999.0 # 本帧距离最近蛋糕的距离
        self._cake_picked_up = False     # 本帧是否拾取了蛋糕

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data, action=None):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value, action=action)

        frame_no = frame_data["frame_no"]
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    @staticmethod
    def _sigmoid(x):
        if x < -20.0:
            return 0.0
        if x > 20.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _safe_get(d, key, default=0):
        if d is None:
            return default
        value = d.get(key, default)
        return default if value is None else value

    @staticmethod
    def _is_attacking_frame(action):
        if action is None:
            return False
        return action[0] in GameConfig.ATTACK_BUTTON_INDICES

    def _calc_total_exp(self, hero):
        if hero is None:
            return 0.0
        level = int(self._safe_get(hero, "level", 1))
        cur_level_exp = float(self._safe_get(hero, "exp", 0))
        total_exp = 0.0
        for lv in range(1, max(level, 1)):
            total_exp += self.m_each_level_max_exp.get(lv, 0)
        total_exp += cur_level_exp
        return total_exp

    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        # 【修复3.1】：增加 enemy_hero
        main_hero, enemy_hero, main_tower, enemy_tower = None, None, None, None

        for hero in frame_data["hero_states"]:
            if hero["camp"] == camp:
                main_hero = hero
            else:
                enemy_hero = hero  # <--- 【必须提取敌方英雄】

        for organ in frame_data["npc_states"]:
            if organ["sub_type"] == 21:
                if organ["camp"] == camp:
                    main_tower = organ
                else:
                    enemy_tower = organ

        hp = float(self._safe_get(main_hero, "hp", 0))
        max_hp = float(self._safe_get(main_hero, "max_hp", 1))
        ep = float(self._safe_get(main_hero, "ep", 0))
        max_ep = float(self._safe_get(main_hero, "max_ep", 1))
        money = float(self._safe_get(main_hero, "money", 0))
        kill_cnt = float(self._safe_get(main_hero, "kill_cnt", 0))
        dead_cnt = float(self._safe_get(main_hero, "dead_cnt", 0))
        total_exp = self._calc_total_exp(main_hero)

        if max_hp <= 0: max_hp = 1.0
        if max_ep <= 0: max_ep = 1.0

        is_main_side = (camp == self.main_hero_camp)
        if is_main_side and main_hero is not None:
            used_delta, hit_delta = self._skill_events_this_frame(main_hero)
        else:
            used_delta, hit_delta = [0]*7, [0]*7

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            if reward_name == "tower_hp_point":
                if main_tower is not None and main_tower.get("max_hp", 0):
                    reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = hp / max_hp
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = ep / max_ep
            elif reward_name == "money":
                reward_struct.cur_frame_value = money
            elif reward_name == "exp":
                reward_struct.cur_frame_value = total_exp
            elif reward_name == "kill":
                reward_struct.cur_frame_value = kill_cnt
            elif reward_name == "death":
                reward_struct.cur_frame_value = dead_cnt
            elif reward_name == "hero_combo_window":
                if is_main_side:
                    # 将 frameNo 改为 frame_no
                    inc = self.calc_hero_combo_reward(frame_data.get("frame_no", 0), main_hero, enemy_hero, used_delta, hit_delta)
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value
            # ==========================================
            # 宏观节奏纠偏 1：击杀但不发育惩罚
            # ==========================================
            elif reward_name == "kill_gold_consistency":
                if is_main_side:
                    # 使用 pre-extracted 局部变量，避免字段名不一致导致取值恒为0
                    emy_kills  = float(self._safe_get(enemy_hero, "kill_cnt", 0)) if enemy_hero else 0
                    emy_gold   = float(self._safe_get(enemy_hero, "money", 0))   if enemy_hero else 0

                    kill_diff  = kill_cnt - emy_kills
                    gold_diff  = money    - emy_gold

                    # 阈值设置：如果我方人头领先 >= 1，但经济差却 <= 0（说明在无效打架漏兵线）
                    KILL_LEAD_THRESH = 1
                    PENALTY = -0.05        # 每帧惩罚，加重以倒逼发育意识

                    inc = PENALTY if (kill_diff >= KILL_LEAD_THRESH and gold_diff <= 0) else 0.0
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            # ==========================================
            # 宏观节奏纠偏 2：击杀但不推塔惩罚
            # ==========================================
            elif reward_name == "kill_tower_consistency":
                if is_main_side:
                    # 安全获取防御塔血量百分比
                    def _tower_pct(tower):
                        if tower is None:
                            return 1.0
                        hp = float(self._safe_get(tower, "hp", 0))
                        mx = float(self._safe_get(tower, "max_hp", 1))
                        return (hp / mx) if mx > 0 else 1.0

                    my_tower_pct  = _tower_pct(main_tower)
                    emy_tower_pct = _tower_pct(enemy_tower)

                    # 当前推塔净进展（我方塔血量百分比 - 敌方塔血量百分比）
                    tower_pressure_now = my_tower_pct - emy_tower_pct

                    emy_kills = float(self._safe_get(enemy_hero, "kill_cnt", 0)) if enemy_hero else 0
                    kill_diff = kill_cnt - emy_kills

                    # 阈值设置：如果人头领先，但推塔进度 <= 1% 的微小缓冲值（说明杀完人就发呆/回城，不推线）
                    KILL_LEAD_THRESH = 1
                    BUFFER = 0.01
                    PENALTY = -0.05

                    inc = PENALTY if (kill_diff >= KILL_LEAD_THRESH and tower_pressure_now <= BUFFER) else 0.0
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            # ==========================================
            # 闪现边沿检测与事件关联奖励
            # ==========================================
            elif reward_name == "skill5_flash":
                if is_main_side:
                    # 提取闪现槽位 (slot_states[5]) 的 CD 比例
                    slots = (main_hero.get("skill_state", {}) or {}).get("slot_states", []) or []
                    flash_cd_ratio = 0.0
                    if len(slots) > 5 and self._safe_get(slots[5], "level", 0) > 0:
                        cd = float(self._safe_get(slots[5], "cd", 0))
                        max_cd = float(self._safe_get(slots[5], "max_cd", 0)
                                       or self._safe_get(slots[5], "maxCd", 1))
                        if max_cd > 0:
                            flash_cd_ratio = cd / max_cd

                    inc = 0.0
                    # 边沿跳变检测：CD 比例从 0 瞬间拉高到 >0.8，说明本帧刚交了闪现
                    FLASH_EDGE_THRESH = 0.8
                    if self._prev_flash_cd_ratio <= 0.01 and flash_cd_ratio > FLASH_EDGE_THRESH:
                        # 基础惩罚：防止 AI 乱交闪赶路
                        inc -= 0.07

                        # 关联事件检测：同一帧内暴击/阵亡判定
                        cur_kill = float(self._safe_get(main_hero, "kill_cnt", 0))
                        cur_dead = float(self._safe_get(main_hero, "dead_cnt", 0))

                        if cur_kill > self._prev_main_kill_cnt:
                            # 闪现进攻成功击杀，巨额正反馈
                            inc += 2.0
                        if cur_dead > self._prev_main_dead_cnt:
                            # 交闪后仍被击杀，重罚
                            inc -= 1.0

                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc

                    # 持久化本帧状态供下一帧边沿检测比对
                    self._prev_flash_cd_ratio = flash_cd_ratio
                    self._prev_main_kill_cnt = float(self._safe_get(main_hero, "kill_cnt", 0))
                    self._prev_main_dead_cnt = float(self._safe_get(main_hero, "dead_cnt", 0))
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            # ==========================================
            # 蛋糕/血包检测与奖励 (Cake Hunt & Pickup)
            # ==========================================
            elif reward_name == "cake_hunt":
                if is_main_side and main_hero is not None:
                    cake_dist, cake_exists, cake_id = self._find_nearest_cake(frame_data, camp)
                    self._cake_nearby_this_frame = cake_exists
                    self._cake_dist_this_frame = cake_dist
                    # 记录蛋糕 ID 供下一帧拾取检测
                    if cake_exists:
                        self._prev_cake_npc_id = cake_id
                    # 趋向奖励：距离越近奖励越高，蛋糕不存在时无奖励
                    hunt_reward = 0.0
                    if cake_exists and cake_dist < 15000.0:
                        hunt_reward = max(0.0, 0.03 * (1.0 - cake_dist / 15000.0))
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + hunt_reward
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            elif reward_name == "cake_pickup":
                if is_main_side and main_hero is not None:
                    hp_abs = float(self._safe_get(main_hero, "hp", 0))
                    hp_max = float(self._safe_get(main_hero, "max_hp", 1))
                    pickup_reward = 0.0

                    # 检测拾取事件：双重验证
                    #   (a) 上一帧追踪的蛋糕 NPC 在本帧已消失 (hp<=0 或从 npc_states 移除)
                    #   (b) 英雄血量跳增 >= 4% 最大血量
                    cake_gone = self._check_cake_disappeared(frame_data, self._prev_cake_npc_id)
                    HP_JUMP_THRESH = 0.04  # 4% 最大血量跳变（降低阈值减少漏检）
                    hp_jump = 0.0
                    if hp_max > 0:
                        prev_hp_rate = self._prev_main_hp_abs / max(self._prev_main_hp_max, 1.0)
                        cur_hp_rate = hp_abs / hp_max
                        hp_jump = cur_hp_rate - prev_hp_rate

                    if cake_gone and hp_jump >= HP_JUMP_THRESH:
                        # 双重验证通过：蛋糕确实被吃掉了
                        pickup_reward = hp_jump * 3.0
                        self._cake_picked_up = True
                        self._prev_cake_npc_id = None

                    reward_struct.cur_frame_value = reward_struct.last_frame_value + pickup_reward
                    self._prev_main_hp_abs = hp_abs
                    self._prev_main_hp_max = hp_max
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value
                    self._cake_nearby_this_frame = False
                    self._cake_dist_this_frame = 999999.0

            elif reward_name == "last_hit":
                if is_main_side:
                    reward_struct.cur_frame_value = self._calc_last_hit_from_dead_action(frame_data, main_hero, enemy_hero)
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            else:
                # 兼容新增的自定义键 (hp_trade等)，占位设为0即可
                reward_struct.cur_frame_value = 0.0

    def calc_hero_combo_reward(self, frame_no, hero, enemy, used_delta, hit_delta):
        """
        根据英雄类型，动态计算高阶连招与状态窗口奖励。
        - 狄仁杰(133): 黄牌破甲追击 + 二技能防滥用（见 direnjie_advanced_reward）
        - 鲁班(112): 被动扫射连招状态机（见 luban_passive_combo_reward）
        """
        inc = 0.0

        # 尝试获取英雄配置ID（数据协议中 config_id 仅在 actor_state 内）
        config_id = hero.get("actor_state", {}).get("config_id", 0) if hero else 0

        # ==========================================
        # 英雄 1：狄仁杰 (133) - 黄牌破甲 + 二技能防滥用
        # ==========================================
        if config_id == 133:
            inc += self.direnjie_advanced_reward(frame_no, hero, enemy, used_delta, hit_delta)


        # ==========================================
        # 英雄 2：鲁班七号 (112) - 被动扫射连招状态机
        # ==========================================
        elif config_id == 112:
            inc += self.luban_passive_combo_reward(frame_no, hero, enemy, used_delta, hit_delta)

        return inc

    def direnjie_advanced_reward(self, frame_no, hero, enemy, used_delta, hit_delta):
        """
        狄仁杰专属高级奖励塑形（独立于鲁班的 _luban_* 通道）。

        机制一：黄牌破甲追击窗口 (S3 Combo Window)
          IDLE ──(S3 命中)──> WINDOW (30 steps) ──(窗口结束)──> 检查跟进情况
                               │
                               ├── 敌方掉血 → 1.5× 破甲伤害奖励（鼓励集火）
                               └── 窗口结束 + 伤害<3% + 我方健康 → -0.1 惩罚（浪费机会）

        机制二：二技能防滥用与极限反杀 (S2 Anti-Spam & Survival)
          - S2 使用 + HP>80% + 无近期掉血 → -0.05（满血乱交保命技）
          - S2 使用 + HP<30% → +0.5（极限保命）
          - S2 后 30 帧内击杀敌方 → +0.5（反杀奖励）
        """
        inc = 0.0

        if enemy is None or hero is None:
            self._dj_s3_active = False
            return inc

        now_step = self._step_no(frame_no)

        # 安全获取血量
        emy_hp = float(self._safe_get(enemy, "hp", 0))
        emy_hp_max = float(self._safe_get(enemy, "max_hp", 1))
        if emy_hp_max <= 0:
            emy_hp_max = 1.0
        hero_hp = float(self._safe_get(hero, "hp", 0))
        hero_hp_max = float(self._safe_get(hero, "max_hp", 1))
        if hero_hp_max <= 0:
            hero_hp_max = 1.0
        hero_hp_rate = hero_hp / hero_hp_max

        # 我方死亡 → 强制关闭所有窗口
        if hero_hp <= 0:
            self._dj_s3_active = False
            self._dj_s2_used_frame = -1
            return inc

        # ================================================
        # 机制 1：黄牌破甲追击窗口 (S3 Combo Window)
        # ================================================
        s3_used = (used_delta[3] > 0)
        s3_hit = (hit_delta[3] > 0)
        S3_WINDOW_STEPS = 30    # 破甲窗口时长（~5 秒 @6f/step）
        S3_MIN_DMG_PCT = 0.03   # 窗口内累计伤害 < 3% 视为未跟进
        S3_NO_FOLLOW_PENALTY = -0.1

        # S3 空大惩罚（保留原版逻辑）
        if s3_used and not s3_hit:
            inc -= 0.1

        # S3 命中 → 开启/刷新窗口
        if s3_hit:
            self._dj_s3_active = True
            self._dj_s3_end_step = now_step + S3_WINDOW_STEPS
            self._dj_s3_enemy_hp_open = emy_hp
            self._dj_s3_enemy_hp_max = emy_hp_max
            self._dj_s3_dmg_dealt = 0.0

        # 窗口内收益结算
        if self._dj_s3_active and emy_hp > 0:
            if now_step <= self._dj_s3_end_step:
                if self._dj_s3_enemy_hp_open is not None:
                    # 敌方回血 → 上修基线，避免把回血后的自然血量误判为掉血
                    if emy_hp > self._dj_s3_enemy_hp_open:
                        self._dj_s3_enemy_hp_open = emy_hp

                    hp_drop = max(0, self._dj_s3_enemy_hp_open - emy_hp)
                    if hp_drop > 0:
                        dmg_pct = hp_drop / self._dj_s3_enemy_hp_max
                        self._dj_s3_dmg_dealt += dmg_pct
                        # 破甲期间伤害 ×1.5 倍乘子 —— 趁他病要他命！
                        inc += 1.5 * dmg_pct
                        self._dj_s3_enemy_hp_open = emy_hp
            else:
                # 窗口关闭：判定 AI 是否浪费了破甲机会
                if (self._dj_s3_dmg_dealt < S3_MIN_DMG_PCT
                        and hero_hp_rate > 0.5
                        and emy_hp > 0):
                    inc += S3_NO_FOLLOW_PENALTY
                self._dj_s3_active = False

        # ================================================
        # 机制 2：二技能防滥用与极限反杀 (S2 Anti-Spam)
        # ================================================
        s2_used = (used_delta[2] > 0)
        S2_ABUSE_HP_THRESH = 0.8     # HP > 80% 视为健康
        S2_SURVIVAL_HP_THRESH = 0.3  # HP < 30% 视为危险
        S2_RECENT_DMG_THRESH = 0.02  # 近期掉血 < 2% 视为无承伤
        S2_COUNTER_KILL_WINDOW = 30  # S2 后 30 帧内的击杀视为反杀

        # 计算近期掉血（对比上一帧）
        hp_drop_recent_pct = 0.0
        if self._dj_prev_frame_hp > 0:
            hp_drop = max(0, self._dj_prev_frame_hp - hero_hp)
            hp_drop_recent_pct = hp_drop / self._dj_prev_frame_hp_max

        if s2_used:
            # 滥用检测：满血且无承伤 → 瞎交保命技
            if hero_hp_rate > S2_ABUSE_HP_THRESH and hp_drop_recent_pct < S2_RECENT_DMG_THRESH:
                inc -= 0.05

            # 保命奖励：残血交二 → 正确时机
            if hero_hp_rate < S2_SURVIVAL_HP_THRESH:
                inc += 0.5

            # 记录使用帧号，供后续反杀检测
            self._dj_s2_used_frame = frame_no

        # 反杀检测：S2 使用后的短时间内发生击杀
        if self._dj_s2_used_frame >= 0:
            frames_since_s2 = frame_no - self._dj_s2_used_frame
            cur_kill = float(self._safe_get(hero, "kill_cnt", 0))
            if cur_kill > self._dj_prev_kill_cnt:
                if frames_since_s2 <= S2_COUNTER_KILL_WINDOW:
                    inc += 0.5
                self._dj_s2_used_frame = -1  # 防止同一击杀重复奖励
            elif frames_since_s2 > S2_COUNTER_KILL_WINDOW:
                # 超时未反杀，关闭追踪
                self._dj_s2_used_frame = -1

        # 持久化帧状态
        self._dj_prev_frame_hp = hero_hp
        self._dj_prev_frame_hp_max = hero_hp_max
        self._dj_prev_kill_cnt = float(self._safe_get(hero, "kill_cnt", 0))

        return inc

    def luban_passive_combo_reward(self, frame_no, hero, enemy, used_delta, hit_delta):
        """
        鲁班七号被动连招状态机（独立于狄仁杰的 _combo_active 通道）。

        状态转换：
          IDLE ──(技能1/2/3 使用)──> WAITING ──(扫射命中)──> IDLE (发放奖励)
                                    │
                                    ├──(再次放技能)──> 吞被动惩罚 + 重置窗口
                                    │
                                    └──(超时/死亡)──> IDLE (无奖励)

        关键检测手段：
        - 窗口开启：used_delta[1|2|3] > 0
        - 扫射命中：敌方血量显著下降（>= 3% 最大生命值）
        - 吞被动：窗口期内再次释放技能
        - 卡手：窗口超过 30 帧仍处于交战状态却未平A
        """
        inc = 0.0

        if enemy is None or hero is None:
            self._luban_passive_ready = False
            return inc

        # 安全获取血量
        emy_hp = float(self._safe_get(enemy, "hp", 0))
        emy_hp_max = float(self._safe_get(enemy, "max_hp", 1))
        if emy_hp_max <= 0:
            emy_hp_max = 1.0
        hero_hp = float(self._safe_get(hero, "hp", 0))

        # 任一方死亡 → 强制关闭窗口
        if hero_hp <= 0 or emy_hp <= 0:
            self._luban_passive_ready = False
            return inc

        skill_used = (used_delta[1] > 0 or used_delta[2] > 0 or used_delta[3] > 0)

        # ================================================
        # 状态 0：空闲 —— 等待技能释放以开启窗口
        # ================================================
        if not self._luban_passive_ready:
            if skill_used:
                self._luban_passive_ready = True
                self._luban_passive_open_frame = frame_no
                self._luban_passive_skills_used = 1
                self._luban_passive_enemy_hp_open = emy_hp
                self._luban_passive_enemy_hp_max_open = emy_hp_max
            return inc

        # ================================================
        # 状态 1：等待期 —— 窗口已开启，等待扫射打出
        # ================================================
        frames_elapsed = frame_no - self._luban_passive_open_frame
        MAX_WINDOW = 60      # 最大窗口帧数（~4 秒 @15fps），超过则被动自然消失
        COMBO_WINDOW = 20    # 最佳连招窗口（打得越快奖励越高）
        HOLD_THRESH = 30     # 卡手警告阈值
        COMBO_DMG_THRESH = 0.03  # 敌方掉血 >= 3% max_hp 判定为扫射命中
        SKIP_PENALTY = -0.04     # 吞被动惩罚
        HOLD_PENALTY = -0.01     # 卡手每帧惩罚
        COMBO_BASE = 0.5         # 连招基础奖励
        DECAY_TAU = 10.0         # 衰减常数（帧），越小衰减越快

        # ---- 超时保护 ----
        if frames_elapsed > MAX_WINDOW:
            self._luban_passive_ready = False
            return inc

        # ---- 检测 1：吞被动（窗口期内再次释放技能） ----
        if skill_used:
            inc += SKIP_PENALTY
            self._luban_passive_skills_used += 1
            # 重置窗口基准：新技能重新刷新了被动，以当前帧为新起点
            self._luban_passive_open_frame = frame_no
            self._luban_passive_enemy_hp_open = max(emy_hp, self._luban_passive_enemy_hp_open or 0)
            self._luban_passive_enemy_hp_max_open = emy_hp_max

        # ---- 辅助：若敌方回血，上修基线防止假阳性 ----
        if emy_hp > (self._luban_passive_enemy_hp_open or 0):
            self._luban_passive_enemy_hp_open = emy_hp

        # ---- 检测 2：扫射命中（敌方血量显著下降） ----
        hp_drop = (self._luban_passive_enemy_hp_open or 0) - emy_hp
        hp_drop_pct = hp_drop / self._luban_passive_enemy_hp_max_open

        if hp_drop_pct >= COMBO_DMG_THRESH and frames_elapsed <= COMBO_WINDOW:
            # 连招成功！指数衰减：帧数越小奖励越高
            combo_bonus = COMBO_BASE * math.exp(-frames_elapsed / DECAY_TAU)
            inc += combo_bonus
            self._luban_passive_ready = False
            return inc

        # ---- 检测 3：卡手惩罚（交战状态下长时间不A） ----
        if frames_elapsed > HOLD_THRESH:
            in_combat = self._luban_is_in_combat(hero, enemy)
            if in_combat:
                inc += HOLD_PENALTY

        return inc

    def _luban_is_in_combat(self, hero, enemy):
        """判断是否处于交战状态：敌方存活且双方距离 < 15000（同屏可视范围）。"""
        if hero is None or enemy is None:
            return False
        try:
            h_loc = hero.get("location") or {}
            e_loc = enemy.get("location") or {}
            hx = float(self._safe_get(h_loc, "x", 0))
            hz = float(self._safe_get(h_loc, "z", 0))
            ex = float(self._safe_get(e_loc, "x", 0))
            ez = float(self._safe_get(e_loc, "z", 0))
            # 跳过死亡传回值 100000
            if abs(hx) > 90000 or abs(hz) > 90000 or abs(ex) > 90000 or abs(ez) > 90000:
                return False
            return math.hypot(hx - ex, hz - ez) < 15000.0
        except Exception:
            return False

    def _step_no(self, frame_no):
        """将帧号转化为宏观决策步(step)，默认引擎1秒15帧，决策频率大概是2-3帧一次"""
        step_len = 6
        return int(frame_no // step_len)

    def _skill_events_this_frame(self, hero):
        """核心辅助：捕捉当前这 1 帧内的技能是否刚刚按下或命中"""
        used_delta = [0]*7
        hit_delta  = [0]*7
        slots = (hero.get("skill_state", {}) or {}).get("slot_states", []) or []

        for i in range(min(7, len(slots))):
            level = int(slots[i].get("level", 0) or 0)
            u = int(slots[i].get("usedTimes", 0) or 0)
            h = int(slots[i].get("hitHeroTimes", 0) or 0)
            if level > 0:
                used_delta[i] = max(0, u - self._skill_prev_used[i])
                hit_delta[i]  = max(0, h - self._skill_prev_hit[i])
            self._skill_prev_used[i] = u
            self._skill_prev_hit[i]  = h

        return used_delta, hit_delta

    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        """
        纯几何的推进计算：英雄越靠近敌方防御塔，返回值越大。
        【修复】：使用缓存坐标，防止防御塔被摧毁当帧坐标为 None 导致推进奖励断崖式跌落。
        """
        if main_hero is None:
            return 0.0
            
        # 只要塔还在，就更新缓存的坐标
        if main_tower is not None:
            self.cached_main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        if enemy_tower is not None:
            self.cached_enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
            
        # 如果缓存里还没有数据（比如刚开局第一帧），直接返回 0
        if self.cached_main_tower_pos is None or self.cached_enemy_tower_pos is None:
            return 0.0

        hero_pos = (main_hero["location"]["x"], main_hero["location"]["z"])
        
        # 使用缓存坐标计算距离，即使塔没了，也会算到“塔的废墟”的距离
        dist_hero2emy = math.dist(hero_pos, self.cached_enemy_tower_pos)
        dist_main2emy = math.dist(self.cached_main_tower_pos, self.cached_enemy_tower_pos)
        
        # 归一化的推进量
        forward_value = (dist_main2emy - dist_hero2emy) / max(dist_main2emy, 1.0)
        return forward_value

    def _find_nearest_cake(self, frame_data, camp):
        """在 npc_states 中寻找距离我方英雄最近的蛋糕/血包。
        返回 (距离, 是否存在, runtime_id)。蛋糕不存在时距离为 999999.0, id 为 None。
        """
        main_hero = None
        for hero in frame_data.get("hero_states", []):
            if hero.get("camp") == camp:
                main_hero = hero
                break

        if main_hero is None:
            return 999999.0, False, None

        hero_loc = main_hero.get("location", {})
        hx = float(self._safe_get(hero_loc, "x", 0))
        hz = float(self._safe_get(hero_loc, "z", 0))

        min_dist = 999999.0
        found = False
        found_id = None

        for npc in frame_data.get("npc_states", []):
            if npc.get("hp", 0) <= 0:
                continue
            sub = int(npc.get("sub_type", -1) or -1)
            at = str(npc.get("actor_type", ""))
            # 排除小兵和防御塔
            if sub in (11, 21) or str(sub) in ("11", "21"):
                continue
            if at in ("ACTOR_SUB_SOLDIER", "ACTOR_SUB_TOWER"):
                continue

            # 蛋糕特征：sub_type 在已知范围内，或 actor_type 中有 cake/organ
            is_cake = (sub in (12, 13, 14, 22, 24, 31, 32))
            if not is_cake:
                at_upper = at.upper()
                if "CAKE" not in at_upper and "ORGAN" not in at_upper and "CHERRY" not in at_upper:
                    continue

            loc = npc.get("location", {})
            nx = float(self._safe_get(loc, "x", 0))
            nz = float(self._safe_get(loc, "z", 0))
            dist = math.hypot(hx - nx, hz - nz)

            if dist < min_dist:
                min_dist = dist
                found = True
                found_id = npc.get("runtime_id")

        return min_dist, found, found_id

    def _check_cake_disappeared(self, frame_data, prev_cake_id):
        """检查上一帧追踪的蛋糕 NPC 是否在本帧消失（被拾取）。
        返回 True 表示蛋糕消失了。
        """
        if prev_cake_id is None:
            return False
        for npc in frame_data.get("npc_states", []):
            if npc.get("runtime_id") == prev_cake_id:
                if npc.get("hp", 0) > 0:
                    return False  # 蛋糕还在且存活
        # 蛋糕不在 npc_states 中，或 hp <= 0
        return True

    def _calc_last_hit_from_dead_action(self, frame_data, main_hero, enemy_hero):
        """从 frame_action[dead_action] 精确计算补刀奖励/惩罚。
        我方英雄杀敌方小兵 +1.2; 我方小兵/塔杀敌方小兵 -0.3 (漏刀);
        敌方英雄杀我方小兵 -1.1 (被压制).
        """
        reward = 0.0
        frame_action = frame_data.get("frame_action", {}) or {}
        dead_actions = frame_action.get("dead_action", []) or []

        main_id = main_hero.get("runtime_id") if main_hero else None
        enemy_id = enemy_hero.get("runtime_id") if enemy_hero else None

        for da in dead_actions:
            killer = (da.get("killer") or {})
            victim = (da.get("death") or {})

            killer_id = killer.get("runtime_id")
            killer_type = str(killer.get("sub_type", ""))
            victim_type = str(victim.get("sub_type", ""))

            # 只处理小兵死亡事件
            if victim_type not in ("ACTOR_SUB_SOLDIER", "11"):
                continue

            if killer_id == main_id:
                # 我方英雄补刀敌方小兵 → 正反馈
                reward += 1.2
            elif killer_id == enemy_id:
                # 敌方英雄补刀我方小兵 → 惩罚
                reward -= 1.1
            elif killer_type in ("ACTOR_SUB_SOLDIER", "11", "ACTOR_SUB_TOWER", "21"):
                # 小兵/防御塔击杀小兵（漏刀）→ 轻度惩罚
                reward -= 0.3

        return reward

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
                
                # 更新自身轨迹记录，用于防发呆检测
                hero_x = self._safe_get(hero["location"], "x", 0)
                hero_z = self._safe_get(hero["location"], "z", 0)
                self.pos_window.append((hero_x, hero_z))
                if len(self.pos_window) > 15:
                    self.pos_window.pop(0)
            else:
                enemy_camp = hero["camp"]
                
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict, action=None):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():

            # 共享变量，避免各奖励项重复遍历帧数据
            main_hp_ratio = self.m_main_calc_frame_map["hp_point"].cur_frame_value
            enemy_hp_ratio = self.m_enemy_calc_frame_map["hp_point"].cur_frame_value
            enemy_alive = enemy_hp_ratio > 0
            is_attacking = self._is_attacking_frame(action)

            # =========================================================
            # 机制 1：血线门控的推进奖励 (HP-Gated Forward)
            # =========================================================
            if reward_name == "forward":
                cur_forward = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last_forward = self.m_main_calc_frame_map[reward_name].last_frame_value
                forward_delta = cur_forward - last_forward

                if not enemy_alive:
                    if main_hp_ratio > 0.3:
                        # 敌方阵亡且我方健康：激发推塔欲望，放大 5 倍
                        reward_struct.value = forward_delta * 5.0
                        # P3: 击杀后推塔放大 — 在敌塔12000范围内额外激励
                        if self.cached_enemy_tower_pos:
                            main_hero_pos = None
                            for hero in frame_data.get("hero_states", []):
                                if hero["runtime_id"] == self.main_hero_player_id:
                                    main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                                    break
                            if main_hero_pos:
                                dist_to_enemy_tower = math.dist(main_hero_pos, self.cached_enemy_tower_pos)
                                if dist_to_enemy_tower < 12000.0:
                                    t = 1.0 - dist_to_enemy_tower / 12000.0
                                    reward_struct.value += 0.03 * t
                    else:
                        # 敌方阵亡但我方残血：触发求生欲，禁止推进
                        if forward_delta > 0:
                            # 试图走向敌方基地，给予明确负面惩罚斩断贪念
                            reward_struct.value = -0.01
                        else:
                            # 后退或原地回城，不予惩罚
                            reward_struct.value = 0.0
                else:
                    # 敌方存活时的常规推进奖励
                    reward_struct.value = forward_delta

            # =========================================================
            # 机制 2：安全回城的显式动作奖励 (Explicit Recall Reward)
            # =========================================================
            elif reward_name == "recall":
                # 获取双方英雄坐标
                main_hero_pos, enemy_hero_pos = None, None
                for hero in frame_data.get("hero_states", []):
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                    else:
                        enemy_hero_pos = (hero["location"]["x"], hero["location"]["z"])

                # 获取防御塔坐标，判定英雄是否在己方塔后
                main_tower_pos = None
                enemy_tower_pos = None
                for organ in frame_data.get("npc_states", []):
                    if organ["sub_type"] == 21:
                        if organ["camp"] == self.main_hero_camp:
                            main_tower_pos = (organ["location"]["x"], organ["location"]["z"])
                        else:
                            enemy_tower_pos = (organ["location"]["x"], organ["location"]["z"])

                behind_tower = False
                if main_hero_pos and main_tower_pos and enemy_tower_pos:
                    hero_to_enemy_tower = math.dist(main_hero_pos, enemy_tower_pos)
                    main_to_enemy_tower = math.dist(main_tower_pos, enemy_tower_pos)
                    behind_tower = hero_to_enemy_tower > main_to_enemy_tower

                is_safe = (not enemy_alive) or behind_tower

                if main_hp_ratio <= 0.3 and is_safe and action is not None:
                    if action[0] == GameConfig.RECALL_BUTTON_INDEX:
                        self.consecutive_recall_frames += 1
                        if self.consecutive_recall_frames >= 2:
                            reward_struct.value = 0.05
                        else:
                            reward_struct.value = 0.0
                    else:
                        self.consecutive_recall_frames = 0
                        reward_struct.value = 0.0
                else:
                    self.consecutive_recall_frames = 0
                    reward_struct.value = 0.0

            # =========================================================
            # 【核心战术 1】：极限拉扯与白嫖奖励 (HP Trade Bonus)
            # =========================================================
            elif reward_name == "hp_trade":
                main_hp_delta = (self.m_main_calc_frame_map["hp_point"].cur_frame_value - 
                                 self.m_main_calc_frame_map["hp_point"].last_frame_value)
                enemy_hp_delta = (self.m_enemy_calc_frame_map["hp_point"].cur_frame_value - 
                                  self.m_enemy_calc_frame_map["hp_point"].last_frame_value)

                trade_bonus = 0.0
                if enemy_hp_delta < 0: # 只要打到了敌方
                    if main_hp_delta >= 0:
                        # 神级操作：完美拉扯，白嫖血量！给 1.5 倍爆发奖励
                        trade_bonus = -enemy_hp_delta * 1.5
                    elif enemy_hp_delta < main_hp_delta:
                        # 高阶操作：对点换血，但我方扣得比敌方少
                        trade_bonus = -(enemy_hp_delta - main_hp_delta) * 0.5
                reward_struct.value = trade_bonus

            elif reward_name in ("passive", "skill1", "skill2", "skill3", "skill5_flash", "hero_combo_window", "kill_gold_consistency", "kill_tower_consistency", "cake_hunt", "cake_pickup", "last_hit"):
                # 单边奖励：只看我方增量，不和敌方做差
                reward_struct.cur_frame_value  = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # =========================================================
            # 【核心战术 3】：防站桩/怠惰惩罚 (Anti-Camping)
            # =========================================================
            elif reward_name == "anti_camp":
                camp_penalty = 0.0
                if len(self.pos_window) == 15:
                    start_x, start_z = self.pos_window[0]
                    end_x, end_z = self.pos_window[-1]
                    dist = math.sqrt((start_x - end_x)**2 + (start_z - end_z)**2)

                    if dist < GameConfig.ANTI_CAMP_MIN_DIST:
                        dist_factor = 1.0 - dist / GameConfig.ANTI_CAMP_MIN_DIST
                        hp_center = (GameConfig.ANTI_CAMP_HP_UPPER + GameConfig.ANTI_CAMP_HP_LOWER) / 2.0
                        hp_span = (GameConfig.ANTI_CAMP_HP_UPPER - GameConfig.ANTI_CAMP_HP_LOWER) / 6.0
                        hp_factor = self._sigmoid((main_hp_ratio - hp_center) / max(hp_span, 0.01))
                        camp_penalty = GameConfig.ANTI_CAMP_MAX_PENALTY * dist_factor * hp_factor
                reward_struct.value = camp_penalty
            elif reward_name == "kiting":
                kiting_bonus = 0.0
                main_hero_pos, enemy_hero_pos = None, None
                for hero in frame_data["hero_states"]:
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                    else:
                        enemy_hero_pos = (hero["location"]["x"], hero["location"]["z"])

                if main_hero_pos and enemy_hero_pos and enemy_alive:
                    dist_enemy = math.dist(main_hero_pos, enemy_hero_pos)

                    # 追逐动态缩放：敌方越残，危险区边界越小，鼓励追击收割
                    chase_weight = max(0.0, 1.0 - enemy_hp_ratio / GameConfig.KITING_CHASE_HP_THRESHOLD)
                    eff_danger = GameConfig.KITING_DANGER_DIST * (1.0 - chase_weight)
                    eff_optimal_min = GameConfig.KITING_OPTIMAL_MIN * (1.0 - chase_weight)

                    if dist_enemy <= eff_danger:
                        t = (dist_enemy / eff_danger) if eff_danger > 1.0 else 1.0
                        raw_penalty = -GameConfig.KITING_DIST_COEFF * 2.0 * (1.0 - t)
                        kiting_bonus = 0.0 if is_attacking else raw_penalty

                    elif dist_enemy <= eff_optimal_min:
                        span = eff_optimal_min - eff_danger
                        t = ((dist_enemy - eff_danger) / span) if span > 1.0 else 1.0
                        kiting_bonus = GameConfig.KITING_DIST_COEFF * t

                    elif dist_enemy <= GameConfig.KITING_OPTIMAL_MAX:
                        kiting_bonus = GameConfig.KITING_DIST_COEFF

                    elif dist_enemy <= GameConfig.IDLE_PENALTY_DIST:
                        span = GameConfig.IDLE_PENALTY_DIST - GameConfig.KITING_OPTIMAL_MAX
                        t = ((dist_enemy - GameConfig.KITING_OPTIMAL_MAX) / span) if span > 1.0 else 1.0
                        kiting_bonus = GameConfig.KITING_DIST_COEFF * (1.0 - t)

                    else:
                        # 距离敌人 > IDLE_PENALTY_DIST：怠惰惩罚
                        if is_attacking:
                            kiting_bonus = 0.0
                        elif main_hp_ratio > GameConfig.IDLE_PENALTY_HP_THRESHOLD:
                            kiting_bonus = GameConfig.IDLE_PENALTY_VALUE
                        else:
                            kiting_bonus = 0.0
                reward_struct.value = kiting_bonus

            # ---------- 基础状态零和差分 ----------
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value

        # ================================================================
        # P0 & P1: 推塔即时反馈 + 拆塔里程碑奖励
        #
        # P1 — 塔伤害即时反馈: 当己方英雄在敌方塔附近(15000内)且塔本帧
        #      受到伤害时，按掉血比例给予即时奖励。解决了原来 "攻击→等一帧
        #      →HP下降→tower_hp_point差分" 的延迟问题。
        # P0 — 拆塔里程碑: 敌方塔血量首次归零时给予 +10.0 一次性奖励，
        #      直接强化胜利条件。
        # ================================================================
        enemy_tower_hp_ratio = self.m_enemy_calc_frame_map["tower_hp_point"].cur_frame_value
        if self._prev_enemy_tower_hp_ratio > 0:
            tower_damage_ratio = self._prev_enemy_tower_hp_ratio - enemy_tower_hp_ratio

            # P1: 塔伤害即时反馈 (仅在英雄靠近敌方塔时触发)
            if tower_damage_ratio > 0:
                hero_near_enemy_tower = False
                main_hero_pos = None
                for hero in frame_data.get("hero_states", []):
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                        break
                if main_hero_pos and self.cached_enemy_tower_pos:
                    dist_to_enemy_tower = math.dist(main_hero_pos, self.cached_enemy_tower_pos)
                    hero_near_enemy_tower = dist_to_enemy_tower < 15000.0

                if hero_near_enemy_tower:
                    tower_damage_reward = tower_damage_ratio * 3.0
                    reward_dict["tower_damage"] = tower_damage_reward
                    reward_sum += tower_damage_reward

            # P0: 拆塔里程碑奖励 (敌方塔从存活变为被摧毁)
            if enemy_tower_hp_ratio <= 0:
                reward_dict["tower_destroy"] = 10.0
                reward_sum += 10.0

        self._prev_enemy_tower_hp_ratio = enemy_tower_hp_ratio
        reward_dict["reward_sum"] = reward_sum