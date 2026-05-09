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
    # 【战术升级】：强行注入高阶微操奖励项
    # =========================================================
    calc_frame_map["hp_trade"] = RewardStruct(3.0)  # 拉扯与白嫖奖励
    calc_frame_map["last_hit"] = RewardStruct(2.0)  # 补刀瞬时刺激
    calc_frame_map["anti_camp"] = RewardStruct(1.0) # 防发呆/站桩惩罚
    calc_frame_map["kiting"] = RewardStruct(1.0)    # 极限拉扯奖励
    calc_frame_map["recall"] = RewardStruct(1.0)
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
        self.cached_main_tower_pos = None
        self.cached_enemy_tower_pos = None
        self._combo_active = False          
        self._combo_end_step = -10**9       
        self._combo_enemy_hp_prev = None    
        self._combo_hero_pos_prev = None    
        # 用于记录上一帧的技能使用情况，防止增量计算报错
        self._skill_prev_used = [0]*7
        self._skill_prev_hit  = [0]*7

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
    def _safe_get(d, key, default=0):
        if d is None:
            return default
        value = d.get(key, default)
        return default if value is None else value

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
                    my_kills   = main_hero.get("killCnt", 0)
                    emy_kills  = enemy_hero.get("killCnt", 0) if enemy_hero else 0
                    my_gold    = main_hero.get("moneyCnt", 0)
                    emy_gold   = enemy_hero.get("moneyCnt", 0) if enemy_hero else 0

                    kill_diff  = my_kills - emy_kills
                    gold_diff  = my_gold  - emy_gold

                    # 阈值设置：如果我方人头领先 >= 1，但经济差却 <= 0（说明在无效打架漏兵线）
                    KILL_LEAD_THRESH = 1   
                    PENALTY = -0.02        # 每帧给一个小惩罚

                    inc = PENALTY if (kill_diff >= KILL_LEAD_THRESH and gold_diff <= 0) else 0.0
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            # ==========================================
            # 宏观节奏纠偏 2：击杀但不推塔惩罚
            # ==========================================
            elif reward_name == "kill_tower_consistency":
                if is_main_side:
                    def pct(hp, mx):
                        return (hp / float(mx)) if mx and mx > 0 else 1.0

                    my_tower_pct   = pct(main_tower["hp"],  main_tower["max_hp"])  if main_tower  else 1.0
                    emy_tower_pct  = pct(enemy_tower["hp"], enemy_tower["max_hp"]) if enemy_tower else 1.0
                    
                    # 当前推塔净进展（我方塔血量百分比 - 敌方塔血量百分比）
                    tower_pressure_now = my_tower_pct - emy_tower_pct   

                    my_kills   = main_hero.get("killCnt", 0)
                    emy_kills  = enemy_hero.get("killCnt", 0) if enemy_hero else 0
                    kill_diff  = my_kills - emy_kills

                    # 阈值设置：如果人头领先，但推塔进度 <= 1% 的微小缓冲值（说明杀完人就发呆/回城，不推线）
                    KILL_LEAD_THRESH = 1
                    BUFFER = 0.01          
                    PENALTY = -0.02

                    inc = PENALTY if (kill_diff >= KILL_LEAD_THRESH and tower_pressure_now <= BUFFER) else 0.0
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value
            else:
                # 兼容新增的自定义键 (hp_trade等)，占位设为0即可
                reward_struct.cur_frame_value = 0.0
    
    def calc_hero_combo_reward(self, frame_no, hero, enemy, used_delta, hit_delta):
        """
        根据英雄类型，动态计算高阶连招与状态窗口奖励。
        - 狄仁杰(133): 大招命中后进入破甲爆发窗口，空大惩罚。
        - 鲁班(112): 放技能后进入被动扫射窗口，站桩输出给巨奖，乱动打断扫射给惩罚。
        """
        inc = 0.0
        now_step = self._step_no(frame_no)
        
        # 获取敌方当前血量，用于计算窗口内的爆发伤害
        emy_hp = (enemy or {}).get("actor_state", {}).get("hp", 0)
        emy_hp_max = (enemy or {}).get("actor_state", {}).get("max_hp", 1) or 1
        
        # 尝试获取英雄配置ID，判断是鲁班还是狄仁杰 (数据协议中通常在 actor_state 里面)
        config_id = hero.get("actor_state", {}).get("config_id", 0)

        # ==========================================
        # 英雄 1：狄仁杰 (133) - 大招破甲爆发流
        # ==========================================
        if config_id == 133:
            used3 = used_delta[3] > 0
            hit3 = hit_delta[3] > 0
            
            # 1. 大招事件判定
            if used3:
                if hit3:
                    # 大招命中，开启 20 步（约 2.5 秒）的破甲爆发窗口
                    self._combo_active = True
                    self._combo_end_step = now_step + 20
                    self._combo_enemy_hp_prev = emy_hp
                else:
                    # 大招空了，给一个较大的动作惩罚，教 AI 捏死大招别乱放
                    inc -= 0.1 

            # 2. 窗口内收益结算
            if self._combo_active and now_step <= self._combo_end_step:
                if self._combo_enemy_hp_prev is not None and emy_hp_max > 0:
                    delta_hp = max(0, self._combo_enemy_hp_prev - emy_hp)
                    if delta_hp > 0:
                        dmg_pct = delta_hp / float(emy_hp_max)
                        # 破甲期间造成的所有伤害，给予 2.0 倍权重的超额奖励！鼓励疯狂虚空平A接技能
                        inc += 2.0 * dmg_pct
                        
                # 破甲期间如果接上 1、2 技能命中，额外给连招小奖
                if hit_delta[1] > 0 or hit_delta[2] > 0:
                    inc += 0.05
                    
                self._combo_enemy_hp_prev = emy_hp
                
            elif now_step > self._combo_end_step:
                self._combo_active = False


        # ==========================================
        # 英雄 2：鲁班七号 (112) - 站桩被动扫射流
        # ==========================================
        elif config_id == 112:
            # 只要使用了1、2、3任意技能，就会触发被动扫射
            used_any_skill = used_delta[1] > 0 or used_delta[2] > 0 or used_delta[3] > 0
            
            # 1. 技能释放事件
            if used_any_skill:
                # 开启 8 步（约 1 秒）的扫射窗口
                self._combo_active = True
                self._combo_end_step = now_step + 8
                self._combo_enemy_hp_prev = emy_hp
                # 记录开扫射时的原点位置，用于检测走位是否打断扫射
                hero_x = hero.get("actor_state", {}).get("location", {}).get("x", 0)
                hero_z = hero.get("actor_state", {}).get("location", {}).get("z", 0)
                self._combo_hero_pos_prev = (hero_x, hero_z)

            # 2. 窗口内收益与惩罚结算
            if self._combo_active and now_step <= self._combo_end_step:
                # 首先检测是否走位过度打断了被动（在王者荣耀里，移动轮盘会取消鲁班扫射）
                curr_x = hero.get("actor_state", {}).get("location", {}).get("x", 0)
                curr_z = hero.get("actor_state", {}).get("location", {}).get("z", 0)
                
                if self._combo_hero_pos_prev:
                    dist_moved = math.dist((curr_x, curr_z), self._combo_hero_pos_prev)
                    if dist_moved > 1500:  # 这个阈值代表轻微挪动，超过说明AI摇动了方向盘
                        inc -= 0.05  # 惩罚打断扫射
                        self._combo_active = False  # 窗口提前结束
                        return inc
                
                # 如果乖乖站好打出了伤害
                if self._combo_enemy_hp_prev is not None and emy_hp_max > 0:
                    delta_hp = max(0, self._combo_enemy_hp_prev - emy_hp)
                    if delta_hp > 0:
                        dmg_pct = delta_hp / float(emy_hp_max)
                        # 扫射是鲁班核心输出，打出伤害给 1.5 倍奖励
                        inc += 1.5 * dmg_pct
                        
                self._combo_enemy_hp_prev = emy_hp

            elif now_step > self._combo_end_step:
                self._combo_active = False

        return inc
    
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

            # =========================================================
            # 机制 1：血线门控的推进奖励 (HP-Gated Forward)
            # =========================================================
            if reward_name == "forward":
                cur_forward = self.m_main_calc_frame_map[reward_name].cur_frame_value
                last_forward = self.m_main_calc_frame_map[reward_name].last_frame_value
                forward_delta = cur_forward - last_forward

                hp_rate = self.m_main_calc_frame_map["hp_point"].cur_frame_value
                enemy_hp = self.m_enemy_calc_frame_map["hp_point"].cur_frame_value
                enemy_alive = enemy_hp > 0

                if not enemy_alive:
                    if hp_rate > 0.3:
                        # 敌方阵亡且我方健康：激发推塔欲望，放大 5 倍
                        reward_struct.value = forward_delta * 5.0
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
                hp_rate = self.m_main_calc_frame_map["hp_point"].cur_frame_value
                enemy_hp = self.m_enemy_calc_frame_map["hp_point"].cur_frame_value
                enemy_alive = enemy_hp > 0
                
                # 获取双方坐标计算距离
                main_hero_pos, enemy_hero_pos = None, None
                for hero in frame_data.get("hero_states", []):
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                    else:
                        enemy_hero_pos = (hero["location"]["x"], hero["location"]["z"])

                dist_enemy = 999999.0
                if main_hero_pos and enemy_hero_pos:
                    dist_enemy = math.dist(main_hero_pos, enemy_hero_pos)

                # 判断绝对安全环境（引擎尺度下，普通射程约8000，12000以上属于绝对安全区）
                is_safe = (not enemy_alive) or (dist_enemy > 12000.0)

                # 条件触发：残血 + 环境安全 + 按下回城键
                if hp_rate <= 0.3 and is_safe and action is not None:
                    # 提取 button action
                    if action[0] == GameConfig.RECALL_BUTTON_INDEX:
                        self.consecutive_recall_frames += 1
                        # 核心修复点：要求必须连续不间断吟唱 50 帧（约几秒）才判定为真实回城意图
                        if self.consecutive_recall_frames == 50:
                            reward_struct.value = 1.0  # 给予一次性大额奖励
                        else:
                            reward_struct.value = 0.0  # 吟唱期间（或已发过奖励后）不给分
                    else:
                        self.consecutive_recall_frames = 0
                        reward_struct.value = 0.0
                else:
                    # 不满足条件（例如血量回上来了，或者敌人靠近了）清空打断计数
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

            # =========================================================
            # 【核心战术 2】：补刀/经济跃升瞬间奖励 (Last-hit Bonus)
            # =========================================================
            elif reward_name == "last_hit":
                money_delta = (self.m_main_calc_frame_map["money"].cur_frame_value - 
                               self.m_main_calc_frame_map["money"].last_frame_value)
                hit_bonus = 0.0
                # 自然跳钱大概在1~3，当金币单帧突增大于 15 时，必然是击杀了小兵/英雄/防御塔
                if money_delta > 15.0:
                    hit_bonus = money_delta / 100.0  # 给予强烈的瞬时正反馈
                reward_struct.value = hit_bonus
            
            elif reward_name in ("passive", "skill1", "skill2", "skill3", "skill5_flash", "hero_combo_window", "kill_gold_consistency", "kill_tower_consistency"):
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
                    main_hp_cur = self.m_main_calc_frame_map["hp_point"].cur_frame_value
                    
                    # 惩罚条件：过去15帧(近1秒)没怎么动(距离<1000)，且血量健康(>70%)
                    # 如果残血，可能是躲在塔下回城，予以宽容；如果满血还在发呆，重罚！
                    if dist < 1000 and main_hp_cur > 0.7:
                        camp_penalty = -0.05
                reward_struct.value = camp_penalty
            elif reward_name == "kiting":
                kiting_bonus = 0.0
                main_hero_pos, enemy_hero_pos = None, None
                enemy_hp = 0

                # 从当前帧数据中遍历获取双方位置和敌方血量
                for hero in frame_data["hero_states"]:
                    if hero["runtime_id"] == self.main_hero_player_id:
                        main_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                    else:
                        enemy_hero_pos = (hero["location"]["x"], hero["location"]["z"])
                        enemy_hp = float(self._safe_get(hero, "hp", 0))

                # 必须确认拿到了坐标，且敌方存活才计算奖励
                if main_hero_pos and enemy_hero_pos and enemy_hp > 0:
                    dist_enemy = math.dist(main_hero_pos, enemy_hero_pos)
                    if 6000 <= dist_enemy <= 8500:
                        kiting_bonus = 0.02   # 处于极佳射程，持续给微小正反馈
                    elif dist_enemy < 4000:
                        kiting_bonus = -0.05  # 距离过近，面临被秒杀风险，重罚
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
            
        reward_dict["reward_sum"] = reward_sum