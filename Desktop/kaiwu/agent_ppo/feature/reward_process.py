#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig
from collections import deque


# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True




# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id,logger):
        self.main_hero_player_id = main_hero_runtime_id
        # 监控主英雄的阵营 0-1（PLAYCAMP0-PLAYCAMP1）
        self.main_hero_camp = -1
        # 代表主英雄血量，代码没有使用
        self.main_hero_hp = -1
        # 代表主英雄防御塔血量，代码没有使用
        self.main_hero_organ_hp = -1
        # 全局的奖励字典，经过get_result函数进行初始化，每次初始化都是当前帧
        self.m_reward_value = {}
        # 代表上一帧号，代码没有使用
        self.m_last_frame_no = -1
        # 这个代表的是权重状态map，和下面两个的区别就是没有带前后帧值
        self.m_cur_calc_frame_map = init_calc_frame_map()
        # 主英雄的状态帧map
        self.m_main_calc_frame_map = init_calc_frame_map()
        # 敌方英雄的状态帧map
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        # 时间折扣因子
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        # 代表当前英雄的id号，代码没有使用
        self.m_main_hero_config_id = -1
        # 初始化英雄各个等级的最大经验值字典
        self.m_each_level_max_exp = {}

        self.logger = logger

        self._last_flash_cd = 0.0
        self._flash_seen = False

        # —— 技能增量跟踪（0..6 槽，对应你的槽位表）——
        self._skill_prev_used = [0]*7
        self._skill_prev_hit  = [0]*7

        # —— 孙尚香二技能“破甲窗口”状态（按 step 管理）——
        self._sxx2_active = False
        self._sxx2_open_step = -10**9
        self._sxx2_end_step  = -10**9
        self._sxx2_enemy_hp_prev = None   # 窗口内上一帧敌方HP（按帧采样，防止漏伤害）
        self._sxx2_dmg_acc = 0.0          # 窗口内累计敌方 HP% 下降
        self._sxx2_had_follow = False     # 窗口内是否出现过“有效跟进”（命中或伤害）
        self._sxx2_first_follow_step = None
        self._sxx2_last_3_step = -10**9   # 最近一次 3 技能的 step（用于收割窗口）

        # —— S1 强化普攻“卡→释放” 状态（按 step）——
        self._s1_enh_ready = False        # 是否持有强化（S1 使用后获得）
        self._s1_open_step = -10**9       # 获得强化的 step（使用 S1 的步号）
        self._s1_ready_step = None        # S1 冷却比例首次 ≤ 阈值 的步号（进入“应当释放期”）
        self._s1_chain_expect_end = -10**9  # 成功释放后，期待再次 S1 的窗口结束 step

        # —— 对抗态判定用到的标记（不用 behav_mode）——
        self._last_attack_or_cast_step = -10**9  # 任一方的技能 used/hit 事件最近发生的步
        self._last_damage_step = -10**9          # 任一方 HP 下降最近发生的步
        self._last_close_step = -10**9           # （可选）双方靠近最近发生的步
        self._last_step_seen = -10**9            # 防止同一 step 重复更新

        self._prev_my_hp = None                  # 上一帧我方 HP
        self._prev_emy_hp = None                 # 上一帧敌方 HP


        self._s1_prev_passive_cd = 0.0   # 记录被动槽位上一次看到的冷却（秒/帧单位按原始数据）
        self._s1_passive_seen = False    # 是否已采过一次（用于跳过第一帧）

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
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

    def result(self, frame_data):
        # 获取当前帧号
        frame_no = frame_data["frameNo"]
        # 初始化英雄的各个等级经验
        self.init_max_exp_of_each_hero()
        # 对英雄的特征数据进行处理
        self.frame_data_process(frame_data)
        # 得到对应的奖励信息
        self.get_reward(frame_data, self.m_reward_value)
        # 如果设置了时间折扣因子，则进行开始奖励退火
        if self.time_scale_arg > 0:
            # 遍历所有的奖励，并且对对应的值进行衰减
            for key in self.m_reward_value:
                # 下面是实现退火逻辑，默认的逻辑函数是0.6^(frame_no/scale_arg)
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value


    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体

        # 主视角英雄和敌人英雄
        main_hero, enemy_hero = None, None
        # 英雄列表
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        # 主英雄当前血量
        main_hero_hp = main_hero["actor_state"]["hp"]
        # 主英雄最大血量
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        # 主英雄法力值
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        # 主英雄最大法力值
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # Get both defense towers
        # 获取双方防御塔
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            # 防御塔阵营
            organ_camp = organ["camp"]
            # 防御塔类型,有sprint-水晶， tower-防御塔
            organ_subtype = organ["sub_type"]
            # 下面的逻辑是确定此时的防御塔阵营
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ

        
        is_main_side = (camp == self.main_hero_camp)
        if is_main_side:
            used_delta, hit_delta = self._skill_events_this_frame(main_hero)
        else:
            used_delta, hit_delta = [0]*7, [0]*7
        if is_main_side:
            # 每 step 更新一次“对抗态”迹象（不依赖 behav_mode）
            self._update_combat_markers(self._step_no(frame_data.get("frameNo", 0)),
                                    main_hero, enemy_hero, used_delta, hit_delta, None, None)
        
        for reward_name, reward_struct in cul_calc_frame_map.items():
            # 将前一个帧的信息给last_frame_value
            reward_struct.last_frame_value = reward_struct.cur_frame_value


            # 金钱
            if reward_name == "money":
                # 计算当前的经济
                reward_struct.cur_frame_value = main_hero["moneyCnt"]

            # 生命值
            elif reward_name == "hp_point":
                # 计算对应的血量
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))

            # 法力值
            elif reward_name == "ep_rate":
                # 统计法力百分比
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)

            # 击杀
            elif reward_name == "kill":
                # 统计击杀次数
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                # 统计死亡次数
                reward_struct.cur_frame_value = main_hero["deadCnt"]

            # 塔血量
            elif reward_name == "tower_hp_point":
                # 统计塔血量百分比
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                # 先默认初始化当前帧信息为0
                reward_struct.cur_frame_value = 0.0
                # 取出当前的死亡事件信息
                frame_action = frame_data["frame_action"]
                # 如果发生了死亡事件
                if "dead_action" in frame_action:
                    # 取出当前的死亡事件信息
                    dead_actions = frame_action["dead_action"]

                    for dead_action in dead_actions:
                        # 这个部分鼓励我方英雄去杀小兵
                        if (
                            # 杀人者-我方英雄
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            # 死亡者-敌方小兵
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            # 此时当前帧的value + 1
                            reward_struct.cur_frame_value += 1.2
                        elif (
                            # 杀人者-我方小兵或者塔
                            (dead_action["killer"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                             or dead_action["killer"]["sub_type"] == "ACTOR_SUB_TOWER")
                            # 死亡者-敌方小兵
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            # 此时当前帧value + 0.1
                            reward_struct.cur_frame_value -= 0.3
                        # 这个部分阻碍敌方英雄去杀小兵
                        elif (
                            # 杀人者-敌方英雄
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            # 死亡者-我方小兵
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            # 此时当前帧value-1
                            reward_struct.cur_frame_value -= 1.1

            # 经验值
            elif reward_name == "exp":
                # 根据self.calculate_exp_sum获取经验value
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            # elif reward_name == "forward":
                # 根据self.calculate_forward获取forward的value
                # reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            elif reward_name == "passive":
                reward_struct.cur_frame_value = self.skill_passive_reward(main_hero)
            elif reward_name == "skill1":
                reward_struct.cur_frame_value = self.skill_one_reward(main_hero)
            elif reward_name == "skill2":
                reward_struct.cur_frame_value = self.skill_two_reward(main_hero)
            elif reward_name == "skill3":
                reward_struct.cur_frame_value = self.skill_three_reward(main_hero)
            elif reward_name == "skill5_flash":
                if is_main_side:
                    inc = self.skill_five_flash_reward(main_hero,enemy_hero,frame_data)
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            elif reward_name == "kill_gold_consistency":
                # 条件：我方击杀数领先但经济差（我方-敌方）≤ 0 → 当帧给一个很小的负增量
                my_kills   = main_hero.get("killCnt", 0)
                emy_kills  = enemy_hero.get("killCnt", 0) if enemy_hero else 0
                my_gold    = main_hero.get("moneyCnt", 0)
                emy_gold   = enemy_hero.get("moneyCnt", 0) if enemy_hero else 0

                kill_diff  = my_kills - emy_kills
                gold_diff  = my_gold  - emy_gold

                # 阈值与幅度（可调）
                KILL_LEAD_THRESH = 1   # 也可改成 2，避免偶然单杀触发
                PENALTY = -0.02        # 每帧小罚；整局累计不要超过终局奖惩的 10–20%

                inc = PENALTY if (kill_diff >= KILL_LEAD_THRESH and gold_diff <= 0) else 0.0
                reward_struct.cur_frame_value = reward_struct.last_frame_value + inc


            elif reward_name == "kill_tower_consistency":
                # 条件：我方击杀领先但“净塔压”（我塔% - 敌塔%）≤ buffer → 当帧轻罚
                def pct(hp, mx):
                    return (hp / float(mx)) if mx and mx > 0 else 1.0

                my_tower_pct   = pct(main_tower["hp"],  main_tower["max_hp"])  if main_tower  else 1.0
                emy_tower_pct  = pct(enemy_tower["hp"], enemy_tower["max_hp"]) if enemy_tower else 1.0
                tower_pressure_now = my_tower_pct - emy_tower_pct   # 等价于“净推塔进展”的当前快照

                my_kills   = main_hero.get("killCnt", 0)
                emy_kills  = enemy_hero.get("killCnt", 0) if enemy_hero else 0
                kill_diff  = my_kills - emy_kills

                KILL_LEAD_THRESH = 1
                BUFFER = 0.01          # 容忍 1% 的浮动，避免误罚
                PENALTY = -0.02

                inc = PENALTY if (kill_diff >= KILL_LEAD_THRESH and tower_pressure_now <= BUFFER) else 0.0
                reward_struct.cur_frame_value = reward_struct.last_frame_value + inc

            elif reward_name == "sxx_armorbreak_window":
                if is_main_side:
                    inc = self.sxx_armorbreak_reward(frame_data.get("frameNo", 0), main_hero, enemy_hero, used_delta, hit_delta, frame_data)
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value

            elif reward_name == "sxx_s1_enh_aa_timing":
                if is_main_side:
                    inc = self.sxx_s1_enh_aa_reward(frame_data.get("frameNo", 0), main_hero, enemy_hero, used_delta, hit_delta, frame_data)
                    reward_struct.cur_frame_value = reward_struct.last_frame_value + inc
                else:
                    reward_struct.cur_frame_value = reward_struct.last_frame_value


    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        # 通过我方当前的等级去遍历
        for i in range(1, this_hero_info["level"]):
            # 统计累计经验
            exp_sum += self.m_each_level_max_exp[i]
        # 因为前面sum是加到1-当前角色等级-1，所以此处要补充当前的角色等级的经验,这样做的目的是避免当前角色没有升级满，但是加了满经验
        exp_sum += this_hero_info["exp"]
        return exp_sum


    """    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        # 获取主防御塔在地图里面的x和z
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        # 获取敌人防御塔在地图里面的x和z
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        # 获取英雄当前在地图里面的x和z
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        # 初始化forward_value
        forward_value = 0
        # 用math里面的距离函数去计算当前主英雄和敌人防御塔的距离
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        # 计算我方防御塔和敌方防御塔直接的距离
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        # and左边代表英雄血量健康， and右边是此时的主英雄在两个防御塔之间的战争之外
        if main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            # 计算逻辑为（两塔之间的距离 - 主英雄与敌方防御塔的距离）/ 主英雄与敌方防御塔的距离
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy

        return forward_value"""

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        # 获取主英雄的各项信息
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        # 获取敌方英雄的各项信息
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        # 清空当前的奖励字典
        reward_dict.clear()
        # 初始化总奖励和总权重
        reward_sum, weight_sum = 0.0, 0.0
        # 从权重map取出对应的权重名和权重值来着GameConfig里面
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():

            # 英雄血量
            if reward_name == "hp_point":
                if (
                    # 前一帧主英雄和敌方英雄血量都为空的时候
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    # 给reward_struct的前一帧和当前帧初始化为0
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                # 前一帧我方主英雄血量为空
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    # 给reward_struct的前一帧和当前帧初始化为（0 - 敌方血量）
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                # 前一帧敌方英雄血量为空
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    # 给reward_struct的前一帧和当前帧初始化为（我方血量 - 0）
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                # 都不为空
                else:
                    # 给reward_struct的前一帧初始化为（我方血量 - 敌方血量）
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    # 给reward_struct的当前帧初始化为（我方血量 - 敌方血量）
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                # 最后给reward_struct的value赋值为（当前帧的value - 前一帧的value）
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # 法力值
            elif reward_name == "ep_rate":
                # 前一帧和当前帧初始化为主英雄对应的数值
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                # 如果前一帧对应的数值是正数即为大于0
                if reward_struct.last_frame_value > 0:
                    # 给reward_struct的value赋值为（当前帧的value - 前一帧的value）
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                # 如果没了则为0
                else:
                    reward_struct.value = 0

            # 经验值
            elif reward_name == "exp":
                # 默认初始化主英雄为空
                main_hero = None
                # 遍历所有的英雄
                for hero in frame_data["hero_states"]:
                    # 如果此时满足为主英雄则将此时的主英雄指定
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                # 主英雄已经制定了，而且主英雄的等级满了，将reward_struct的value赋值为0
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                # 如果没有满
                else:
                    # 对应的reward_struct的当前帧为（我方经验 - 敌方经验）
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    # 对应的reward_struct的前一帧为（我方经验 - 敌方经验）
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    # 然后将reward_struct的value赋值为（当前帧的value - 前一帧的value）
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # 英雄与塔之间的距离，判断英雄是否进入战场，此为惩罚
            # elif reward_name == "forward":
                # 具体参考奖励函数
                # reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            # 最后一次击杀（对击杀事件的分析）
            elif reward_name == "last_hit":
                # 直接将reward_struct的value赋值给主英雄当前帧对应的值，对应的实现在初始化里面，详细参考set_cur_calc_frame_vec函数，对应179行之后
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name in ("passive", "skill1", "skill2", "skill3","skill5_flash","kill_gold_consistency", "kill_tower_consistency","sxx_armorbreak_window","sxx_s1_enh_aa_timing"):
                # 只看我方；把命中率*10 的平滑变化当成奖励
                reward_struct.cur_frame_value  = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            
            elif reward_name == "time_decay":
                reward_struct.value = -0.002
            # 其他的奖励包括tower_hp_point、money、death、kill、
            else:
                # 对应的reward_struct的当前帧为（我方 - 敌方）
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                # 对应的reward_struct的前一帧为（我方 - 敌方）
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                # 然后将reward_struct的value赋值为（当前帧的value - 前一帧的value）
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            # 权重相加
            weight_sum += reward_struct.weight
            # 奖励*权重 = 总奖励 ， 总奖励累加
            reward_sum += reward_struct.value * reward_struct.weight
            # 将对应的奖励名称的奖励值赋值给奖励字典
            reward_dict[reward_name] = reward_struct.value

            self.logger.info(f"reward_name: {reward_name}, reward_value: {reward_struct.value}, weight: {reward_struct.weight}")

        # 总奖励的值赋值给reward_dict
        reward_dict["reward_sum"] = reward_sum

    def _skill_score(self, slot_state, hit_bonus=1.0, miss_penalty=0.5):
        """
        返回“累计技能评分” = 累计命中 * 命中奖励  -  累计空放 * 空放惩罚
        注意：这不是增量，而是到当前帧为止的累计值；get_reward() 会做帧差得到增量奖励。
        """
        hits = slot_state.get("hitHeroTimes", 0) or 0
        used = slot_state.get("usedTimes", 0) or 0
        misses = max(0, used - hits)
        return hits * hit_bonus - misses * miss_penalty

    def skill_passive_reward(self, hero):
        # slot 0
        s0 = hero["skill_state"]["slot_states"][0]
        return self._skill_score(s0, hit_bonus=1, miss_penalty=0.2)

    def skill_one_reward(self, hero):
        s1 = hero["skill_state"]["slot_states"][1]
        # 命中给 1.0，空放小罚 0.5（按需要再调或交给 REWARD_WEIGHT_DICT）
        return self._skill_score(s1, hit_bonus=1.0, miss_penalty=0.3)

    def skill_two_reward(self, hero):
        s2 = hero["skill_state"]["slot_states"][2]
        # 技能2更关键的话，可稍微加重命中奖励或减轻空放罚
        return self._skill_score(s2, hit_bonus=1.2, miss_penalty=0.5)

    def skill_three_reward(self, hero):
        s3 = hero["skill_state"]["slot_states"][3]
        return self._skill_score(s3, hit_bonus=1.0, miss_penalty=0.5)
    
    def skill_five_flash_reward(self, hero, enemy_hero, frame_data):
        reward = 0.0

        # 读 slot5（闪现）
        s5 = (hero.get("skill_state", {}) or {}).get("slot_states", []) or []
        s5 = s5[5] if len(s5) > 5 else {}
        cd = float(s5.get("cooldown", 0) or 0.0)
        mx = float(s5.get("cooldown_max", 0) or 0.0)
        rate = (cd / mx) if mx > 0 else 0.0

        if not self._flash_seen:
            self._flash_seen = True
            self._last_flash_cd = cd
            return 0.0

        # —— 仅在“刚交闪”的那一帧扣一次（边沿检测）——
        prev_cd = getattr(self, "_last_flash_cd", 0.0)
        used_now = (mx > 0) and (cd > prev_cd + 1e-6) and (rate > 0.8)  # 冷却刚被拉高
        if used_now:
            reward -= 0.07  # 轻惩罚，别设太大（0.05~0.2 先试）

        # —— 击杀/阵亡事件（只有“刚用过”才计）——
        frame_action = frame_data.get("frame_action", {}) or {}
        dead_actions = frame_action.get("dead_action", []) or []

        my_id = (hero.get("actor_state", {}) or {}).get("runtime_id")
        enemy_id = (enemy_hero or {}).get("actor_state", {}).get("runtime_id") if enemy_hero else None

        if rate > 0.8 and enemy_id is not None:
            for da in dead_actions:
                killer = ((da.get("killer") or {}).get("runtime_id"))
                victim = ((da.get("death") or {}).get("runtime_id"))
                if killer == my_id and victim == enemy_id:     # 我方击杀敌方英雄
                    reward += 2.0
                elif killer == enemy_id and victim == my_id:   # 敌方击杀我方
                    reward -= 1.0

        self._last_flash_cd = cd
        return reward


    def _step_no(self, frame_no):
        """帧号 → step 号（向下取整）。"""
        step_len = getattr(GameConfig, "STEP_LEN_FRAMES", 6)
        return int(frame_no // step_len)

    def _skill_events_this_frame(self, hero):
        """
        返回 used_delta[7], hit_delta[7]：各槽位本帧“新增释放/命中”的次数（>=0）。
        对未学习的技能（level==0）不计增量；但依然更新历史计数，避免学会当帧出现巨额差值。
        """
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

    def sxx_armorbreak_reward(self, frame_no, hero, enemy, used_delta, hit_delta, frame_data):
        """
        二技能命中 → 开启/刷新“破甲窗口”（按 step）。
        窗口内：
        - 按敌方 HP% 下降累计奖励（SXX2_DMG_SCALE）
        - 1/3 命中给离散奖励（延迟衰减；只在各自推荐窗口内最香）
        - 若在“3后收割窗口”内击杀，给收割加成
        二技能未命中：轻罚一次
        窗口结束且无有效跟进：小罚一次
        设计为“累计量”→ get_reward() 会取帧差。
        """
        # 槽位索引（你的表：0被动,1一技,2二技,3三技,5闪现）
        IDX_1, IDX_2, IDX_3 = 1, 2, 3

        # 读参数（带默认）
        STEP_LEN  = getattr(GameConfig, "STEP_LEN_FRAMES", 6)
        WIN_STEPS = getattr(GameConfig, "SXX2_WIN_STEPS", 20)  # 你要严格→20步≈120帧
        S1_MAX    = getattr(GameConfig, "SXX2_S1_MAX_STEPS", 5)
        S3_MAX    = getattr(GameConfig, "SXX2_S3_MAX_STEPS", 8)
        FIN_WIN   = getattr(GameConfig, "SXX2_FINISH_WINDOW_STEPS", 8)

        MISS   = getattr(GameConfig, "SXX2_MISS_PENALTY",  -0.05)
        SCALE  = getattr(GameConfig, "SXX2_DMG_SCALE",      1.0)
        B1     = getattr(GameConfig, "SXX2_S1_HIT_BONUS",   0.15)
        B3     = getattr(GameConfig, "SXX2_S3_HIT_BONUS",   0.30)
        KBON   = getattr(GameConfig, "SXX2_KILL_BONUS",     0.8)
        NOPEN  = getattr(GameConfig, "SXX2_NO_FOLLOW_PEN", -0.02)
        EPS    = getattr(GameConfig, "SXX2_MIN_DMG_EPS",    0.01)
        TAU    = getattr(GameConfig, "SXX2_LATENCY_TAU",    4)

        inc = 0.0
        now_step = self._step_no(frame_no)

        # 敌方 HP / MaxHP（按帧采样以尽量捕获伤害）
        emy_hp     = (enemy or {}).get("actor_state", {}).get("hp", 0)
        emy_hp_max = (enemy or {}).get("actor_state", {}).get("max_hp", 1) or 1

        # 本帧技能事件
        used2 = used_delta[IDX_2] > 0; hit2 = hit_delta[IDX_2] > 0
        used1 = used_delta[IDX_1] > 0; hit1 = hit_delta[IDX_1] > 0
        used3 = used_delta[IDX_3] > 0; hit3 = hit_delta[IDX_3] > 0

        # —— 开窗/刷新 —— 
        if used2:
            if hit2:
                if (not self._sxx2_active) or (now_step > self._sxx2_end_step):
                    # 开新窗
                    self._sxx2_active = True
                    self._sxx2_open_step = now_step
                    self._sxx2_end_step  = now_step + WIN_STEPS
                    self._sxx2_enemy_hp_prev = emy_hp
                    self._sxx2_dmg_acc = 0.0
                    self._sxx2_had_follow = False
                    self._sxx2_first_follow_step = None
                else:
                    # 窗内再次2命中 → 刷新窗口
                    self._sxx2_end_step = now_step + WIN_STEPS
                    # 如已出现有效跟进，则给非常小的刷新奖励（防止无脑 spam）
                    if self._sxx2_had_follow:
                        inc += 0.05
                    # 重置“首次跟进步”，开启下一轮延迟计量
                    self._sxx2_first_follow_step = None
                    self._sxx2_enemy_hp_prev = emy_hp
            else:
                # 2 未命中：轻罚
                inc += MISS

        # —— 窗口内计分（按 step）——
        if self._sxx2_active and now_step <= self._sxx2_end_step:
            # 1/3 命中 → 离散奖励（带延迟衰减）
            if hit1 or hit3:
                self._sxx2_had_follow = True
                if self._sxx2_first_follow_step is None:
                    self._sxx2_first_follow_step = now_step
                latency = max(0, now_step - self._sxx2_open_step)
                decay = math.exp(- latency / float(TAU))
                if hit1 and (now_step - self._sxx2_open_step) <= S1_MAX:
                    inc += B1 * decay
                if hit3 and (now_step - self._sxx2_open_step) <= S3_MAX:
                    inc += B3 * decay

            # 伤害持续奖励：按 HP% 下降累计（按帧采样，避免漏记）
            if self._sxx2_enemy_hp_prev is not None and emy_hp_max > 0:
                delta_hp = max(0, self._sxx2_enemy_hp_prev - emy_hp)
                if delta_hp > 0:
                    dmg_pct = delta_hp / float(emy_hp_max)
                    inc += SCALE * dmg_pct
                    self._sxx2_dmg_acc += dmg_pct
                    self._sxx2_had_follow = True
            self._sxx2_enemy_hp_prev = emy_hp

            # 记录三技能的 step，用于收割窗口
            if used3 or hit3:
                self._sxx2_last_3_step = now_step

            # 收割加成：3 后 FIN_WIN 内发生击杀 → 加分一次
            my_id    = (hero.get("actor_state", {}) or {}).get("runtime_id")
            enemy_id = (enemy or {}).get("actor_state", {}).get("runtime_id") if enemy else None
            last3 = self._sxx2_last_3_step
            if (now_step - last3) >= 0 and (now_step - last3) <= FIN_WIN:
                for da in (frame_data.get("frame_action", {}) or {}).get("dead_action", []) or []:
                    if da.get("killer", {}).get("runtime_id") == my_id and da.get("death", {}).get("runtime_id") == enemy_id:
                        inc += KBON
                        self._sxx2_last_3_step = -10**9
                        break

            # 窗口结束：若无任何有效跟进（几乎无伤害且无命中），小罚
            if now_step >= self._sxx2_end_step:
                if (self._sxx2_dmg_acc < EPS) and (not self._sxx2_had_follow):
                    inc += NOPEN
                # 关闭窗口
                self._sxx2_active = False
                self._sxx2_enemy_hp_prev = None

        return inc


    def _update_combat_markers(self, now_step, main_hero, enemy_hero, used_delta_me, hit_delta_me,
                            used_delta_foe=None, hit_delta_foe=None):
        """不用 behav_mode 的交战迹象更新：技能事件 + HP 下降 + （可选）近身。每个 step 调一次。"""
        if self._last_step_seen == now_step:
            return
        self._last_step_seen = now_step

        # 1) 技能事件：任一方的 used/hit（只看 1/2/3 槽，排除回城/召唤师等）
        if any(used_delta_me[1:4]) or any(hit_delta_me[1:4]) \
        or (used_delta_foe and any(used_delta_foe[1:4])) \
        or (hit_delta_foe and any(hit_delta_foe[1:4])):
            self._last_attack_or_cast_step = now_step

        # 2) 伤害事件：任一方 HP 下降
        my_hp  = (main_hero.get("actor_state", {}) or {}).get("hp", 0)
        em_hp  = (enemy_hero or {}).get("actor_state", {}).get("hp", 0)
        if self._prev_my_hp is None:  self._prev_my_hp = my_hp
        if self._prev_emy_hp is None: self._prev_emy_hp = em_hp
        if my_hp < self._prev_my_hp or em_hp < self._prev_emy_hp:
            self._last_damage_step = now_step
        self._prev_my_hp, self._prev_emy_hp = my_hp, em_hp

        # 3) （可选）近身
        prox = getattr(GameConfig, "SXX_COMBAT_PROX_DIST", None)
        if prox is not None and main_hero and enemy_hero:
            ax = main_hero["actor_state"]["location"]["x"]; az = main_hero["actor_state"]["location"]["z"]
            bx = enemy_hero["actor_state"]["location"]["x"]; bz = enemy_hero["actor_state"]["location"]["z"]
            if math.dist((ax, az), (bx, bz)) <= prox:
                self._last_close_step = now_step

    def _is_in_combat(self, now_step):
        W  = getattr(GameConfig, "SXX_COMBAT_RECENT_STEPS", 2)
        Wc = getattr(GameConfig, "SXX_COMBAT_PROX_STEPS", 2)
        in_by_events = (now_step - self._last_attack_or_cast_step) <= W or (now_step - self._last_damage_step) <= W
        in_by_prox   = (now_step - self._last_close_step) <= Wc if getattr(GameConfig, "SXX_COMBAT_PROX_DIST", None) is not None else False
        return in_by_events or in_by_prox


    def sxx_s1_enh_aa_reward(self, frame_no, hero, enemy, used_delta, hit_delta, frame_data):
        """
        让 AI 学会：S1 后持有强化 → 等 S1 冷却将好时打出强化普攻 → 迅速再滚；
        同时避免对抗态下“硬卡不打”。
        不用 behav_mode，靠 HP 下降 + “本帧无技能命中” 来推断普攻。
        """
        IDX_1 = 1  # 一技能槽位
        now_step = self._step_no(frame_no)

        # === 读参数 ===
        CD_READY = getattr(GameConfig, "SXX1_CD_READY_RATIO", 0.20)
        GRACE    = getattr(GameConfig, "SXX1_RELEASE_GRACE_STEPS", 2)
        HOLD_MIN = getattr(GameConfig, "SXX1_HOLD_MIN_STEPS", 3)

        COMBAT_MAX_HOLD = getattr(GameConfig, "SXX1_COMBAT_MAX_HOLD_STEPS", 2)
        BASE    = getattr(GameConfig, "SXX1_RELEASE_BASE", 0.25)
        IN_AB   = getattr(GameConfig, "SXX1_IN_AB_MULT", 1.3)
        TAU     = getattr(GameConfig, "SXX1_LATENCY_TAU", 2.5)

        CHAIN_STEPS = getattr(GameConfig, "SXX1_CHAIN_S1_STEPS", 3)
        CHAIN_BONUS = getattr(GameConfig, "SXX1_CHAIN_BONUS", 0.20)

        SKIP_PEN   = getattr(GameConfig, "SXX1_SKIP_PENALTY", -0.04)
        HOLD_PEN   = getattr(GameConfig, "SXX1_COMBAT_HOLD_PEN", -0.01)
        LATE_PEN   = getattr(GameConfig, "SXX1_LATE_RELEASE_PEN", -0.01)



        inc = 0.0

        # === S1 冷却比例 ===
        slots = (hero.get("skill_state", {}) or {}).get("slot_states", []) or []
        s1 = slots[IDX_1] if len(slots) > IDX_1 else {}
        cd = float(s1.get("cooldown", 0) or 0.0)
        mx = float(s1.get("cooldown_max", 0) or 0.0)
        cd_ratio = (cd / mx) if mx > 0 else 1.0

        # === 更新“对抗态”标记（不用 behavemode） ===
        # 敌方 used/hit 不需要就传 None；本工程靠 HP 下降已经足够
        self._update_combat_markers(now_step, hero, enemy, used_delta, hit_delta, None, None)
        in_combat = self._is_in_combat(now_step)

        # === S1 使用：获得强化；若已持有又放 S1 → 小罚（浪费强化） ===
        if used_delta[IDX_1] > 0:
            if self._s1_enh_ready:
                inc += SKIP_PEN
            self._s1_enh_ready = True
            self._s1_open_step = now_step
            self._s1_ready_step = None
            # 清掉上一次“期待再滚”的窗口
            self._s1_chain_expect_end = -10**9

        # === 仅在持有强化时，进行“卡→释放”的判定 ===
        if self._s1_enh_ready:
            # 进入“应当释放期”：首次 cd_ratio ≤ 阈值，且至少持有了 HOLD_MIN 步
            if self._s1_ready_step is None and cd_ratio <= CD_READY and (now_step - self._s1_open_step) >= HOLD_MIN:
                self._s1_ready_step = now_step

            # —— 用 HP 下降 + “无技能命中(1/2/3)” 来判定本帧是否打出普攻 —— 
            # —— 用被动槽位(0)的冷却跳变来识别“刚打出强化普攻” ——
            slots = (hero.get("skill_state", {}) or {}).get("slot_states", []) or []
            s0 = slots[0] if len(slots) > 0 else {}
            cd0 = float(s0.get("cooldown", 0) or 0.0)
            mx0 = float(s0.get("cooldown_max", 0) or 0.0)

            is_attack_now = False
            if not self._s1_passive_seen:
                # 第一帧只做采样，不判定
                self._s1_passive_seen = True
            else:
                # 冷却从 0 突然拉高（比例>0.8 避免抖动）→ 强化普攻刚打出
                ratio = (cd0 / mx0) if mx0 > 0 else 0.0
                is_attack_now = (mx0 > 0) and (cd0 > self._s1_prev_passive_cd + 1e-6) and (ratio > 0.8)


            if is_attack_now:
                # —— 视为“打出强化普攻”，计算奖励 —— 
                if self._s1_ready_step is not None:
                    latency = max(0, now_step - self._s1_ready_step)
                    decay = math.exp(- latency / float(TAU))
                    inc += BASE * decay
                else:
                    # 还没到“应当释放期”就打：给较低奖励（或置 0）
                    inc += BASE * 0.4

                # 若在 S2 破甲窗口内 → 乘子
                if getattr(self, "_sxx2_active", False) and now_step <= getattr(self, "_sxx2_end_step", -10**9):
                    inc *= IN_AB

                # 开一个“成功释放后期待再滚”的短窗口
                self._s1_chain_expect_end = now_step + CHAIN_STEPS

                # 强化已消耗
                self._s1_enh_ready = False
                self._s1_ready_step = None

            else:
                # 没打：对抗态下超过“硬卡上限”则每步轻罚
                if in_combat and (now_step - self._s1_open_step) > COMBAT_MAX_HOLD:
                    inc += HOLD_PEN
                # 进入“应当释放期”且超过宽限还不打 → 每步轻罚
                if self._s1_ready_step is not None and (now_step - self._s1_ready_step) > GRACE:
                    inc += LATE_PEN

            self._s1_prev_passive_cd = cd0


        # === 成功释放后的“再滚”确认（卡强普→再滚） ===
        if self._s1_chain_expect_end >= 0:
            if used_delta[IDX_1] > 0 and now_step <= self._s1_chain_expect_end:
                inc += CHAIN_BONUS
                self._s1_chain_expect_end = -10**9
            elif now_step > self._s1_chain_expect_end:
                self._s1_chain_expect_end = -10**9

        return inc
