#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import json
import numpy as np
from enum import Enum
import math


class Action(Enum):
    WHICH_BUTTON = 0
    MOVE = 1
    # MOVE_Z = 2
    # OFFSET_X = 3
    # OFFSET_Z = 4
    TARGET = 5


class Button(Enum):
    NONE_ACTION = 0
    NONE_ACTION_2 = 1
    MOVE = 2
    NORMAL_ATTACK = 3
    SKILL_1 = 4
    SKILL_2 = 5
    SKILL_3 = 6
    SKILL_4 = 7
    CHOSEN_SKILL = 8
    RECALLING_TO_THE_BASE = 9
    EQUIPMENT_SKILL = 10
    HEAL_SKILL = 11
    FRIEND_SKILL = 12


_dir = [None] + [(180 - 15 * i) % 360 for i in range(24)]


class Direction(Enum):
    DIR_0 = 0
    DIR_1 = 1
    DIR_2 = 2
    DIR_3 = 3
    DIR_4 = 4
    DIR_5 = 5
    DIR_6 = 6
    DIR_7 = 7
    DIR_8 = 8
    DIR_9 = 9
    DIR_10 = 10
    DIR_11 = 11
    DIR_12 = 12
    DIR_13 = 13
    DIR_14 = 14
    DIR_15 = 15
    DIR_16 = 16
    DIR_17 = 17
    DIR_18 = 18
    DIR_19 = 19
    DIR_20 = 20
    DIR_21 = 21
    DIR_22 = 22
    DIR_23 = 23
    DIR_24 = 24

    def to_dir(self):
        return _dir[self.value]


class TargetType(Enum):
    NONE = 0
    ENEMY_HERO = 1
    FRIEND_HERO = 2
    SELF = 3
    MONSTER = 4
    ENEMY_MINIONS = 5
    ENEMY_TURRET = 6
    UNKNOWN = 7


class Target(Enum):
    NONE = 0
    Enemy_Hero_0 = 1
    SELF = 2
    ENEMY_MINION_0 = 3
    ENEMY_MINION_1 = 4
    ENEMY_MINION_2 = 5
    ENEMY_MINION_3 = 6
    ENEMY_TURRET = 7
    River_Sprite = 8

    def get_target_type(self):
        if self.value == 0:
            return TargetType.NONE
        elif self.value == 1:
            return TargetType.ENEMY_HERO
        elif self.value == 2:
            return TargetType.SELF
        elif self.value <= 6:
            return TargetType.ENEMY_MINIONS
        elif self.value == 7:
            return TargetType.ENEMY_TURRET
        elif self.value == 8:
            return TargetType.MONSTER
        else:
            return TargetType.UNKNOWN

    def get_config_id(self, self_hero_config_ids, enemy_hero_config_ids, self_hero_config_id):
        self_hero_config_ids.sort()
        enemy_hero_config_ids.sort()
        self_hero_config_ids += [-1] * (1 - len(self_hero_config_ids))
        enemy_hero_config_ids += [-1] * (1 - len(enemy_hero_config_ids))

        target_to_config_id = {
            Target.NONE: -1,
            Target.Enemy_Hero_0: -1,
            Target.SELF: -1,
            Target.River_Sprite: 6827,
        }
        hero_configs = [-1] + enemy_hero_config_ids + self_hero_config_ids + [self_hero_config_id]

        for target, config_id in enumerate(hero_configs):
            target_to_config_id[Target(target)] = config_id

        return target_to_config_id.get(self, -1)


def convert_numpy_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(element) for element in data]
    else:
        return data


def calc_degree(direction, camp):
    degree = 0
    if direction[0] == 0:
        degree = -90 if direction[1] > 0 else 90
    else:
        degree = int(math.atan2(-1.0 * direction[1], direction[0]) * (180.0 / math.pi))

    if camp == 1:
        return (360 + degree) % 360
    else:
        return (180 + 360 + degree) % 360


def get_closest_angle_idx(degree, _dir):
    return min(range(1, 25), key=lambda k: abs(degree - _dir[k]))


def calculate_angle_prob(prob_2d, _dir, state_dicts):
    angle_prob = np.zeros(25)
    angle_prob[0] = 0

    for i in range(prob_2d.shape[0]):
        for j in range(prob_2d.shape[1]):
            if prob_2d[i, j] > 0:
                direction = [i - 8, j - 8]  # 假设原点在(8, 8)
                degree = calc_degree(direction, state_dicts["player_camp"])
                closest_angle_idx = get_closest_angle_idx(degree, _dir)
                angle_prob[closest_angle_idx] += prob_2d[i, j]

    return angle_prob


def get_top_n_indices(prob_2d, n=10):
    flat_indices = np.argsort(prob_2d, axis=None)[-n:]  # 获取最大的n个索引
    flat_indices = flat_indices[::-1]  # 逆序处理，使得第一个索引对应的是最大概率值
    return np.unravel_index(flat_indices, prob_2d.shape)


class DumpProbs:
    def __init__(self, state_dicts, act_data):
        self.state_dicts = state_dicts
        self.act_data = [act_data]
        # print(f"act_data is {act_data}")

    def save_to_file(self, output_file):
        # 获取预测数据
        data = self.parse_prob()
        # print(f"data is {data}")

        # 转换数据
        converted_data = convert_numpy_types(data)

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "a") as f:
            json.dump(converted_data, f)
            f.write("\n")

    def _parse_button(self, button, hero_idx):
        button = Button(button)
        return {"name": button.name, "value": button.value}

    def _parse_move(self, move, hero_idx):
        move = Direction(move)
        return {"name": move.name, "value": move.value, "direction": move.to_dir()}

    def _get_hero_config_ids(self, hero_idx):
        """
        获取两边阵营各个英雄的config_id并排序
        在1v1的实现中, 两边阵营各1个英雄
        """
        self_hero_config_id = -1
        self_camp_id = self.state_dicts["player_camp"]
        self_player_id = self.state_dicts["player_id"]

        self_hero_config_ids = []
        enemy_hero_config_ids = []

        self_hero, enemy_hero = None, None
        hero_list = self.state_dicts["frame_state"]["hero_states"]
        for hero in hero_list:
            if hero["player_id"] == self_player_id:
                main_camp = hero["actor_state"]["camp"]
                self_hero = hero
                self_hero_config_id = hero["actor_state"]["config_id"]
                self_hero_config_ids.append(self_hero_config_id)
            else:
                enemy_camp = hero["actor_state"]["camp"]
                enemy_hero = hero
                enemy_hero_config_ids.append(hero["actor_state"]["config_id"])

        self_hero_config_ids.sort()
        enemy_hero_config_ids.sort()
        return self_hero_config_ids, enemy_hero_config_ids, self_hero_config_id

    def _parse_target(self, target, hero_idx):
        target = Target(target)

        return {
            "name": target.name,
            "value": target.value,
            "type": target.get_target_type().name,
            "config_id": target.get_config_id(*self._get_hero_config_ids(hero_idx)),
        }

    def get_action_parse_fn(self, action):
        def _same(x, hero_idx):
            return {"name": "{}_{}".format(action.name, x), "value": x}

        ret = {
            Action.WHICH_BUTTON: self._parse_button,
            Action.MOVE: self._parse_move,
            Action.TARGET: self._parse_target,
        }
        return ret.get(Action(action), _same)

    def _parse_prob(self, values, hero_idx, action_parser):
        # print(f"values is {values}, action_parser is {action_parser}")
        top_3 = sorted([(x, i) for i, x in enumerate(values)], reverse=True)[:3]

        ret_top = [{"prob": prob, **action_parser(i, hero_idx)} for prob, i in top_3]

        return ret_top

    def get_probs_pasre_fn(self, action, action_parser):
        def _parse_prob(values, hero_idx):
            return self._parse_prob(values, hero_idx, action_parser)

        return _parse_prob

    def parse_prob(self):

        runtime_to_config = {
            hero["player_id"]: hero["actor_state"]["config_id"]
            for hero in self.state_dicts["frame_state"]["hero_states"]
        }

        heros = []
        for hero_idx in range(len(self.act_data)):
            config_id = runtime_to_config[self.state_dicts["player_id"]]

            _raw_prob = self.act_data[hero_idx].d_prob[0]
            _final_prob_list = [
                _raw_prob[:12],
                _raw_prob[12 : 12 + 16],
                _raw_prob[12 + 16 : 12 + 2 * 16],
                _raw_prob[12 + 2 * 16 : 12 + 3 * 16],
                _raw_prob[12 + 3 * 16 : 12 + 4 * 16],
                _raw_prob[12 + 4 * 16 :],
            ]
            # print(f"_final_prob_list is {_final_prob_list}")
            # [len(x) for x in final_prob_list] == [13, 25, 42, 42, 39, 1]

            _actions = self.act_data[hero_idx].d_action
            # print(f"_actions is {_actions}")

            # len(sub_actions) == len(_legal_action) == len(_actions) == 5
            # [len(x) for x in _legal_action] == [13, 25, 42, 42, 39]

            actions, sub_actions, legal_action, probs = {}, {}, {}, {}

            # parse actions
            for action_type in Action:
                # 1v1的move采用独立的规则
                if action_type == Action.MOVE:
                    continue

                # _actions => action:
                #    (2, 7, 11, 31, 0) =>
                #    {
                #        "WHICH_BUTTON": {"name": "MOVE", "value": 2},
                #        "MOVE": {"name": "DIR_7", "value": 7, "direction": 75},
                #        "OFFSET_X": 11,
                #        "OFFSET_Z": 31,
                #        "TARGET": {"name": "NONE", "value": 0, "type": "None", "config_id": -1},
                #    }
                action_value = _actions[action_type.value]
                action_value_parser = self.get_action_parse_fn(action_type)
                actions[action_type.name] = action_value_parser(action_value, hero_idx)

                # Note:当前在目标预测上存在一定的问题，这里增加清洗规则
                if action_type == Action.TARGET:
                    real_target_index = actions["TARGET"]["value"]

                    # 如果动作为none或move，不应该选择野怪为目标
                    if _actions[0] < 3:
                        _final_prob_list[action_type.value][8] = 0.0

                    # 如果动作为atk，且预测目标中实际目标的概率为0，说明采样动作与实际动作不匹配，清洗为实际选择目标
                    elif _actions[0] == 3 and _final_prob_list[action_type.value][real_target_index] < 0.0001:
                        for target_index in range(len(_final_prob_list[action_type.value])):
                            if target_index != real_target_index:
                                _final_prob_list[action_type.value][target_index] = 0.0
                            else:
                                _final_prob_list[action_type.value][target_index] = 1.0

                # _final_prob_list
                probs_parser = self.get_probs_pasre_fn(action_type, action_value_parser)
                probs[action_type.name] = probs_parser(_final_prob_list[action_type.value], hero_idx)

            # 计算移动方向
            move_offset = [_actions[1] - 8, _actions[2] - 8]
            move_dir = calc_degree(move_offset, self.state_dicts["player_camp"])
            closest_angle_idx = get_closest_angle_idx(move_dir, _dir)
            direction_enum = Direction(closest_angle_idx)
            actions[direction_enum.name] = {
                "name": direction_enum.name,
                "value": direction_enum.value,
                "direction": direction_enum.to_dir(),
            }

            # 提取x轴和z轴的概率分布
            x_prob = _final_prob_list[1]
            z_prob = _final_prob_list[2]

            # 计算二维概率分布
            prob_2d = np.outer(x_prob, z_prob)

            # 初始化24个角度的概率
            angle_prob = calculate_angle_prob(prob_2d, _dir, self.state_dicts)

            # 获取概率最大的n个点
            max_indices_num = 10
            max_indices = get_top_n_indices(prob_2d, max_indices_num)
            max_values = prob_2d[max_indices]

            # 打印结果
            probs["MOVE"] = []
            angle_index_list = []

            # print("---------------------")
            for i in range(max_indices_num):
                degree = calc_degree([max_indices[0][i] - 8, max_indices[1][i] - 8], self.state_dicts["player_camp"])
                closest_angle_idx = get_closest_angle_idx(degree, _dir)
                # print(f"Max Value {i+1}: {max_values[i]} at index {max_indices[0][i], max_indices[1][i]} degree is {degree}, closest_angle_idx is {closest_angle_idx}")
                if closest_angle_idx not in angle_index_list:
                    angle_index_list.append(closest_angle_idx)
                if len(angle_index_list) >= 2:
                    break

            for angle_index in angle_index_list:
                direction_enum = Direction(angle_index)
                probs["MOVE"].append(
                    {
                        "prob": angle_prob[angle_index],
                        "name": direction_enum.name,
                        "value": direction_enum.value,
                        "direction": direction_enum.to_dir(),
                    }
                )

            # 按 prob 键的值进行降序排序
            probs["MOVE"].sort(key=lambda x: x["prob"], reverse=True)

            # action_type = Action.MOVE
            # action_value_parser = self.get_action_parse_fn(action_type)

            # # _final_prob_list
            # probs_parser = self.get_probs_pasre_fn(action_type, action_value_parser)
            # probs[action_type.name] = probs_parser(angle_prob, hero_idx)

            # move_offset = [_actions[1] - 8, _actions[2] - 8]
            # move_dir = calc_degree(move_offset, self.state_dicts['player_camp'])
            # closest_angle_idx = min(range(1, 25), key=lambda k: abs(move_dir - _dir[k]))
            # direction_enum = Direction(closest_angle_idx)
            # probs["MOVE"].append({"prob": 1., "name": direction_enum.name, "value": direction_enum.value, "direction": direction_enum.to_dir()})

            # # 示例输出
            # print("x_prob:", x_prob)
            # print("z_prob:", z_prob)
            # print("prob_2d:", prob_2d)
            # print("angle_prob:", angle_prob)

            hero = {
                "config_id": config_id,
                "actions": actions,
                "probs": probs,
                # "probs": {
                #     "which_button": final_prob_list[0],
                #     "move": final_prob_list[1],
                #     "offset_x": final_prob_list[2],
                #     "offset_z": final_prob_list[3],
                #     "target": final_prob_list[4],
                # },
            }
            heros.append(hero)

        data = {
            "sgame_id": self.state_dicts["game_id"],
            "frame_no": self.state_dicts["frame_state"]["frameNo"],
            "camp_id": self.state_dicts["player_camp"],
            "heros": heros,
        }
        return data


if __name__ == "__main__":
    from action_space_test import (
        req_pb_test_data,
        features_test_data,
        process_result_test_data,
    )

    dump_probs = DumpProbs(req_pb_test_data, features_test_data, process_result_test_data)
    dump_probs.save_to_file("probs.json")
