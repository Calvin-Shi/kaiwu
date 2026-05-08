#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_env.conf import yaml_hok1v1_game
import random
import itertools


# Loop through camps, shuffling camps before each major loop
# 循环返回camps, 每次大循环前对camps进行shuffle
def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


# Specify single-side multi-agent lineups, looping through all pairwise combinations
# 指定单边多智能体阵容，两两组合循环
def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if not camp_heroes:
        raise Exception(f"camp_heroes is empty")

    try:
        for camp in camp_heroes:
            hero_id = camp[0]["hero_id"]
            if yaml_hok1v1_game.restrictions_on_hero_ids and hero_id not in yaml_hok1v1_game.available_hero_ids:
                raise Exception(f"hero_id {hero_id} not valid")
    except Exception as e:
        raise Exception(f"check hero valid, exception is {str(e)}")

    camps = []
    for lineups in itertools.product(camp_heroes, camp_heroes):
        camp = []
        for lineup in lineups:
            camp.append(lineup)
        camps.append(camp)
    return _lineup_iterator_shuffle_cycle(camps)
