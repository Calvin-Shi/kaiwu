#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


def handle_disaster_recovery(env_obs, logger):
    # Handle disaster recovery logic
    # 处理容灾逻辑
    if env_obs is None:
        logger.error("env_obs is None, please check")
        return True

    extra_info = env_obs["extra_info"]
    result_code, result_message = extra_info["result_code"], extra_info["result_message"]

    if result_code < 0:
        logger.error(f"env run failed, result_code: {result_code}, result_msg: {result_message}")
        raise RuntimeError(result_message)
    elif result_code > 0:
        return True

    return False
