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


def get_valid_model_pool(logger):
    file_path = "kaiwu.json"
    if not os.path.exists(file_path):
        raise Exception(f"File {file_path} does not exist.")

    with open(file_path, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            raise Exception("Error decoding JSON.")

    if "model_pool" not in data:
        raise Exception("'model_pool' key does not exist in the JSON data.")

    model_pool_length = len(data["model_pool"])
    if model_pool_length > 3:
        raise Exception("The length of 'model_pool' > 3, please check")

    logger.info("The length of 'model_pool' is acceptable.")
    return data["model_pool"]
