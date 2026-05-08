#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import mmap
import random
import numpy as np
from kaiwudrl.common.replay_buffer.replay_buffer_base import ReplayBufferBase
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class FileMMAPReplayBuffer(ReplayBufferBase):
    def __init__(self, file_name, logger):
        self.file_name = file_name
        self.logger = logger

    def read_from_mmap_file(self):
        if not self.file_name or not os.path.exists(self.file_name):
            return []

        with open(self.file_name, "rb") as f:
            file_size = os.path.getsize(self.file_name)
            total_datas = file_size // CONFIG.SAMPLE_DIM

            # 使用 mmap 映射整个文件
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # 随机抽取索引
            sampled_indices = random.sample(range(total_datas), CONFIG.train_batch_size)

            sampled_datas = []
            for idx in sampled_indices:
                offset = idx * data_size
                data = mm[offset : offset + data_size]

                sampled_datas.append(data)

            mm.close()

            return sampled_datas

    def total_size(self):
        if not self.file_name or not os.path.exists(self.file_name):
            return 0

        file_size = os.path.getsize(self.file_name)
        total_data_size = file_size // CONFIG.SAMPLE_DIM

        return total_data_size
