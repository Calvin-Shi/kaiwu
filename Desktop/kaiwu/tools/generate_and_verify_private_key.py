#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.config.config_control import CONFIG
from kaiwudrl.common.checkpoint.model_file_common import (
    model_file_signature_verify_by_file,
)
from kaiwudrl.common.utils.common_func import generate_private_key


# 解析配置文件
CONFIG.set_configure_file("kaiwudrl/conf/kaiwudrl/learner.toml")
CONFIG.parse_learner_configure()


def generate_private_and_public_key(target_dir="./"):
    """
    生成私钥和公钥
    """
    return generate_private_key(target_dir)


def model_file_public_signature_verify(zip_file_path):
    """
    验证数字签名正确性
    """
    if not zip_file_path:
        print(f"zip_file_path is empty, please check")
        return False

    success = model_file_signature_verify_by_file(zip_file_path)
    if success:
        print(f"model_file_signature_verify_by_file success")
        return True
    else:
        print(f"model_file_signature_verify_by_file failed")
        return False
