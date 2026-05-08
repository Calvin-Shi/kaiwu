#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


"""
针对不同的文件加载不同的框架, 采用的方法:
1. 第一次解析时必须要在配置项加载后开始调用, 否则因为CONFIG.use_which_deep_learning_framework配置不正确会出错
2. 根据具体的配置加载不同的框架
"""
if (
    KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
    or KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
):
    from kaiwudrl.common.utils.tf_utils import *
elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
    from kaiwudrl.common.utils.torch_utils import *
else:
    raise ValueError("Unsupported deep learning framework specified.")
