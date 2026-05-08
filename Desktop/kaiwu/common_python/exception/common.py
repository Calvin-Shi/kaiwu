#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import functools
import traceback


def exception_handler(*expected_exceptions, logger=None, extra_message=None):
    """
    通用异常捕获装饰器, 支持传入额外信息

    :param expected_exceptions: 预期捕获的异常类型，不指定则捕获所有异常
    :param logger: 外部传入的日志记录器, 为None则使用logging默认单例
    :param extra_message: 发生异常时需要记录的额外信息

    使用示例
    from common_python import exception

    @exception.exception_handler(ZeroDivisionError, extra_message="除法运算中除数不能为零")
    def divide(a, b):
        return a / b

    # 使用自定义logger。捕获所有异常
    @exception.exception_handler(logger=custom_logger,extra_message="整数转换失败")
    def parse_int(s):
        return int(s)
    """
    # 确定要使用的logger，优先使用传入的，否则使用logging的默认单例
    used_logger = logger

    def decorator(func):
        @functools.wraps(func)  # 保留原函数的元信息
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except expected_exceptions or Exception as e:
                # 构建完整的错误消息
                error_msg = f"function {func.__name__} execution failed: {str(e)} traceback.print_exc() is {traceback.format_exc()}"
                # 如果有额外信息则添加
                if extra_message:
                    error_msg += f" | extra information: {extra_message}"
                used_logger.error(error_msg)

        return wrapper

    return decorator
