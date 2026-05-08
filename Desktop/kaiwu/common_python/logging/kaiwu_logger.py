#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


"""
日志记录类, git代码: https://github.com/Delgan/loguru
框架提供self.logger, 单个进程/线程下面同一个对象, 业务不需要自己定义日志对象

优点:
开箱即用，无需准备
无需初始化，导入函数即可使用
更容易的文件日志记录与转存/保留/压缩方式
更优雅的字符串格式化输出
可以在线程或主线程中捕获异常
可以设置不同级别的日志记录样式
支持异步，且线程和多进程安全
支持惰性计算
适用于脚本和库
完全兼容标准日志记录
更好的日期时间处理

实例, 打印日志地方需要按照需要设置日志级别:
self.logger = getLogger()
self.logger.set_logger_format(f"/actor/actor_server_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'actor_server')
self.logger.info('actor_server process is pid is {}', os.getpid())

系统会对进程aisrv、actor、learner的日志内容增加进程名字样, 其他的日志内容不会增加
"""

import sys
import json
import time
import os
import traceback
from loguru import logger
from functools import wraps
from common_python.utils.singleton import Singleton

g_not_server_label = "not_server"


# 限流装饰器
def rate_limiter(max_calls_attr="max_calls_log_per_min", period=60):
    def decorator(func):
        last_reset = time.time()
        call_count = 0

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            nonlocal last_reset, call_count
            # 动态从实例获取
            max_calls = self.params.get(max_calls_attr, 60)

            current_time = time.time()
            elapsed_time = current_time - last_reset

            if elapsed_time >= period:
                call_count = 0
                last_reset = current_time

            if call_count < max_calls:
                call_count += 1
                return func(self, *args, **kwargs)
            else:
                pass  # 可选：记录限流信息

        return wrapper

    return decorator


@Singleton
class KaiwuLogger(object):
    def __init__(self, svr_name="learner") -> None:

        """
        按照进程ID来做日志过滤, 默认是全部通过
        再新增不通过的, 这样减少了函数调用
        """
        self.not_allowed_pid = []

        # 返回的路径深度, 注意增加了rate_limiter装饰器, 故层数 + 1
        self.depth = 2

        self.svr_name = svr_name

        # 默认参数, 如果用户不传入则按照默认的设置
        self.default_params = {
            "compression": "zip",
            "encoding": "utf-8",
            "rotation": "100MB",
            "level": "INFO",
            "serialize": False,
            "retention": "10 days",
            "max_single_message_len": 1024,
            "max_calls_log_per_min": 60,
        }

        self.params = {}

        # 记录当前进程是否已经初始化过logger
        self.current_process_initialized = False
        self.current_process_pid = None

    """
    使用场景是部分workflow进程的ID是需要落地日志的, 部分的不需要
    """

    def add_not_allowed_pid(self, new_pid):
        self.not_allowed_pid.append(new_pid)

    # 自定义日志文件输出的json格式, 以后需要则继续修改
    def json_formatter(self, record):
        log_record = {
            "time": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f") if "time" in record else "",
            "level": str(record["level"].name) if "level" in record else "",
            "message": str(record["message"]) if "message" in record else "",
            "file": str(record["file"].name) if "file" in record else "",
            "line": str(record["line"]) if "line" in record else "",
            "module": self.svr_name,
            "process": str(record["module"]) if "module" in record else "",
            "function": str(record["function"]) if "function" in record else "",
            "stack": "",
            "pid": os.getpid(),
        }

        # 异常单独处理
        if "exception" in record and record["exception"]:
            # 获取异常的类型、值和堆栈跟踪
            exception_type = record["exception"].type
            exception_value = record["exception"].value
            exception_traceback = record["exception"].traceback

            # 将堆栈跟踪转换为字符串列表
            traceback_str = traceback.format_exception(exception_type, exception_value, exception_traceback)
            log_record["stack"] = "".join(traceback_str)

        record["extra"]["json"] = json.dumps(log_record, ensure_ascii=False)

        # 增加每次json格式的换行的操作
        return "{extra[json]}" + "\n"

    """
    调用设置日志各种参数
    1. file_name是必须的, 即日志生成的配置文件
    2. filter_content, 如果是单进程使用, 则无需设置; 如果是单个进程里需要过滤则需要设置
    3. params, 日志配置的相关参数
    """

    def set_logger_format(self, file_name, filter_content=None, params=None):

        filter_func = None
        if filter_content:
            # 开发测试阶段, 可以采用 filter=lambda x: print(x, filter_content) or filter_content in x['message']打印日志
            filter_func = lambda x: filter_content in x["message"]

        if params is not None:
            self.params = {**self.default_params, **params}
        else:
            self.params = self.default_params

        # 获取当前进程PID
        current_pid = os.getpid()

        # 如果是新的进程（fork后的子进程），或者是第一次初始化
        # 需要移除所有继承来的handler，避免日志写入到父进程的文件
        if self.current_process_pid != current_pid:
            logger.remove(handler_id=None)
            self.current_process_pid = current_pid
            self.current_process_initialized = True

        log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | PID:{process} | {module}.{function}:{line} - {message}"
        logger.add(sys.stdout, level=self.params.get("level"), format=log_format)

        logger.add(
            file_name,
            rotation=self.params.get("rotation"),
            encoding=self.params.get("encoding"),
            enqueue=False,
            compression=self.params.get("compression"),
            retention=self.params.get("retention"),
            level=self.params.get("level"),
            filter=filter_func,
            serialize=self.params.get("serialize"),
            format=self.json_formatter,
        )

        self.logger = logger

    """
    根据填写的字符串, 增加进程名字内容, 便于进行filter操作
    1. 如果是需要包含进程名的日志, 则前面添加进程名, 主要针对aisrv、actor、learner进程, 形如learner msg
    2. 如果是不需要包含进程名的日志, 则前面不需要添加进程名, 主要针对aisrv、actor、learner进程派生的进程, 例如learner model_file_sync
    """

    def make_msg_content(self, msg, not_server=True, msg_length_check=False):
        if not not_server:
            msg = f"{self.svr_name} {msg}"

        # 需要进行字符串长度检测则开始检测
        if msg_length_check:
            max_length = self.params.get("max_single_message_len")
            if len(msg) > max_length:
                return msg[:max_length] + "...message is truncated, print as needed"

        return msg

    # 禁止修改kaiwu_logger的部分函数, 规避绕过框架的限制
    def __setattr__(self, name, value):
        if name in ("debug", "info", "warning", "error", "critical", "exception", "log"):
            print(f"Modifying '{name}' method is prohibited.")
        super().__setattr__(name, value)

    def is_not_server(self, *args):
        return g_not_server_label in args

    @rate_limiter(max_calls_attr="max_calls_log_per_min", period=60)
    def debug(self, msg, *args, **kwargs):
        if os.getpid() in self.not_allowed_pid:
            return

        return self.logger.opt(depth=self.depth).debug(
            self.make_msg_content(msg, self.is_not_server(*args), True), *args, **kwargs
        )

    @rate_limiter(max_calls_attr="max_calls_log_per_min", period=60)
    def info(self, msg, *args, **kwargs):
        if os.getpid() in self.not_allowed_pid:
            return

        return self.logger.opt(depth=self.depth).info(
            self.make_msg_content(msg, self.is_not_server(*args), True), *args, **kwargs
        )

    @rate_limiter(max_calls_attr="max_calls_log_per_min", period=60)
    def warning(self, msg, *args, **kwargs):
        return self.logger.opt(depth=self.depth).warning(
            self.make_msg_content(msg, self.is_not_server(*args), True), *args, **kwargs
        )

    @rate_limiter(max_calls_attr="max_calls_log_per_min", period=60)
    def error(self, msg, *args, **kwargs):
        return self.logger.opt(depth=self.depth).error(
            self.make_msg_content(msg, self.is_not_server(*args), True), *args, **kwargs
        )

    @rate_limiter(max_calls_attr="max_calls_log_per_min", period=60)
    def critical(self, msg, *args, **kwargs):
        return self.logger.opt(depth=self.depth).critical(
            self.make_msg_content(msg, self.is_not_server(*args), True), *args, **kwargs
        )

    @rate_limiter(max_calls_attr="max_calls_log_per_min", period=60)
    def exception(self, msg, *args, **kwargs):
        return self.logger.opt(depth=self.depth).exception(
            self.make_msg_content(msg, self.is_not_server(*args), True), *args, **kwargs
        )

    @rate_limiter(max_calls_attr="max_calls_log_per_min", period=60)
    def log(self, level, msg, *args, **kwargs):
        return self.logger.opt(depth=self.depth).log(
            level,
            self.make_msg_content(msg, self.is_not_server(*args), True),
            *args,
            **kwargs,
        )
