#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from functools import wraps
import time
from typing import Any
from collections.abc import Callable
import logging
import os

LOGGING_LEVEL = os.environ.get("RL_LOGGING_LEVEL", "INFO")
logger = logging.getLogger("kaiwudrl")
logger.setLevel(getattr(logging, LOGGING_LEVEL))
logger.propagate = False

# 全局开关，控制是否启用timeit功能
# 可以通过环境变量ENABLE_TIMEIT控制，默认启用
_TIMEIT_ENABLED = os.environ.get("ENABLE_TIMEIT", "true").lower() in ("true", "1", "yes")


def set_timeit_enabled(enabled: bool):
    """设置timeit功能的全局开关"""
    global _TIMEIT_ENABLED
    _TIMEIT_ENABLED = enabled


def is_timeit_enabled() -> bool:
    """检查timeit功能是否启用"""
    return _TIMEIT_ENABLED


class timeit:
    """A dirty but easy to use decorator for profiling code.

    Args:
        name (str): The name of the timer.
        logger (logging.Logger, optional): Custom logger instance. If None, uses default logger.
            自定义日志句柄。如果为None，使用默认日志句柄。

    Examples:
        >>> from torchrl import timeit
        >>> @timeit("my_function")
        >>> def my_function():
            ...
        >>> my_function()
        >>> with timeit("my_other_function"):
        ...     my_other_function()
        >>> timeit.print()  # prints the state of the timer for each function
        >>> # Use custom logger
        >>> # 使用自定义日志句柄
        >>> custom_logger = logging.getLogger("my_logger")
        >>> @timeit("my_function", logger=custom_logger)
        >>> def my_function():
            ...
        >>> timeit.print(logger=custom_logger)  # prints using custom logger
    """

    _REG = {}
    _default_logger = logger  # Store default logger as class variable
    # 将默认日志句柄存储为类变量

    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger  # Store instance-specific logger
        # 存储实例特定的日志句柄

    def __call__(self, fn: Callable) -> Callable:
        @wraps(fn)
        def decorated_fn(*args, **kwargs):
            if not _TIMEIT_ENABLED:
                return fn(*args, **kwargs)
            with self:
                out = fn(*args, **kwargs)
                return out

        return decorated_fn

    def __enter__(self) -> None:
        if _TIMEIT_ENABLED:
            self.t0 = time.time()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not _TIMEIT_ENABLED:
            return
        t = time.time() - self.t0
        val = self._REG.setdefault(self.name, [0.0, 0.0, 0])

        count = val[2]
        N = count + 1
        val[0] = val[0] * (count / N) + t / N
        val[1] += t
        val[2] = N

    @staticmethod
    def print(prefix: str | None = None, logger=None) -> str:  # noqa: T202
        """Prints the state of the timer.

        Args:
            prefix (str): The prefix to add to the keys. If `None`, no prefix is added.
            logger (logging.Logger, optional): Custom logger instance. If None, uses default logger.
                自定义日志句柄。如果为None，使用默认日志句柄。

        Returns:
            the string printed using the logger.
        """
        if not _TIMEIT_ENABLED:
            return ""

        # Use provided logger or fall back to default
        # 使用提供的日志句柄，否则使用默认日志句柄
        log = logger if logger is not None else timeit._default_logger

        keys = list(timeit._REG)
        keys.sort()
        string = []
        for name in keys:
            strings = []
            if prefix:
                strings.append(prefix)
            strings.append(
                f"{name} took {timeit._REG[name][0] * 1000:4.4f} msec (total = {timeit._REG[name][1]: 4.4f} sec since last reset)."
            )
            string.append(" -- ".join(strings))
            log.info(string[-1])
        return "\n".join(string)

    _printevery_count = 0

    @classmethod
    def printevery(
        cls,
        num_prints: int,
        total_count: int,
        *,
        prefix: str | None = None,
        erase: bool = False,
        logger=None,
    ) -> None:
        """Prints the state of the timer at regular intervals.

        Args:
            num_prints (int): The number of times to print the state of the timer, given the total_count.
            total_count (int): The total number of times to print the state of the timer.
            prefix (str): The prefix to add to the keys. If `None`, no prefix is added.
            erase (bool): If True, erase the timer after printing. Default is `False`.
            logger (logging.Logger, optional): Custom logger instance. If None, uses default logger.
                自定义日志句柄。如果为None，使用默认日志句柄。

        """
        interval = max(1, total_count // num_prints)
        if cls._printevery_count % interval == 0:
            cls.print(prefix=prefix, logger=logger)
            if erase:
                cls.erase()
        cls._printevery_count += 1

    @classmethod
    def todict(cls, percall: bool = True, prefix: str | None = None) -> dict[str, float]:
        """Convert the timer to a dictionary.

        Args:
            percall (bool): If True, return the average time per call.
            prefix (str): The prefix to add to the keys.
        """

        def _make_key(key):
            if prefix:
                return f"{prefix}/{key}"
            return key

        if percall:
            return {_make_key(key): val[0] for key, val in cls._REG.items()}
        return {_make_key(key): val[1] for key, val in cls._REG.items()}

    @staticmethod
    def erase():
        """Erase the timer.

        .. seealso:: :meth:`reset`
        """
        for k in timeit._REG:
            timeit._REG[k] = [0.0, 0.0, 0]

    @classmethod
    def reset(cls):
        """Reset the timer.

        .. seealso:: :meth:`erase`
        """
        cls.erase()
