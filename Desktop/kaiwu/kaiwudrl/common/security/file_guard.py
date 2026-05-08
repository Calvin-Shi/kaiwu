#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
FileOperationGuard - 单容器评估模式下的文件操作防护模块

通过 Monkey Patch 技术拦截 Python 文件操作系统调用, 限制用户代码对受保护目录的访问,
防止用户在评估(eval/exam)模式下篡改评估结果文件。

使用方式:
    from kaiwudrl.common.security.file_guard import get_file_guard
    guard = get_file_guard(logger)
    guard.activate()

设计原则:
    1. 默认关闭, 通过配置项 wrapper_enable_file_guard=true 开启
    2. 仅在 eval/exam 模式下激活
    3. 通过调用栈白名单放行框架自身代码的文件操作
    4. 读操作不受限制, 仅拦截写/删/改操作
"""

import builtins
import ctypes
import ctypes.util
import functools
import importlib
import inspect
import os
import shutil
import subprocess
import threading


# 受保护的目录(默认), 项目侧可通过配置自定义
DEFAULT_PROTECTED_PATHS = [
    "/workspace/battle",
    "/workspace/log",
]

# 框架侧调用者白名单, 这些路径下的代码允许写入受保护目录
FRAMEWORK_CALLER_PATTERNS = [
    "kaiwudrl/",
    "tools/base_env/",
    "tools/eval/",
    "common_python/",
    "/root/tools/",
]

# 禁止 reload 的模块名
_PROTECTED_MODULES = frozenset({"os", "builtins", "shutil", "subprocess", "ctypes"})

# 写模式标志字符
_WRITE_MODE_CHARS = frozenset("waxWAX+")


class FileOperationGuard:
    """文件操作防护器, 通过 Monkey Patch 拦截文件操作。

    单例模式, 通过 get_file_guard() 获取实例。
    """

    def __init__(self, logger=None, protected_paths=None):
        self._logger = logger
        self._protected_paths = [os.path.realpath(p) for p in (protected_paths or DEFAULT_PROTECTED_PATHS)]
        self._is_active = False
        self._lock = threading.Lock()

        # 保存原始函数引用
        self._originals = {}

    def is_active(self):
        return self._is_active

    def activate(self):
        """激活文件操作防护, 替换系统函数为受保护版本。"""
        with self._lock:
            if self._is_active:
                return

            # 保存原始函数
            self._originals = {
                "builtins.open": builtins.open,
                "os.remove": os.remove,
                "os.unlink": os.unlink,
                "os.rmdir": os.rmdir,
                "os.mkdir": os.mkdir,
                "os.makedirs": os.makedirs,
                "os.rename": os.rename,
                "os.replace": os.replace,
                "os.system": os.system,
                "os.popen": os.popen,
                "shutil.rmtree": shutil.rmtree,
                "shutil.copy": shutil.copy,
                "shutil.copy2": shutil.copy2,
                "shutil.copytree": shutil.copytree,
                "shutil.move": shutil.move,
                "subprocess.run": subprocess.run,
                "subprocess.call": subprocess.call,
                "subprocess.Popen": subprocess.Popen,
                "ctypes.CDLL": ctypes.CDLL,
                "importlib.reload": importlib.reload,
            }

            # 替换为受保护版本
            builtins.open = self._guarded_open()
            os.remove = self._guarded_path_op(os.remove, "remove")
            os.unlink = self._guarded_path_op(os.unlink, "unlink")
            os.rmdir = self._guarded_path_op(os.rmdir, "rmdir")
            os.mkdir = self._guarded_path_op(os.mkdir, "mkdir")
            os.makedirs = self._guarded_path_op(os.makedirs, "makedirs")
            os.rename = self._guarded_rename_op(os.rename, "rename")
            os.replace = self._guarded_rename_op(os.replace, "replace")
            os.system = self._guarded_system()
            os.popen = self._guarded_popen()
            shutil.rmtree = self._guarded_path_op(shutil.rmtree, "shutil.rmtree")
            shutil.copy = self._guarded_dst_op(shutil.copy, "shutil.copy")
            shutil.copy2 = self._guarded_dst_op(shutil.copy2, "shutil.copy2")
            shutil.copytree = self._guarded_dst_op(shutil.copytree, "shutil.copytree")
            shutil.move = self._guarded_dst_op(shutil.move, "shutil.move")
            subprocess.run = self._guarded_subprocess(subprocess.run, "subprocess.run")
            subprocess.call = self._guarded_subprocess(subprocess.call, "subprocess.call")
            subprocess.Popen = self._guarded_subprocess_popen()
            ctypes.CDLL = self._guarded_ctypes_cdll()
            importlib.reload = self._guarded_reload()

            self._is_active = True
            self._log_info("FileOperationGuard activated, protecting: %s", self._protected_paths)

    def deactivate(self):
        """停用文件操作防护, 恢复原始系统函数。"""
        with self._lock:
            if not self._is_active:
                return

            builtins.open = self._originals["builtins.open"]
            os.remove = self._originals["os.remove"]
            os.unlink = self._originals["os.unlink"]
            os.rmdir = self._originals["os.rmdir"]
            os.mkdir = self._originals["os.mkdir"]
            os.makedirs = self._originals["os.makedirs"]
            os.rename = self._originals["os.rename"]
            os.replace = self._originals["os.replace"]
            os.system = self._originals["os.system"]
            os.popen = self._originals["os.popen"]
            shutil.rmtree = self._originals["shutil.rmtree"]
            shutil.copy = self._originals["shutil.copy"]
            shutil.copy2 = self._originals["shutil.copy2"]
            shutil.copytree = self._originals["shutil.copytree"]
            shutil.move = self._originals["shutil.move"]
            subprocess.run = self._originals["subprocess.run"]
            subprocess.call = self._originals["subprocess.call"]
            subprocess.Popen = self._originals["subprocess.Popen"]
            ctypes.CDLL = self._originals["ctypes.CDLL"]
            importlib.reload = self._originals["importlib.reload"]

            self._is_active = False
            self._log_info("FileOperationGuard deactivated")

    # ------------------------------------------------------------------
    # 路径检查
    # ------------------------------------------------------------------

    def _is_protected(self, path):
        """检查路径是否在受保护目录下。"""
        if not path:
            return False
        try:
            real_path = os.path.realpath(str(path))
        except (TypeError, ValueError):
            return False
        return any(real_path.startswith(p + "/") or real_path == p for p in self._protected_paths)

    def _is_framework_caller(self):
        """通过调用栈检查是否为框架自身代码发起的调用。

        遍历调用栈, 如果所有非本模块帧都来自框架白名单路径, 则放行。
        只要发现有用户代码帧(不在白名单内), 就拦截。
        """
        try:
            stack = inspect.stack()
        except Exception:
            return False

        for frame_info in stack:
            filename = frame_info.filename or ""

            # 跳过本模块自身
            if "security/file_guard" in filename or "security\\file_guard" in filename:
                continue

            # 跳过 Python 标准库和内置模块
            if filename.startswith("<") or "/lib/python" in filename:
                continue

            # 检查是否在框架白名单内
            if any(pattern in filename for pattern in FRAMEWORK_CALLER_PATTERNS):
                continue

            # 发现非白名单调用者 → 用户代码
            return False

        return True

    def _check_and_raise(self, path, operation):
        """检查路径, 如果是受保护路径且调用者不在白名单, 则抛出 PermissionError。"""
        if self._is_protected(path) and not self._is_framework_caller():
            self._log_warning(
                "FileOperationGuard BLOCKED: %s on protected path: %s",
                operation,
                path,
            )
            raise PermissionError(
                f"[FileOperationGuard] Operation '{operation}' is not allowed on protected path: {path}"
            )

    # ------------------------------------------------------------------
    # 受保护的文件操作包装器
    # ------------------------------------------------------------------

    def _guarded_open(self):
        """创建受保护的 builtins.open, 仅拦截写模式。"""
        original_open = self._originals["builtins.open"]

        @functools.wraps(original_open)
        def guarded_open(file, mode="r", *args, **kwargs):
            # 只拦截写模式
            if any(c in mode for c in _WRITE_MODE_CHARS):
                self._check_and_raise(file, f"open(mode='{mode}')")
            return original_open(file, mode, *args, **kwargs)

        return guarded_open

    def _guarded_path_op(self, original_func, op_name):
        """创建受保护的单路径操作(remove/unlink/rmdir/mkdir/makedirs/shutil.rmtree)。"""

        @functools.wraps(original_func)
        def guarded(path, *args, **kwargs):
            self._check_and_raise(path, op_name)
            return original_func(path, *args, **kwargs)

        return guarded

    def _guarded_rename_op(self, original_func, op_name):
        """创建受保护的重命名操作(rename/replace), 检查源和目标路径。"""

        @functools.wraps(original_func)
        def guarded(src, dst, *args, **kwargs):
            self._check_and_raise(src, f"{op_name}(src)")
            self._check_and_raise(dst, f"{op_name}(dst)")
            return original_func(src, dst, *args, **kwargs)

        return guarded

    def _guarded_dst_op(self, original_func, op_name):
        """创建受保护的拷贝/移动操作, 检查目标路径。"""

        @functools.wraps(original_func)
        def guarded(src, dst, *args, **kwargs):
            self._check_and_raise(dst, op_name)
            return original_func(src, dst, *args, **kwargs)

        return guarded

    def _guarded_system(self):
        """创建受保护的 os.system。"""
        original_system = self._originals["os.system"]

        @functools.wraps(original_system)
        def guarded(command):
            if isinstance(command, str):
                for p in self._protected_paths:
                    if p in command:
                        if not self._is_framework_caller():
                            self._log_warning("FileOperationGuard BLOCKED: os.system with protected path in command")
                            raise PermissionError(
                                f"[FileOperationGuard] os.system is not allowed with protected path: {p}"
                            )
            return original_system(command)

        return guarded

    def _guarded_popen(self):
        """创建受保护的 os.popen。"""
        original_popen = self._originals["os.popen"]

        @functools.wraps(original_popen)
        def guarded(command, *args, **kwargs):
            if isinstance(command, str):
                for p in self._protected_paths:
                    if p in command:
                        if not self._is_framework_caller():
                            self._log_warning("FileOperationGuard BLOCKED: os.popen with protected path in command")
                            raise PermissionError(
                                f"[FileOperationGuard] os.popen is not allowed with protected path: {p}"
                            )
            return original_popen(command, *args, **kwargs)

        return guarded

    def _guarded_subprocess(self, original_func, op_name):
        """创建受保护的 subprocess.run/call。"""

        @functools.wraps(original_func)
        def guarded(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
            for p in self._protected_paths:
                if p in cmd_str:
                    if not self._is_framework_caller():
                        self._log_warning("FileOperationGuard BLOCKED: %s with protected path in command", op_name)
                        raise PermissionError(f"[FileOperationGuard] {op_name} is not allowed with protected path: {p}")
            return original_func(*args, **kwargs)

        return guarded

    def _guarded_subprocess_popen(self):
        """创建受保护的 subprocess.Popen。"""
        original_popen = self._originals["subprocess.Popen"]

        class GuardedPopen(original_popen):
            _guard = self

            def __init__(self_popen, *args, **kwargs):
                cmd = args[0] if args else kwargs.get("args", [])
                cmd_str = " ".join(str(c) for c in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
                for p in self_popen._guard._protected_paths:
                    if p in cmd_str:
                        if not self_popen._guard._is_framework_caller():
                            self_popen._guard._log_warning(
                                "FileOperationGuard BLOCKED: subprocess.Popen with protected path"
                            )
                            raise PermissionError(
                                f"[FileOperationGuard] subprocess.Popen is not allowed with protected path: {p}"
                            )
                super().__init__(*args, **kwargs)

        return GuardedPopen

    def _guarded_ctypes_cdll(self):
        """创建受保护的 ctypes.CDLL, 拦截加载 libc。"""
        original_cdll = self._originals["ctypes.CDLL"]

        @functools.wraps(original_cdll)
        def guarded(name, *args, **kwargs):
            if name and isinstance(name, str):
                lower_name = name.lower()
                if "libc" in lower_name:
                    if not self._is_framework_caller():
                        self._log_warning("FileOperationGuard BLOCKED: ctypes.CDLL loading libc: %s", name)
                        raise PermissionError(
                            f"[FileOperationGuard] Loading libc via ctypes.CDLL is not allowed: {name}"
                        )
            return original_cdll(name, *args, **kwargs)

        return guarded

    def _guarded_reload(self):
        """创建受保护的 importlib.reload, 阻止 reload 关键模块。"""
        original_reload = self._originals["importlib.reload"]

        @functools.wraps(original_reload)
        def guarded(module):
            module_name = getattr(module, "__name__", "")
            if module_name in _PROTECTED_MODULES:
                if not self._is_framework_caller():
                    self._log_warning("FileOperationGuard BLOCKED: importlib.reload(%s)", module_name)
                    raise PermissionError(f"[FileOperationGuard] Reloading module '{module_name}' is not allowed")
            return original_reload(module)

        return guarded

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def _log_info(self, msg, *args):
        formatted = msg % args if args else msg
        if self._logger:
            self._logger.info(formatted)

    def _log_warning(self, msg, *args):
        formatted = msg % args if args else msg
        if self._logger:
            self._logger.warning(formatted)


# ------------------------------------------------------------------
# 单例
# ------------------------------------------------------------------

_guard_instance = None
_guard_lock = threading.Lock()


def get_file_guard(logger=None, protected_paths=None):
    """获取 FileOperationGuard 单例。"""
    global _guard_instance
    if _guard_instance is None:
        with _guard_lock:
            if _guard_instance is None:
                _guard_instance = FileOperationGuard(logger=logger, protected_paths=protected_paths)
    return _guard_instance
