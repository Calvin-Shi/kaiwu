#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
"""
Training smoke test engine.

Provides a reusable function to run end-to-end training smoke tests.
Each project only needs to specify its algorithm configuration and
project-specific environment variables, then call run_train_test().
"""


import glob
import os
import platform
import shutil
import sys
import time
from multiprocessing import Process
from typing import List

from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.common_func import (
    python_exec_shell,
    scan_for_errors,
)
import kaiwudrl.server.aisrv.aisrv as aisrv
import kaiwudrl.server.learner.learner as learner


# Base environment variables shared by all projects
# 所有项目共用的基础环境变量
BASE_ENV_VARS = {
    "use_ckpt_sync": "False",
    "replay_buffer_capacity": "10",
    "train_batch_size": "2",
    "use_prometheus": "True",
    "dump_model_freq": "1",
    "aisrv_connect_to_kaiwu_env_count": "1",
    "is_train_test": "True",
    "aisrv_env_ipc_method": "zmq",
}


def _stop_all(shell: str = "sh", extra_stop_commands: list = None):
    """Execute stop commands and exit.

    :param shell: Shell to use ('sh' or 'bash').
    :param extra_stop_commands: Additional stop commands to execute.

    设计目标：train_test 是冒烟测试，跑完后调用 tools/stop.sh all
    停掉所有业务组件并释放 GPU 显存。GPU 清理的实际逻辑在
    tools/clean_gpu.sh，由 stop.sh all 分支内部调用，以 setsid
    独立会话后台运行，fire-and-forget 语义。
    """
    # 业务层停机 + GPU 显存清理
    # stop.sh all 会先启动 clean_gpu.sh（独立 setsid 会话 fire-and-forget），
    # 再依次 kill modelpool/actor/learner/.../train_test 各组件。即使 kill
    # train_test 把本进程链带走，clean_gpu.sh 也已经脱离进程组独立运行，
    # sleep 3 后扫 /proc/*/fd/nvidia* 兜底清理 GPU 残留。
    python_exec_shell(f"{shell} tools/stop.sh all")
    if extra_stop_commands:
        for cmd in extra_stop_commands:
            python_exec_shell(cmd)

    # 主进程主动退出（通常走不到这里，stop.sh 的 kill "train_test" 已经杀了自己）
    os._exit(0)


def _check_process(
    proc: Process,
    shell: str = "sh",
    extra_stop_commands: list = None,
):
    """Check if a process is alive, stop all if not.

    :param proc: Process to check.
    :param shell: Shell to use for stop commands.
    :param extra_stop_commands: Additional stop commands to execute.
    """
    if not proc.is_alive():
        print(
            "\033[1;31m"
            + f"{proc.name} (pid={proc.pid}) is not alive, "
            + f"exitcode={proc.exitcode}, please check error log"
            + "\033[0m",
            flush=True,
        )
        time.sleep(5)
        _stop_all(shell, extra_stop_commands)
    else:
        print(f"{proc.name} is alive", flush=True)


def _check_process_stop_done():
    """Check process_stop.done file content (tri-state).

    Tri-state semantics:
      - Returns True  : file exists AND content parses to integer 0
                        -> exit the monitoring loop as SUCCESS.
      - Returns False : file exists but content is non-zero, unparseable,
                        or any read exception occurs
                        -> exit the monitoring loop as FAILURE.
      - Returns None  : file does NOT exist yet
                        -> keep waiting (do NOT exit).

    检查 process_stop.done 文件内容（三态返回）：
      - True ：文件存在且内容可解析为整数 0，表示成功退出；
      - False：文件存在但内容非 0、解析失败或读取异常，表示失败退出；
      - None ：文件不存在，继续等待，不退出。

    :returns: True on success-exit, False on failure-exit, None to keep waiting.
    """
    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    if not os.path.exists(done_file):
        # File not ready yet; caller should keep polling.
        # 文件尚未生成，调用方继续轮询。
        return None
    try:
        with open(done_file, "r") as f:
            content = f.read().strip()
        # Parse as integer; only value 0 means success-exit.
        # 按整数解析，值为 0 才算成功退出。
        return int(content) == 0
    except (ValueError, TypeError) as e:
        print(
            "\033[1;31m" + f"parse {done_file} content to int failed, error={e}, treat as failure-exit" + "\033[0m",
            flush=True,
        )
        return False
    except Exception as e:
        print(
            "\033[1;31m" + f"read {done_file} failed, error={e}, treat as failure-exit" + "\033[0m",
            flush=True,
        )
        return False


def _check_train_success_glob_pkl() -> bool:
    """Check training success by glob matching model.ckpt-[1-9]*.pkl files.

    :returns: True if matching model files found and cleaned up.
    """
    model_files = glob.glob(f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/model.ckpt-[1-9]*.pkl")
    if model_files:
        for file in model_files:
            if os.path.exists(file):
                os.remove(file)
        return True
    return False


def _check_train_success_ckpt1() -> bool:
    """Check training success by looking for model.ckpt-1 with .npy or .pkl extension.

    :returns: True if model.ckpt-1 file found and cleaned up.
    """
    model_base = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/model.ckpt-1"
    for ext in [".npy", ".pkl"]:
        model_file = model_base + ext
        if os.path.exists(model_file):
            os.remove(model_file)
            return True
    return False


def _check_train_success_listdir_count() -> bool:
    """Check training success by counting model.ckpt-* files in the directory.

    :returns: True if at least 4 model checkpoint files exist.
    """
    folder = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/"
    prefix = "model.ckpt-"
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        return False

    count = 0
    for fname in files:
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath) and fname.startswith(prefix):
            count += 1
    return count >= 4


# Model check method dispatch table
# 模型检测方式分发表
_CHECK_MODEL_METHODS = {
    "glob_pkl": _check_train_success_glob_pkl,
    "check_ckpt1": _check_train_success_ckpt1,
    "listdir_count": _check_train_success_listdir_count,
}


def run_train_test(
    algorithm_name: str,
    algorithm_name_list: list,
    env_vars: dict = None,
    conf_path: str = "kaiwudrl/conf/kaiwudrl/learner.toml",
    shell: str = "sh",
    extra_stop_commands: list = None,
    check_model_method: str = "glob_pkl",
    skip_aisrv_alive_check: bool = False,
    skip_error_scan: bool = False,
    check_train_success_flag: bool = True,
):
    """Run an end-to-end training smoke test.

    This function encapsulates the common train_test.py logic shared across
    all kaiwu_projects. Each project only needs to pass its specific
    configuration parameters.

    :param algorithm_name: Algorithm to use for training.
    :param algorithm_name_list: List of supported algorithm names.
    :param env_vars: Project-specific environment variables (merged onto BASE_ENV_VARS).
    :param conf_path: Path to the learner config file.
    :param shell: Shell to use ('sh' or 'bash').
    :param extra_stop_commands: Additional stop commands on exit (e.g. ['sh tools/stop_sim_python.sh']).
    :param check_model_method: Model detection method: 'glob_pkl' | 'check_ckpt1' | 'listdir_count'.
    :param skip_aisrv_alive_check: Skip aisrv alive check (for non-blocking aisrv.main).
    :param skip_error_scan: Skip error log scanning.
    :param check_train_success_flag: Whether to check model file to determine training success.
    """
    try:
        _run_train_test_impl(
            algorithm_name=algorithm_name,
            algorithm_name_list=algorithm_name_list,
            env_vars=env_vars,
            conf_path=conf_path,
            shell=shell,
            extra_stop_commands=extra_stop_commands,
            check_model_method=check_model_method,
            skip_aisrv_alive_check=skip_aisrv_alive_check,
            skip_error_scan=skip_error_scan,
            check_train_success_flag=check_train_success_flag,
        )
    except KeyboardInterrupt:
        print("\033[1;31m" + "KeyboardInterrupt, please check" + "\033[0m")
        _stop_all(shell, extra_stop_commands)


def _run_train_test_impl(
    algorithm_name: str,
    algorithm_name_list: list,
    env_vars: dict = None,
    conf_path: str = "kaiwudrl/conf/kaiwudrl/learner.toml",
    shell: str = "sh",
    extra_stop_commands: list = None,
    check_model_method: str = "glob_pkl",
    skip_aisrv_alive_check: bool = False,
    skip_error_scan: bool = False,
    check_train_success_flag: bool = True,
):
    """Internal implementation of the training smoke test.

    :param algorithm_name: Algorithm to use for training.
    :param algorithm_name_list: List of supported algorithm names.
    :param env_vars: Project-specific environment variables.
    :param conf_path: Path to the learner config file.
    :param shell: Shell to use.
    :param extra_stop_commands: Additional stop commands.
    :param check_model_method: Model detection method.
    :param skip_aisrv_alive_check: Skip aisrv alive check.
    :param skip_error_scan: Skip error log scanning.
    :param check_train_success_flag: Whether to check model file for success.
    """
    start_time = time.time()

    # Step 1: Set environment variables
    # 设置环境变量
    merged_env = dict(BASE_ENV_VARS)
    if env_vars:
        merged_env.update(env_vars)
    os.environ.update(merged_env)

    # Step 2: Validate algorithm name and switch algorithm
    # 校验算法名并切换算法
    if algorithm_name not in algorithm_name_list:
        print("\033[92m" + f"algorithm_name: {algorithm_name} not in list {algorithm_name_list}" + "\033[0m")
        _stop_all(shell, extra_stop_commands)

    python_exec_shell(f"{shell} /root/tools/change_algorithm_all.sh {algorithm_name}")
    print(f"current algorithm_name is {algorithm_name}")

    # Step 3: Set sample transmission type based on architecture
    # 根据 CPU 架构选择 reverb 或 zmq
    architecture = platform.machine()
    platform_maps = {
        "aarch64": "zmq",
        "arm64": "zmq",
        "x86_64": "reverb",
        "AMD64": "reverb",
    }
    sample_tool_type = platform_maps.get(architecture)
    if sample_tool_type is None:
        print(f"Architecture '{architecture}' may not exist or not be supported.")
    else:
        result_code, result_str = python_exec_shell(f"{shell} tools/change_sample_server.sh {sample_tool_type}")
        if result_code != 0:
            raise ValueError(f"Execution error! Please check the error detail: {result_str}")

    # Step 4: Parse configuration
    # 解析配置
    CONFIG.set_configure_file(conf_path)
    CONFIG.parse_learner_configure()

    # Step 5: Clean up previous model files and logs
    # 清理旧的模型文件和日志
    python_exec_shell(f"rm -rf {CONFIG.user_ckpt_dir}/{CONFIG.app}_{algorithm_name}/*")
    python_exec_shell(f"rm -rf {CONFIG.log_dir}/*")

    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    if os.path.exists(done_file):
        os.remove(done_file)

    # Clean model files based on the check method
    # 根据检测方式清理模型文件
    # 只按前缀 model.ckpt-* 匹配，不限制后缀，兼容业务侧自定义保存格式（.pkl/.pt/.pth/.safetensors/无后缀/目录型等）
    model_pattern = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/model.ckpt-*"
    for path in glob.glob(model_pattern):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path) or os.path.islink(path):
            os.remove(path)

    # Step 6: Start learner, aisrv, and modelpool processes
    # 启动 learner、aisrv、modelpool 进程
    procs: List[Process] = []
    procs.append(Process(target=learner.main, name="learner"))
    procs.append(Process(target=aisrv.main, name="aisrv"))
    python_exec_shell(f"{shell} tools/modelpool_start.sh learner")

    for proc in procs:
        proc.start()
        time.sleep(10)
        if skip_aisrv_alive_check and proc.name == "aisrv":
            print(
                f"{proc.name} started (skip alive check, non-blocking main)",
                flush=True,
            )
        else:
            _check_process(proc, shell, extra_stop_commands)

    # Step 7: Main monitoring loop
    # 主监控循环
    check_model_func = _CHECK_MODEL_METHODS.get(check_model_method, _check_train_success_glob_pkl)

    while True:
        error_msg = None
        if not skip_error_scan:
            error_msg = scan_for_errors(CONFIG.log_dir, error_indicator="ERROR")

        train_success = check_model_func()
        # Tri-state: True=success-exit, False=failure-exit, None=keep waiting
        # 三态：True=成功退出，False=失败退出，None=继续等待
        process_done = _check_process_stop_done()

        # If error log found, exit early as failure
        # 如果有错误日志产生，作为失败提前退出
        if error_msg:
            time.sleep(5)
            print(
                f"\033[1;31m"
                + f"Train test failed, algorithm_name {algorithm_name}, find error log, please check"
                + "\033[0m"
            )
            _stop_all(shell, extra_stop_commands)
        elif process_done is False:
            # process_stop.done exists but its value is non-zero -> failure exit
            # process_stop.done 存在但值非 0 -> 失败退出
            time.sleep(5)
            print(
                f"\033[1;31m"
                + f"Train test failed, algorithm_name {algorithm_name}, find error log, please check, will exit, cost "
                + f"{time.time() - start_time:.2f} seconds"
                + "\033[0m"
            )
            _stop_all(shell, extra_stop_commands)
        else:
            # process_done is True (success) or None (keep waiting)
            # process_done 为 True（成功）或 None（继续等待）
            should_exit = process_done is True
            if check_train_success_flag:
                should_exit = should_exit or train_success

            if should_exit:
                time.sleep(5)
                print(
                    f"\033[1;32m"
                    + f"Train test succeeded, algorithm_name {algorithm_name}, will exit, cost "
                    + f"{time.time() - start_time:.2f} seconds"
                    + "\033[0m"
                )
                _stop_all(shell, extra_stop_commands)

        time.sleep(1)
        for proc in procs:
            if skip_aisrv_alive_check and proc.name == "aisrv":
                continue
            _check_process(proc, shell, extra_stop_commands)
