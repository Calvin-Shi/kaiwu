#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# 必须放在torch前导入, 否则兼容问题需要解决
try:
    from isaacgym import gymutil, gymapi
except Exception as e:
    pass

# 明确的在文件开始就设置下参数, 规避调用了其他库的函数从而不生效问题
import torch
import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
import multiprocessing as mp
from multiprocessing import get_context
import os
import datetime
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from common_python.worker.worker import Worker, WorkerConfig
import importlib
from common_python.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.components.environment.env_wrapper import ExtraInfo


class IsaacProcessWorker(Worker):
    """Isaac环境工作进程"""

    def __init__(self, cmd_queue, data_queue):
        self.cmd_queue = cmd_queue
        self.data_queue = data_queue

        # 解析aisrv进程的配置, 因为是独立的进程需要单独的解析下
        configure_file = f"kaiwudrl/conf/kaiwudrl/aisrv.toml"
        CONFIG.set_configure_file(configure_file)
        CONFIG.parse_aisrv_configure()

        # 进程pid
        self.current_pid = os.getpid()
        # 由于采用的是spawn模式, logger, monitor, alloc对象都无法序列化, 故无法使用
        worker_config = WorkerConfig(
            worker_name="isaac_env",
            father_pid=self.current_pid,
            use_logger=False,
            use_default_monitor=False,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        # 根据配置获取到具体的对象
        module = importlib.import_module(CONFIG.env_business_module_name)
        cls = getattr(module, CONFIG.env_business_class_name)
        # 实例化业务对象
        self.env = cls()

    def before_run(self):
        # 先调用基类初始化
        if not super().before_run():
            return False

        # 由于从aisrv无法通过多进程传递KaiwuLogger对象, 故env自己定义了日志句柄
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        params = {
            "compression": CONFIG.compression,
            "encoding": CONFIG.encoding,
            "rotation": CONFIG.rotation,
            "level": CONFIG.level,
            "serialize": CONFIG.serialize,
            "retention": CONFIG.retention,
            "max_single_message_len": CONFIG.max_single_message_len,
            "max_calls_log_per_min": CONFIG.max_calls_log_per_min,
        }
        self.logger.set_logger_format(
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/kaiwu_env_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "isaac_env",
            params,
        )

        self.logger.info(f"isaac_env start at pid {self.current_pid}, env is {self.env}")
        return True

    def after_run(self) -> bool:
        pass

    def run_once(self) -> None:
        cmd, payload = self.cmd_queue.get()
        if cmd == "reset":
            self._handle_reset(payload)
            return False

        elif cmd == "step":
            self._handle_step(payload)
            return False

        elif cmd == "apply_function":
            func_name, args, kwargs = payload
            self._handle_apply_function(func_name, *args, **kwargs)
            return False

        elif cmd == "close":
            self._close()
            self.data_queue.put((0, "close_ok", None))
            return True

        else:
            return False

    def run(self):
        # before_run
        if not self.before_run():
            self.logger.error(f"isaac_env before_run failed, so return")
            return

        try:
            while True:
                done = self.run_once()
                if done:
                    break
        except Exception as e:
            raise RuntimeError(f"isaac_env Environment run failed, error msg is {str(e)}")

    def _handle_reset(self, usr_conf):
        """
        直接调用业务的实现
        """
        try:
            result = self.env.reset(usr_conf)
            self.data_queue.put((0, "success", result))
        except Exception as e:
            error_msg = f"Error executing reset: {str(e)}"
            self.data_queue.put((1, error_msg, None))

    def _handle_step(self, actions):
        """
        直接调用业务的实现
        """
        try:
            result = self.env.step(actions)
            self.data_queue.put((0, "success", result))
        except Exception as e:
            error_msg = f"Error executing step: {str(e)}"
            self.data_queue.put((1, error_msg, None))

    def _handle_apply_function(self, func_name, *args, **kwargs):
        """
        直接调用业务的实现
        """

        # 检查环境是否有这个函数
        try:
            if hasattr(self.env, func_name) and callable(getattr(self.env, func_name)):
                # 获取函数并调用
                func = getattr(self.env, func_name)
                result = func(*args, **kwargs)
                self.data_queue.put((0, "success", result))
            else:
                # 函数不存在
                error_msg = f"Function '{func_name}' not found in environment"
                self.data_queue.put((1, error_msg, None))
        except Exception as e:
            # 函数执行出错
            error_msg = f"Error executing function '{func_name}': {str(e)}"
            self.data_queue.put((2, error_msg, None))

    def _close(self):
        """
        直接调用业务的实现
        """
        self.env.close()


class IsaacEnv:
    """
    进程管理器（使用显式上下文）
    """

    def __init__(self, logger, monitor_proxy) -> None:

        # 使用独立上下文
        self.ctx = get_context("spawn")
        self.cmd_queue = self.ctx.Queue()
        self.data_queue = self.ctx.Queue()

        # 创建工作进程实例
        self.worker = self.ctx.Process(target=self._worker_main, args=(self.cmd_queue, self.data_queue))
        self.worker.start()

        super(IsaacEnv, self).__init__()

    def _worker_main(self, cmd_q, data_q):
        """工作进程主函数"""
        worker = IsaacProcessWorker(cmd_q, data_q)
        worker.run()

    def init(self, ip, port):
        pass

    def reset(self, usr_conf):
        # 是否是评估是框架侧知晓的, 故这里设置进去
        usr_conf["is_eval"] = CONFIG.run_mode in [KaiwuDRLDefine.RUN_MODE_EVAL, KaiwuDRLDefine.RUN_MODE_EXAM]
        game_id = f'{os.getenv("KAIWU_TASK_ID", "1")}_{os.getenv("KAIWU_ROUND_INDEX", "1")}_1'
        usr_conf["game_id"] = game_id

        self.cmd_queue.put(("reset", usr_conf))
        result_code, result_message, data = self.data_queue.get()

        """
        成功的情况, 直接返回data
        不成功的情况, 需要更新下result_code和result_message, 因为此时是try catch出现的
        """
        if result_code != 0:
            data["extra_info"]["result_code"] = result_code
            data["extra_info"]["result_message"] = result_message

        return data

    def step(self, action_data):
        self.cmd_queue.put(("step", action_data))
        result_code, result_message, data = self.data_queue.get()
        """
        成功的情况, 直接返回data
        不成功的情况, 需要更新下result_code和result_message, 因为此时是try catch出现的
        """
        if result_code != 0:
            data["extra_info"]["result_code"] = result_code
            data["extra_info"]["result_message"] = result_message

        return data

    def apply_function(self, func_name, *args, **kwargs):
        """
        应用环境中的任意函数
        """
        self.cmd_queue.put(("apply_function", (func_name, args, kwargs)))
        result_code, result_message, data = self.data_queue.get()
        """
        成功的情况, 直接返回data
        不成功的情况, 需要更新下result_code和result_message, 因为此时是try catch出现的
        """
        if result_code != 0:
            data["extra_info"]["result_code"] = result_code
            data["extra_info"]["result_message"] = result_message

        return data

    def close(self):
        self.cmd_queue.put(("close", None))
        result_code, result_message, data = self.data_queue.get()
        self.worker.join(timeout=30)

        if self.worker.is_alive():
            self.worker.terminate()
            print("工作进程未正常退出，已强制终止")
