#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import sys
import os
import time
import setproctitle
from multiprocessing import Process
import signal
import atexit
from kaiwudrl.common.utils.common_func import python_exec_shell


class ServiceManager:
    def __init__(self, service_name):
        self.processes = {}
        self.service_name = service_name
        self.service_map = {
            "learner": {"module": "kaiwudrl.server.learner.learner", "log": "log/learner.log"},  # 新增日志路径配置
            "actor": {"module": "kaiwudrl.server.actor.actor", "log": "log/actor.log"},
            "aisrv": {"module": "kaiwudrl.server.aisrv.aisrv", "log": "log/aisrv.log"},
        }

        # 注册退出清理
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def _service_wrapper(self, module_path, log_file, service_name):
        """带日志重定向的服务包装器"""
        import importlib

        # 设置进程名
        service_name = f"python3 kaiwudrl/server/{service_name}/{service_name}.py --conf=kaiwudrl/conf/kaiwudrl/{service_name}.toml"
        setproctitle.setproctitle(service_name)

        # 配置日志重定向
        sys.stdout = open(log_file, "a", buffering=1)
        sys.stderr = sys.stdout  # 将stderr也重定向到同一文件

        # eval时需要设置下环境变量的run_mode
        os.environ["RUN_MODE"] = "eval"

        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "main"):
                module.main()
            else:
                print(f"{module_path} has no main function!")
        except Exception as e:
            print(f"Service {module_path} failed: {str(e)}")
        finally:
            sys.stdout.close()
            sys.stderr = sys.__stderr__

    def start_service(self, service_name):
        """启动单个服务"""
        if service_name not in self.service_map:
            raise ValueError(f"Unknown service: {service_name}")

        config = self.service_map[service_name]
        # 传递日志路径参数
        p = Process(target=self._service_wrapper, args=(config["module"], config["log"], service_name), daemon=False)
        p.start()
        self.processes[service_name] = p
        print(f"[{service_name}] Started with PID {p.pid}, logging to {config['log']}")

    def start_all(self):
        """批量启动所有服务"""
        if self.service_name == "all":
            for service in self.service_map.keys():
                self.start_service(service)

            python_exec_shell(f"sh tools/modelpool_start.sh learner")
        else:
            for service in self.service_map.keys():
                if service == self.service_name:
                    self.start_service(service)

            if self.service_name == "learner":
                python_exec_shell(f"sh tools/modelpool_start.sh learner")
            elif self.service_name == "actor":
                python_exec_shell(f"sh tools/modelpool_start.sh actor")
            else:
                pass

    def cleanup(self):
        """清理所有子进程"""
        print("Terminating child processes...")
        for name, p in self.processes.items():
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.exitcode is None:
                    print(f"[{name}] Force killing process")
                    p.kill()
                print(f"[{name}] Terminated")

    def signal_handler(self, signum, frame):
        """信号处理"""
        print(f"Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)


if __name__ == "__main__":

    service_name = sys.argv[1] if len(sys.argv) > 1 else None
    if service_name is None:
        service_name = "all"

    manager = ServiceManager(service_name)

    try:
        # 启动核心服务
        manager.start_all()

        # 主进程保持存活
        while True:
            for name, p in manager.processes.items():
                if not p.is_alive():
                    print(f"[{name}] Process died unexpectedly! Exit code: {p.exitcode}")
                    manager.cleanup()
                    sys.exit(1)
            time.sleep(60)

    except KeyboardInterrupt:
        manager.cleanup()
    except Exception as e:
        print(f"Master controller crashed: {str(e)}")
        manager.cleanup()
        sys.exit(1)
