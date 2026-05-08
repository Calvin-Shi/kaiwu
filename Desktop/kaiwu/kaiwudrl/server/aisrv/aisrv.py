#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import sys
import faulthandler
import signal
import io
import os
import datetime
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.utils.common_func import make_single_dir
from kaiwudrl.common.utils.common_func import machine_device_check
from kaiwudrl.server.aisrv.aisrv_server_standard import AiServer
from common_python.logging.kaiwu_logger import KaiwuLogger


def proc_flags(configure_file):
    # 解析aisrv进程的配置
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_aisrv_configure()

    # 解析业务app的配置
    AppConf.load_conf(CONFIG.app_conf, CONFIG.svr_name)

    # 加载配置文件kaiwudrl/conf/algo_conf.json
    AlgoConf.load_conf(CONFIG.algo_conf)

    # 确保框架需要的文件目录存在
    make_single_dir(CONFIG.log_dir)

    # 主要是解决单机单进程下learner进程没有启动时, aisrv的一些操作出现没有文件目录的问题
    if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
        make_single_dir(CONFIG.restore_dir)
        make_single_dir(CONFIG.user_ckpt_dir)
        make_single_dir(CONFIG.summary_dir)
        make_single_dir(CONFIG.ckpt_dir)
        make_single_dir(CONFIG.pb_model_dir)
        make_single_dir(f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}")
        make_single_dir(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}")

    if (
        KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
        or KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
    ):
        # 设置TensorFlow日志级别
        from kaiwudrl.common.utils.tf_utils import set_tensorflow_log_level

        set_tensorflow_log_level()

        # actor需要设置在GPU机器上运行
        if "GPU" == CONFIG.actor_device_type:
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(get_local_rank())

    elif KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
        from kaiwudrl.common.utils.torch_utils import torch_is_gpu_available

    else:
        pass


def register_signal():
    try:
        faulthandler.register(signal.SIGUSR1)
    except io.UnsupportedOperation:
        pass


def register_pprof():
    """
    性能优化时按照需要打开, 发布版本一定不能打开
    """
    if CONFIG.enable_flamegraph:
        # 产生单独的日志句柄, 规避日志混淆问题
        logger = KaiwuLogger()
        current_pid = os.getpid()
        logger.set_logger_format(
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/aisrv_flamegraph_pid{current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "aisrv",
        )

        try:
            from common_python.pprof.server import start_flamegraph_server

            start_flamegraph_server(host=CONFIG.flamegraph_host, port=CONFIG.flamegraph_port, logger=logger)
            logger.info(f"进程 {os.getpid()} 已启动火焰图服务")
        except Exception as e:
            logger.warning(f"启动火焰图服务失败: {e}")


def app_check_param():
    """
    下面是目前业务的正确配置项, 如果配置错误, 则强制进行修正
    如果是on-policy但是设置的的remote_agent_default_runtime_mode是local_aisrv_workflow则报错退出, 目前不需要多份同样的代码
    """
    if (
        CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY
        and CONFIG.remote_agent_default_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW
    ):
        print(f"on-policy not run when remote_agent_default_runtime_mode is local_aisrv_workflow")
        return False

    return True


def check_param():
    """
    在进程启动前进行检测参数合理性
    """
    app_check_param_result = app_check_param()
    machine_device_check_result = machine_device_check(CONFIG.svr_name)

    return app_check_param_result and machine_device_check_result


def main():
    """
    启动命令样例: python3 kaiwudrl/server/aisrv/aisrv.py --conf=kaiwudrl/conf/kaiwudrl/aisrv.toml
    """

    # 限制 glibc malloc arena 数量, 防止 msgpack/numpy 高频序列化导致 arena 碎片化内存膨胀
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    os.chdir(CONFIG.project_root)

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_AISRV)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 步骤3, 检测输入参数正确性
    if not check_param():
        print("conf param error, please check")
        return

    # 步骤4, 处理信号
    register_signal()

    # 步骤5, 根据配置决定是否开启火焰图服务
    register_pprof()

    # 步骤6, 启动进程
    server = AiServer()
    server.run()


if __name__ == "__main__":
    sys.exit(main())
