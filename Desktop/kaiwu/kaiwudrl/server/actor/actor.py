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


import os
import sys
import faulthandler
import signal
import io
import time
import datetime
import multiprocessing
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.utils.common_func import get_local_rank, get_gpu_machine_type
from kaiwudrl.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper
from kaiwudrl.common.utils.common_func import machine_device_check
from kaiwudrl.server.actor.actor_server_sync import ActorServerSync
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    make_single_dir,
    actor_learner_aisrv_count,
    get_host_ip,
    decompress_data,
    decompress_data_parallel,
)
from common_python.monitor.process_health_monitor import ProcessHealthMonitor, ProcessInfo, ProcessExitReason


def proc_flags(configure_file):
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_actor_configure()

    # 加载配置文件kaiwudrl/conf/algo_conf.json
    AlgoConf.load_conf(CONFIG.algo_conf)

    # 加载配置文件kaiwudrl/conf/app_conf.json
    AppConf.load_conf(CONFIG.app_conf)

    # 框架运行前创建必要的文件目录
    make_single_dir(CONFIG.log_dir)
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
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/actor_flamegraph_pid{current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "actor",
        )

        try:
            from common_python.pprof.server import start_flamegraph_server

            start_flamegraph_server(host=CONFIG.flamegraph_host, port=CONFIG.flamegraph_port, logger=logger)
            logger.info(f"进程 {current_pid} 已启动火焰图服务")
        except Exception as e:
            logger.warning(f"启动火焰图服务失败: {e}")


def app_check_param():
    """
    下面是目前业务的正确配置项, 如果配置错误, 则强制进行修正
    """

    return True


def model_file_check():
    """
    如果是eval模式, 需要验证加载的model文件是否正常, 包括:
    1. model文件是否存在
    2. model文件是否加载正常
    """
    if CONFIG.run_mode != KaiwuDRLDefine.RUN_MODE_EVAL:
        return True

    if not CONFIG.eval_model_dir:
        print(f"eval_model_dir {CONFIG.eval_model_dir} is empty")
        return False

    # 如果是tensorflow是目录, 如果是pytorch是文件
    if (
        CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE
        or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX
        or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT
    ):
        # 因为tensorflow的需要加载图后才能判断是否正确, 故放到predictor.py里实现
        if not os.path.exists(CONFIG.eval_model_dir + ".meta"):
            print(f"eval_model_dir {CONFIG.eval_model_dir} is not exist")
            return False
        return True
        # return tensorflow_model_file_valid(CONFIG.eval_model_dir)
    elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH:

        # return pytorch_model_file_valid(CONFIG.eval_model_dir)
        return True
    elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TCNN:
        pass
    else:
        pass

    return False


def check_param():
    """
    在进程启动前进行检测参数合理性, 按照业务来区分
    """

    app_check_result = app_check_param()

    model_file_check_result = model_file_check()

    machine_device_check_result = machine_device_check(CONFIG.svr_name)

    return app_check_result and model_file_check_result and machine_device_check_result


def predictor_loop(actor_send_server, actor_recv_server):

    # 放置在这里, 规避配置文件没有解析时就出现异常的问题
    from kaiwudrl.server.actor.predictor import Predictor

    if CONFIG.actor_server_predict_server_different_queue:
        predictor_queues = []

    # actor上开启的predict进程列表
    predictor_process_objects = []

    # 创建进程健康监控器
    logger = KaiwuLogger()
    logger.set_logger_format(
        f"{CONFIG.log_dir}/{CONFIG.svr_name}/actor_process_monitor_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
        "actor_process_monitor",
    )
    process_health_monitor = ProcessHealthMonitor(
        logger=logger, check_interval=CONFIG.prometheus_stat_per_minutes, log_tag="actor_process_monitor"
    )

    def on_predictor_exit(process_info: ProcessInfo):
        """predictor进程退出告警"""
        if process_info.exit_reason == ProcessExitReason.OOM_KILLED:
            logger.error(
                f"actor_process_monitor Predictor process OOM! name={process_info.name}, pid={process_info.pid}",
                g_not_server_label,
            )
        else:
            logger.warning(
                f"actor_process_monitor Predictor process exited: name={process_info.name}, pid={process_info.pid}, reason={process_info.exit_reason.value}",
                g_not_server_label,
            )

    process_health_monitor.register_alert_callback(on_predictor_exit)

    """
    如果在大规模场景下, 因为model_file_sync进程只需要启动1个, 而actor的predict进程是多个的, 故这里需要采用下面步骤:
    1. model_file_sync进程先启动
    2. 将model_file_sync进程的对象句柄传入到actor的predict进程里进行使用
    """

    model_file_sync_wrapper = ModelFileSyncWrapper()
    model_file_sync_wrapper.init()

    # 根据配置文件kaiwudrl/conf/actor_conf.json找到本次使用的predictor类
    for i in range(CONFIG.actor_predict_process_num):
        predictor = Predictor(actor_send_server, actor_recv_server)
        predictor.set_index(i)
        predictor_process_objects.append(predictor)

        # predictor进程将queue注册到actor_server去
        if CONFIG.actor_server_predict_server_different_queue:

            """
            管道引起性能下降的原因是管道长度操作系统确定64KB, 且无法修改, 如果收方不及时的取走数据, 则发送方阻塞
            # 读方, 写方
            predict_conn, actor_server_conn = multiprocessing.Pipe(duplex=False)
            """

            predict_request_queue = multiprocessing.Queue(CONFIG.queue_size)

            predictor_queues.append(predict_request_queue)

            predictor.set_predict_request_queue_from_actor_server(predict_request_queue)

        predictor.set_model_file_sync_wrapper(model_file_sync_wrapper)

    """
    下面的逻辑主要是为在on-policy实现设计的, 因为单个actor只有1个端口由zmq使用, 且有多个预测进程,
    故这里不能以端口作为唯一标志, 而是需要采用预测进程内通信方式
    需要倒序输出, 此时第0号主进程作为主进程, 然后主进程是需要通知其他进程消息的
    """
    for i in range(CONFIG.actor_predict_process_num - 1, -1, -1):
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if i:
                # 采用pipe通信, 读方, 写方, 因为是同步的场景, 故设置为双向管道
                slave_conn, master_conn = multiprocessing.Pipe(duplex=True)
                predictor_process_objects[0].append_predictor_master_conn(master_conn)
                predictor_process_objects[i].set_predictor_slave_conn(slave_conn)

        # 启动预测进程
        predictor_process_objects[i].start()

        # 注册predictor进程到健康监控器
        process_health_monitor.register_process(
            pid=predictor_process_objects[i].pid, name=f"predictor_{i}", process_type="predictor"
        )

    if CONFIG.actor_server_predict_server_different_queue:
        actor_recv_server.set_predict_request_queues(predictor_queues)

    if CONFIG.actor_server_async:
        actor_send_server.start()
        actor_recv_server.start()

        # 注册actor_send_server和actor_recv_server
        process_health_monitor.register_process(
            pid=actor_send_server.pid, name="actor_send_server", process_type="actor_server"
        )
        process_health_monitor.register_process(
            pid=actor_recv_server.pid, name="actor_recv_server", process_type="actor_server"
        )

    else:
        actor_send_server.start()

        # 注册actor_send_server
        process_health_monitor.register_process(
            pid=actor_send_server.pid, name="actor_send_server", process_type="actor_server"
        )

    # 主循环：定期检查子进程健康状态
    logger.info("actor_process_monitor Actor process health monitor started", g_not_server_label)

    while True:
        try:
            # 执行健康检查
            process_health_monitor.check_once()

            # 休眠一段时间
            time.sleep(5)

        except KeyboardInterrupt:
            logger.info("actor_process_monitor Actor process health monitor stopped by user", g_not_server_label)
            break
        except Exception as e:
            logger.error(f"actor_process_monitor Actor process health monitor error: {e}", g_not_server_label)


def gpu_machine_engine():
    """
    流程如下:
    1. 判定当前GPU机器类型
    2. 拷贝相关文件到对应目录
    """

    gpu_machine_type = get_gpu_machine_type()
    if gpu_machine_type is None:
        return True, gpu_machine_type

    # 因为有存在在不是GPU机器上运行的情况, 故这里不做强判断
    print(f"current gpu machine is {gpu_machine_type}")

    """
    代码里不能调用cp tensorrt的大文件, 容易出现异常, 故这里解决方案
    1. 打镜像时, 采用shell将相关的文件拷贝
    2. 在进程启动前, 采用shell将相关的文件拷贝
    """

    return True, gpu_machine_type


def main():
    """
    启动命令样例: python3 kaiwudrl/server/actor/actor.py --conf=kaiwudrl/conf/kaiwudrl/actor.toml
    """

    # 限制 glibc malloc arena 数量, 防止高频 malloc/free 场景下 arena 碎片化导致内存膨胀
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    os.chdir(CONFIG.project_root)

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_ACTOR)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 步骤3, 检测输入参数正确性
    if not check_param():
        print("conf param error, please check")
        return

    # 步骤4, 支持异构GPU, 主要是tensorrt
    ret, gpu_machine_type = gpu_machine_engine()
    if not ret:
        print(f"unsupport gpu_machine_type or error {gpu_machine_type} , please check")
        return

    # 步骤5, 处理信号
    register_signal()

    # 步骤6, 根据配置决定是否开启火焰图服务
    register_pprof()

    # 步骤7, 启动ActorServer
    actor_send_server = ActorServerSync()
    actor_recv_server = actor_send_server

    # 步骤8, 开始预测
    predictor_loop(actor_send_server, actor_recv_server)

    # 为了model_file_sync里的multiprocessing.Manager通信存在且正常
    while True:
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
