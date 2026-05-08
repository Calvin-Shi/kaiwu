#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import faulthandler
import signal
import os
import io
import sys
import time
import datetime
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

# 某些机器环境没有安装tensorflow的, 故需要按需安装
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.config.app_conf import AppConf
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.monitor.monitor_config_builder import (
    MonitorConfigBuilder,
    load_monitor_config_from_yaml,
    load_user_monitor_config,
    add_log_prefix,
)
from kaiwudrl.common.checkpoint.model_file_common import (
    process_stop_write_file,
)
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    make_single_dir,
    actor_learner_aisrv_count,
    get_host_ip,
    get_uuid,
    register_sigterm_handler,
    stop_process_by_name,
    get_local_rank,
    machine_device_check,
)
from common_python.monitor.process_health_monitor import ProcessHealthMonitor, ProcessInfo, ProcessExitReason


def proc_flags(configure_file):
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_learner_configure()

    # replay_producer 模式下强制启用 v2 监控，确保生成 monitor.yaml 供 Vector 双写使用
    running_mode = os.environ.get(
        KaiwuDRLDefine.ENV_KAIWU_RUNNING_MODE,
        KaiwuDRLDefine.RUNNING_MODE_NORMAL,
    )
    if running_mode == KaiwuDRLDefine.RUNNING_MODE_REPLAY_PRODUCER:
        CONFIG.monitor_version = KaiwuDRLDefine.MONITOR_VERSION_V2

    # 加载配置文件kaiwudrl/conf/algo_conf.json
    AlgoConf.load_conf(CONFIG.algo_conf)

    # 加载配置文件kaiwudrl/conf/app_conf.json
    AppConf.load_conf(CONFIG.app_conf)

    # 框架运行前创建必要的文件目录
    make_single_dir(CONFIG.log_dir)
    make_single_dir(CONFIG.restore_dir)
    make_single_dir(CONFIG.user_ckpt_dir)
    make_single_dir(CONFIG.summary_dir)
    make_single_dir(CONFIG.ckpt_dir)
    make_single_dir(CONFIG.pb_model_dir)
    make_single_dir(f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}")
    make_single_dir(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}")

    # 按照需要创造旁路文件
    if int(CONFIG.use_bypass):
        make_single_dir(CONFIG.bypass_dir)

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


def register_signal():
    try:
        faulthandler.register(signal.SIGUSR1)
    except io.UnsupportedOperation:
        pass


def register_pprof(logger: KaiwuLogger):
    """
    性能优化时按照需要打开, 发布版本一定不能打开

    Args:
        logger: KaiwuLogger实例，用于记录日志
    """
    if CONFIG.enable_flamegraph:
        # 产生单独的日志句柄, 规避日志混淆问题
        flamegraph_logger = KaiwuLogger()
        current_pid = os.getpid()
        flamegraph_logger.set_logger_format(
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/learner_flamegraph_pid{current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "learner",
        )

        try:
            from common_python.pprof.server import start_flamegraph_server

            start_flamegraph_server(host=CONFIG.flamegraph_host, port=CONFIG.flamegraph_port, logger=flamegraph_logger)
            logger.info(f"进程 {current_pid} 已启动火焰图服务")
        except Exception as e:
            logger.warning(f"启动火焰图服务失败: {e}")


def register_monitor(logger: KaiwuLogger, log_prefix: str):
    """
    监控配置注册：仅在monitor_version为v2时生效，按优先级合并配置并生成到指定目录
    优先级：user_monitor > project_default_monitor > kaiwudrl_default_monitor

    Args:
        logger: KaiwuLogger实例，用于记录日志
        log_prefix: 日志前缀过滤标记
    """

    # 1. 仅v2版本生效，v1直接返回
    if CONFIG.monitor_version != KaiwuDRLDefine.MONITOR_VERSION_V2:
        running_mode = os.environ.get(
            KaiwuDRLDefine.ENV_KAIWU_RUNNING_MODE,
            KaiwuDRLDefine.RUNNING_MODE_NORMAL,
        )
        logger.info(
            add_log_prefix(
                f"Current monitor version is {CONFIG.monitor_version}, "
                f"running mode is {running_mode}, no need to generate v2 config file, skipped"
            )
        )
        return

    # 不传入logger，让MonitorConfigBuilder使用自己的logger实例（避免过滤器影响其内部日志）
    final_builder = MonitorConfigBuilder()
    output_dir = CONFIG.standard_monitor_upload_file_dir  # 输出目录
    output_file = os.path.join(output_dir, "monitor.yaml")  # 最终配置文件

    try:
        # --------------------------
        # 步骤1：加载框架默认配置（最低优先级）
        # --------------------------
        logger.info(add_log_prefix(f"Start loading framework default monitor file {CONFIG.monitor_default_file}"))
        kaiwudrl_config = load_monitor_config_from_yaml(
            yaml_file=CONFIG.monitor_default_file, logger=logger, log_prefix=log_prefix
        )
        if kaiwudrl_config:
            final_builder.merge(kaiwudrl_config)
            logger.info(add_log_prefix("Framework default monitor config merged successfully (lowest priority)"))
        else:
            logger.warning(add_log_prefix("Framework default monitor config loading failed, will skip this config"))

        # --------------------------
        # 步骤2：加载项目默认配置（中优先级，覆盖框架默认）
        # --------------------------
        logger.info(add_log_prefix(f"Start loading project default monitor file"))
        project_config = load_monitor_config_from_yaml(
            yaml_file=CONFIG.project_monitor_default_file, logger=logger, log_prefix=log_prefix
        )
        if project_config:
            final_builder.merge(project_config)  # 合并时项目配置覆盖框架默认
            logger.info(
                add_log_prefix(
                    "Project default monitor config merged successfully (medium priority, overrides framework default)"
                )
            )
        else:
            logger.warning(add_log_prefix("Project default monitor config loading failed, will skip this config"))

        # --------------------------
        # 步骤3：加载用户自定义配置（最高优先级，覆盖前两者）
        # --------------------------
        user_conf_monitor_file = f"agent_{CONFIG.algo}/conf/monitor_builder.py"
        logger.info(add_log_prefix(f"Start loading user custom monitor file: {user_conf_monitor_file}"))
        user_config = load_user_monitor_config(
            user_file_path=user_conf_monitor_file, logger=logger, log_prefix=log_prefix
        )
        if user_config:
            final_builder.merge(user_config)  # 合并时用户配置覆盖前两者
            logger.info(
                add_log_prefix(
                    "User custom monitor config merged successfully (highest priority, overrides all default configs)"
                )
            )
        else:
            logger.warning(add_log_prefix("User custom monitor config loading failed, will skip this config"))

        # --------------------------
        # 步骤4：构建最终配置并验证
        # --------------------------
        logger.info(add_log_prefix("Start building final monitor config..."))
        final_config = final_builder.build()  # 触发全量校验，确保配置合法

        # --------------------------
        # 步骤5：生成配置文件到目标目录
        # --------------------------
        # 自动创建输出目录（如/workspace/train/不存在则创建）
        os.makedirs(output_dir, exist_ok=True)
        logger.info(add_log_prefix(f"Output directory {output_dir} ready (created automatically if not exists)"))

        # 写入最终配置文件
        MonitorConfigBuilder.dump_to_yaml_file(config=final_config, file_path=output_file, logger=logger)

        # 写入最终配置文件确认文件 xxx.done 平台检测逻辑
        MonitorConfigBuilder.dump_to_yaml_file(config=final_config, file_path=output_file + ".done", logger=logger)

        logger.info(add_log_prefix("=== Monitor config registration completed ==="))
        logger.info(add_log_prefix(f"Final config file generated: {os.path.abspath(output_file)}"))
    except Exception as e:
        logger.error(add_log_prefix(f"Monitor config registration failed: {str(e)}"))  # exc_info=True打印异常栈
        return  # 监控注册返回，不阻塞主流程


def app_check_param(logger: KaiwuLogger):
    """
    下面是目前业务的正确配置项, 如果配置错误, 则强制进行修正

    Args:
        logger: KaiwuLogger实例, 用于记录日志

    Returns:
        bool: 参数检查是否通过
    """

    # learner的批处理大小需要小于等于replay_buff的capacity
    if CONFIG.train_batch_size > CONFIG.replay_buffer_capacity:
        logger.error(
            f"train_batch_size {CONFIG.train_batch_size} > replay_buffer_capacity {CONFIG.replay_buffer_capacity}"
        )
        return False

    if (
        CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY
        and CONFIG.remote_agent_default_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW
    ):
        logger.error(f"on-policy not run when remote_agent_default_runtime_mode is local_aisrv_workflow")
        return False

    return True


# 在进程启动前进行检测参数合理性
def check_param(logger: KaiwuLogger):
    """
    检查参数合理性

    Args:
        logger: KaiwuLogger实例, 用于记录日志

    Returns:
        bool: 参数检查是否通过
    """

    app_check_param_result = app_check_param(logger)
    machine_device_check_result = machine_device_check(CONFIG.svr_name)

    return app_check_param_result and machine_device_check_result


def start_learner_server(shared_mem_buffer, process_health_monitor=None):
    # 启动learner_server, 包括learner_server_reverb和learner_server_zmq, 主要用于没有reverb而使用zmq的场景里
    if CONFIG.use_learner_server:
        from kaiwudrl.server.learner.learner_server import LearnerServer

        # 人为的增加sleep时间
        time.sleep(CONFIG.start_python_daemon_sleep_after_cpp_daemon_sec)

        learner_server = LearnerServer(shared_mem_buffer)
        learner_server.start()

        # 注册到进程健康监控器
        if process_health_monitor:
            process_health_monitor.register_process(
                pid=learner_server.pid, name="learner_server", process_type="learner_server"
            )

        return learner_server

    else:
        return None


def train_loop(logger: KaiwuLogger):

    # 放置在这里, 规避配置文件没有解析时就出现异常的问题
    from kaiwudrl.server.learner.trainer import Trainer

    # 创建进程健康监控器
    process_health_monitor = ProcessHealthMonitor(
        logger=logger, check_interval=CONFIG.prometheus_stat_per_minutes, log_tag=KaiwuDRLDefine.MONITOR_INIT_LOG_FILTER
    )

    def on_process_exit(process_info: ProcessInfo):
        """子进程退出告警"""
        if process_info.exit_reason == ProcessExitReason.OOM_KILLED:
            logger.error(
                f"learner_init Learner subprocess OOM! name={process_info.name}, pid={process_info.pid}",
                g_not_server_label,
            )
        else:
            logger.warning(
                f"learner_init subprocess exited: name={process_info.name}, pid={process_info.pid}, reason={process_info.exit_reason.value}",
                g_not_server_label,
            )

    process_health_monitor.register_alert_callback(on_process_exit)

    # 对于 ZMQ 类型的 replay buffer，需要在主进程中预先创建 mem_buffer
    shared_mem_buffer = None
    if CONFIG.use_learner_server and CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
        # 在主进程中创建 MemBuffer，让所有子进程共享
        capacity = CONFIG.replay_buffer_capacity

        # 根据配置创建相应的 MemBuffer
        if CONFIG.reverb_rate_limiter == KaiwuDRLDefine.REVERB_RATE_LIMITER_SAMPLE_TO_INSERT_RATIO:
            from kaiwudrl.common.utils.mem_buffer_ratio import MemBuffer

            shared_mem_buffer = MemBuffer(
                capacity,
                logger,
                samples_per_insert=CONFIG.reverb_samples_per_insert,
                error_buffer=CONFIG.reverb_error_buffer,
            )
        else:
            from kaiwudrl.common.utils.mem_buffer import MemBuffer

            shared_mem_buffer = MemBuffer(capacity, logger)

    # 创建 Trainer，传入共享的 mem_buffer
    train = Trainer(shared_mem_buffer)

    # 启动训练进程
    train.start()

    # 注册trainer进程
    process_health_monitor.register_process(pid=train.pid, name="trainer", process_type="trainer")

    # 根据配置决定是否启动learner_server
    learner_server = None
    if CONFIG.use_learner_server:
        # 启动 learner_server，传递共享的 mem_buffer
        learner_server = start_learner_server(shared_mem_buffer, process_health_monitor)

    # 主循环：定期检查子进程健康状态
    logger.info("learner_init process health monitor started", g_not_server_label)

    while True:
        try:
            # 执行健康检查
            process_health_monitor.check_once()

            # 休眠一段时间
            time.sleep(5)

        except KeyboardInterrupt:
            logger.info("learner_init process health monitor stopped by user", g_not_server_label)
            break
        except Exception as e:
            logger.error(f"learner_init process health monitor error: {e}", g_not_server_label)


def main():
    """
    启动命令样例: python3 kaiwudrl/server/learner/learner.py --conf=kaiwudrl/conf/kaiwudrl/learner.toml
    """

    # 限制 glibc malloc arena 数量, 防止高频 malloc/free 场景下 arena 碎片化导致内存膨胀
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_LEARNER)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 使用框架定义的日志过滤标记
    INIT_LOG_FILTER = KaiwuDRLDefine.MONITOR_INIT_LOG_FILTER

    logger = KaiwuLogger()
    pid = os.getpid()
    # 使用 filter_content 参数，只记录包含 INIT_LOG_FILTER 的日志
    logger.set_logger_format(
        f"{CONFIG.log_dir}/{CONFIG.svr_name}/learner_init_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
        INIT_LOG_FILTER,
    )

    # 步骤3, 检测输入参数正确性
    if not check_param(logger):
        logger.error("conf param error, please check")

        # 写process_stop文件, 里面写error_code
        error_code = -4
        process_stop_write_file(error_code, None)
        return

    # 步骤4, 处理信号
    register_signal()

    # 步骤5, 根据配置决定是否开启火焰图服务
    register_pprof(logger)

    # 步骤6, 根据配置决定是否注册v2监控
    running_mode = os.environ.get(
        KaiwuDRLDefine.ENV_KAIWU_RUNNING_MODE,
        KaiwuDRLDefine.RUNNING_MODE_NORMAL,
    )
    logger.info(f"Running mode: {running_mode}, Monitor version: {CONFIG.monitor_version}")
    register_monitor(logger, INIT_LOG_FILTER)

    # 步骤7, 开始轮训处理
    train_loop(logger)


if __name__ == "__main__":
    sys.exit(main())
