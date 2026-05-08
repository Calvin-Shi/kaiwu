#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from multiprocessing import Value
import time
import datetime
import schedule
import os
import threading

# 按照需要导入
from common_python.config.config_control import CONFIG
from common_python.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
from kaiwudrl.common.replay_buffer.replay_buffer_wrapper import ReplayBufferWrapper
from kaiwudrl.common.checkpoint.model_file_save import ModelFileSave
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
    get_uuid,
    register_sigterm_handler,
    stop_process_by_name,
)
from common_python.ipc.zmq_util import ZmqServer, ZmqClient, ZmqConfig
from common_python.alloc.alloc_utils import AllocUtils
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.checkpoint.model_file_common import (
    clear_id_list_file,
    update_id_list,
    clear_user_ckpt_dir,
    process_stop_write_file,
)
from kaiwudrl.common.algorithms.agent_wrapper_common import (
    create_standard_agent_wrapper,
)
from kaiwudrl.server.common.load_model_common import LoadModelCommon
from kaiwudrl.server.learner.strategy import create_strategy
from common_python.worker.worker import Worker, WorkerConfig


class Trainer(Worker):
    def __init__(self, shared_mem_buffer=None):
        # 进程pid
        self.current_pid = os.getpid()

        worker_config = WorkerConfig(
            worker_name="train",
            father_pid=self.current_pid,
            use_logger=False,
            use_default_monitor=True,
            use_default_alloc=True,
        )
        super().__init__(worker_config)
        # 注意: logger的set_logger_format在before_run中调用(子进程中执行)
        # 避免在主进程中添加handler导致日志写入多个文件
        self.logger = KaiwuLogger()

        self.cached_local_step = -1

        self.local_step = Value("d", -1)

        # policy和agent_wrapper对象的map, 为了支持多agent
        self.policy_agent_wrapper_maps = {}

        # tensorflow需要传递下面的三个参数
        self.tensor_names = self.get_tensor_names()
        self.tensor_dtypes = self.get_tensor_dtypes()

        # replay_buffer
        self.replay_buffer_wrapper = ReplayBufferWrapper(self.tensor_names, self.tensor_dtypes, self.logger)

        # 通过组合引入策略
        self.strategy = create_strategy(self)

        # 由于在on-policy时, 每次清空样本, 导致样本次数变小, 故这里设置为全局累加的
        self.sample_receive_cnt = 0

        # 跨进程共享的 mem_buffer（从主进程传入）
        self.shared_mem_buffer = shared_mem_buffer

    # 设置一个属性input_datas
    def get_tensor_names(self):
        return ["input_datas"]

    def get_tensor_dtypes(self):
        dtypes = []
        if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
            dtypes.append(torch.float32)
        else:
            dtypes.append(tf.float32)

        return dtypes

    def start_learner_process_by_type(self):
        """
        根据不同的启动方式进行处理:
        1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
        2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动
        """
        if CONFIG.start_actor_learner_process_type:
            # 按照需要引入ModelFileSave
            self.model_file_saver = ModelFileSave()
            self.model_file_saver.start_actor_process_by_type(self.logger)

    # learner周期性的加载七彩石修改配置, 主要包括进程独有的和公共的
    def rainbow_activate(self):
        self.rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
        self.rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

    # learn上的训练train函数流程, 返回是否真实的训练
    def train_detail(self):
        """
        直接调用业务返回的数据格式上报, 框架不关心具体的类型和值, 格式是map
        """
        try:
            with TimeIt() as ti:
                # 传入self.current_sync_model_version_from_learner主要是用于样本过滤, 在on-policy的情况下使用到
                (
                    train_success,
                    app_monitor_data,
                    has_model_file_changed,
                    model_file_id,
                ) = self.agent_wrapper.train(self.strategy.get_current_sync_model_version_from_learner())
                if app_monitor_data and isinstance(app_monitor_data, dict):
                    self.app_monitor_data = app_monitor_data

            # 如果是错误可以提前返回
            if not train_success:
                self.logger.error(f"train learner train_detail failed")
                # 写process_stop文件, 里面写error_code
                error_code = -2
                process_stop_write_file(error_code, self.logger)
                return

            # 因为需要实时看训练情况, 故这里设置为当次的训练耗时, 而不是最大耗时
            self.batch_train_cost_time_ms = round(ti.interval * 1000, 2)
            self.data_fetch_cost_time_ms = round(self.agent_wrapper.data_fetch_cost_time * 1000, 2)
            self.real_train_cost_time_ms = round(self.agent_wrapper.real_train_cost_time * 1000, 2)

            # 如果learner训练成功即开始走on-policy的逻辑
            if has_model_file_changed:
                self.strategy.process_policy_specific(model_file_id)

        except Exception as e:
            self.logger.exception(f"train learner train_detail failed")
            # 写process_stop文件, 里面写error_code
            error_code = -1
            process_stop_write_file(error_code, self.logger)

    # learner --> actor的model文件同步, 目前采用的是model pool, 后期考虑优化, 当前的actor的local step, 同步learner上的global_step
    def model_file_sync(self):
        self.logger.debug(
            f"train process after model file sync, current global step is {self.agent_wrapper.get_global_step()}"
        )

    # 监控项置位
    def train_stat_reset(self):
        self.batch_train_cost_time_ms = 0
        self.data_fetch_cost_time_ms = 0
        self.real_train_cost_time_ms = 0
        self.sample_production_and_consumption_ratio = 0

    # 获取训练指标, 由于会多处调用故设置print_detail打印明细
    def get_training_metrics_dicts(self, print_detail=False):
        """
        样本的生成速度: reverb的间隔时间里insert的样本数,注意插入次数是一直增长的, 故需要设置2个变量才能计算出差值
        样本的消耗速度: 训练的次数 * batch_size, 注意训练的次数是一直增长的, 故需要设置2个变量才能计算出差值
        样本的消耗/生产比 = 样本消耗的速度 / 样本的生产的速度
        """
        train_count, preload_model_train_count = self.agent_wrapper.train_stat
        train_global_step = self.agent_wrapper.get_global_step()

        reverb_current_size = self.replay_buffer_wrapper.get_current_size()

        # 针对不同的方式计算方法不一样
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            self.sample_product_rate, sample_receive_cnt = self.replay_buffer_wrapper.get_insert_stats()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            reverb_insert_count = self.replay_buffer_wrapper.get_insert_stats()
            self.sample_product_rate = reverb_insert_count - self.last_reverb_insert_count
            self.last_reverb_insert_count = reverb_insert_count

            # 赋值为总的样本次数
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                self.sample_receive_cnt += reverb_insert_count
            else:
                sample_receive_cnt = reverb_insert_count
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            self.sample_product_rate, sample_receive_cnt = self.replay_buffer_wrapper.get_insert_stats()
        else:
            self.sample_product_rate = 0

            sample_receive_cnt = 0

        self.sample_consume_rate = (train_count - self.last_train_count) * int(CONFIG.train_batch_size)

        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            sample_receive_cnt = self.sample_receive_cnt

        if sample_receive_cnt == 0:
            self.sample_production_and_consumption_ratio = 0
        else:
            """
            需要区分on-policy和off-policy的场景:
            1. on-policy, 样本只是使用1次, 此时样本生产消耗比为1
            2. off-policy, 样本消耗比 = 总的训练次数 * batch_size / 总的样本接收次数

            这里先计算off-policy的, 如果是on-policy的可以自动覆盖
            """
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                self.sample_production_and_consumption_ratio = 1
            else:
                self.sample_production_and_consumption_ratio = (
                    (train_count - preload_model_train_count) * CONFIG.train_batch_size / sample_receive_cnt
                )

        self.last_train_count = train_count

        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            training_metrics = {
                KaiwuDRLDefine.MONITOR_REVERB_READY_SIZE: reverb_current_size,
                KaiwuDRLDefine.MONITOR_TRAIN_SUCCESS_CNT: train_count,
                KaiwuDRLDefine.MONITOR_TRAIN_GLOBAL_STEP: train_global_step,
                KaiwuDRLDefine.MONITOR_BATCH_TRAIN_COST_TIME_MS: self.batch_train_cost_time_ms,
                KaiwuDRLDefine.MONITOR_DATA_FETCH_COST_TIME_MS: self.data_fetch_cost_time_ms,
                KaiwuDRLDefine.MONITOR_REAL_TRAIN_COST_TIME_MS: self.real_train_cost_time_ms,
                # KaiwuDRLDefine.LEARNER_TCP_AISRV: actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.SAMPLE_PRODUCTION_AND_CONSUMPTION_RATIO: self.sample_production_and_consumption_ratio,
                KaiwuDRLDefine.SAMPLE_PRODUCT_RATE: self.sample_product_rate,
                KaiwuDRLDefine.SAMPLE_CONSUME_RATE: self.sample_consume_rate,
                KaiwuDRLDefine.SAMPLE_RECEIVE_CNT: sample_receive_cnt,
            }

            # 策略特定的监控数据
            training_metrics.update(self.strategy.train_stat())

            # 按照业务数据返回的map格式直接赋值, 然后去普罗米修斯监控上设置下展示字段即可
            for key, value in self.app_monitor_data.items():
                training_metrics[key] = float(value)

        if print_detail:
            self.logger.info(
                f"train process now input ready size is {reverb_current_size}, "
                f"train process now train count is {train_count}, global step is {train_global_step}, "
                f"train once cost time is {self.batch_train_cost_time_ms} ms "
                f"(data_fetch: {self.data_fetch_cost_time_ms} ms, real_train: {self.real_train_cost_time_ms} ms), "
                f"filter sample count is {self.agent_wrapper.filter_sample_count}, "
                f"sample_production_and_consumption_ratio is {self.sample_production_and_consumption_ratio}, "
                f"replay buffer monitor is {self.replay_buffer_wrapper.get_replay_buffer_monitor()}"
            )

        return training_metrics

    # 这里增加train的统计项
    def train_stat(self):

        # 注意，单机单进程实际是在aisrv完成训练监控上报，因此这里learner是不上报的
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            if CONFIG.rl_type == KaiwuDRLDefine.ON_LINE:
                monitor_data = self.get_training_metrics_dicts(print_detail=True)
                if int(CONFIG.use_prometheus):
                    self.monitor_proxy.put_data({self.current_pid: monitor_data})

        # 指标复原, 计算的是周期性的上报指标
        self.train_stat_reset()

    def get_replay_buffer_object(self):
        """
        主要是需要before_run下实例化了ReplayBufferWrapper后调用
        """
        return self.replay_buffer_wrapper

    def before_run(self):

        # 在子进程中初始化logger, 避免在主进程中添加handler导致日志写入多个文件
        self.current_pid = os.getpid()
        self.logger.set_logger_format(
            f"{CONFIG.log_dir}/{CONFIG.svr_name}/learner_train_pid{self.current_pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
        )
        self.logger.info("train process start at pid is {}", self.current_pid)

        # 支持间隔N分钟, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            self.rainbow_wrapper = RainbowWrapper(self.logger)

            # 第一次配置主动从七彩石拉取, 后再设置为周期性拉取
            self.rainbow_activate()
            set_schedule_event(CONFIG.rainbow_activate_per_minutes, self.rainbow_activate)

        # 先调用基类初始化
        if not super().before_run():
            return False

        try:
            # 必须放在before_run里执行, 此时reverb使用才正常
            # 如果传入了 shared_mem_buffer，则使用共享的 mem_buffer
            if self.shared_mem_buffer is not None:
                self.replay_buffer_wrapper.init_with_shared_mem_buffer(self.shared_mem_buffer)
            else:
                self.replay_buffer_wrapper.init()

            self.replay_buffer_wrapper.extra_threads()

        except Exception as e:
            error_code = -3
            self.logger.exception(f"train self.replay_buffer_wrapper start failed")
            process_stop_write_file(error_code, self.logger)

        # 根据不同启动方式来进行处理
        self.start_learner_process_by_type()

        self.process_run_count = 0

        # 获取本机IP
        self.host = get_host_ip()

        # 注册定时器任务, 因为关键日志需要打印, 故无论需要进行普罗米修斯监控否都调用下
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.train_stat)

        # policy_name 主要是和kaiwudrl/conf/app_conf.json设置一致
        self.policy_conf = AppConf.get_app_conf(CONFIG.app, "policies")

        # 需要引入业务自定义的workflow, 即while True循环
        self.workflow = None

        # agent_wrapper, 由于ModelFileSyncWrapper和ModelFileSave需要判断是否是主learner才能进行下一步处理, 故提前到这里进行
        create_standard_agent_wrapper(
            self.policy_conf,
            self.policy_agent_wrapper_maps,
            self.replay_buffer_wrapper,
            self.logger,
            self.monitor_proxy,
        )

        # 因为在learner上默认只有1个agent对象, 故设置CONFIG.policy_name所在的
        self.agent_wrapper = self.policy_agent_wrapper_maps.get(CONFIG.policy_name)
        if self.agent_wrapper.is_chief:
            # model_file_saver, 用于保存模型文件到持久化设备, 比如COS, 采用单独的进程处理, 只有主learner进程才会执行
            self.model_file_saver = ModelFileSave()
            self.model_file_saver.start()

        # 预先加载模型文件模式, 只有在train训练模式下预加载才有效
        if int(CONFIG.preload_model):
            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                self.load_model_common_object = LoadModelCommon(self.logger)
                if not self.load_model_common_object.preload_model_file(self.policy_agent_wrapper_maps):
                    self.logger.error(f"train preload_model_file failed, please check")
                    error_code = -1
                    self.learner_process_stop(error_code)
                    return False
            else:
                self.logger.warning(f"train only run_mode is {KaiwuDRLDefine.RUN_MODE_TRAIN} support preload model")

        # 启动zmq_server, 处理来自aisrv, actor的管理流, 端口设置为CONFIG.reverb_svr_port - 2, 需要考虑到zmq_server和reverb_server同时启动的情况
        self.zmq_config = ZmqConfig(
            zmq_io_threads_server=CONFIG.zmq_io_threads_server,
            zmq_io_threads_client=CONFIG.zmq_io_threads_client,
            tcp_keep_alive=CONFIG.tcp_keep_alive,
            tcp_keep_alive_idle=CONFIG.tcp_keep_alive_idle,
            tcp_keep_alive_intvl=CONFIG.tcp_keep_alive_intvl,
            tcp_keep_alive_cnt=CONFIG.tcp_keep_alive_cnt,
            sock_buff_size=CONFIG.sock_buff_size,
            backlog_size=CONFIG.backlog_size,
            tcp_immediate=CONFIG.tcp_immediate,
            zmq_ops_sendhwm=CONFIG.zmq_ops_sendhwm,
            zmq_ops_recvhwm=CONFIG.zmq_ops_recvhwm,
        )
        self.zmq_server = ZmqServer(KaiwuDRLDefine.ALL_HOST_IP, CONFIG.reverb_svr_port - 2, self.zmq_config)
        self.zmq_server.bind()
        self.logger.info(
            f"train zmq server on learner bind at "
            f"{KaiwuDRLDefine.ALL_HOST_IP}:{CONFIG.reverb_svr_port - 2} for aisrv"
        )

        # 如果是pytorch, 则默认第一次保存文件
        if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH:
            # 清空id_list文件, 否则文件会持续增长
            clear_id_list_file(framework=True)

            # 第一次保存模型时id的默认值即0
            self.agent_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK)

            # 更新id_list文件
            if int(CONFIG.preload_model):
                update_id_list(CONFIG.preload_model_id, framework=True)
            else:
                update_id_list(0, framework=True)

            # 清空使用者保存的文件目录
            clear_user_ckpt_dir()

        """
        统计监控指标
        1. 批处理的训练耗时
        """
        self.batch_train_cost_time_ms = 0
        self.data_fetch_cost_time_ms = 0
        self.real_train_cost_time_ms = 0
        self.sample_production_and_consumption_ratio = 0
        self.last_reverb_insert_count = 0
        self.last_train_count = 0
        self.sample_product_rate = 0
        self.sample_consume_rate = 0

        # 业务算法类监控值是个map形式
        self.app_monitor_data = {}

        # 因为需要从learner获取aisrv地址
        self.alloc_util = AllocUtils(self.logger)
        # 该set_name下的aisrv地址个数
        self.aisrv_process_count = 0

        # 针对来自aisrv的管理流启动单个线程处理
        t = threading.Thread(target=self.learner_process_message_by_aisrv)
        t.daemon = True
        t.start()

        # 注册SIGTERM信号处理
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            register_sigterm_handler(self.handle_sigterm, CONFIG.sigterm_pids_file)

        if self.agent_wrapper.is_chief:
            # 策略特定的初始化
            self.strategy.before_run()

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"train process start success at {self.current_pid}, on-policy/off-policy is {self.strategy.strategy_name()}, "
            f"trainer global step {self.local_step.value}, "
            f"app {CONFIG.app} algo {CONFIG.algo}, train_batch_size is {CONFIG.train_batch_size}"
        )

        return True

    def learner_get_aisrv_address(self):
        """
        learner获取aisrv地址, 只是获取到地址, 不会进行建立连接
        """

        """
        1. 如果不使用alloc服务, 则直接使用本地配置, 本地配置为空则使用127.0.0.1
        2. 如果使用alloc服务, 则直接使用alloc服务
        """

        if int(CONFIG.use_alloc):
            self.alloc_util.registry()
            # on-policy情况下learner需要启动与aisrv的通信, 采用在aisrv 8000端口号 + 100的端口上监听, learner为client, aisrv为server
            aisrv_address = self.alloc_util.get_all_address_by_srv_name(KaiwuDRLDefine.SERVER_AISRV)
            if not aisrv_address:
                self.logger.error(f"train get aisrv_address error, retry next time")
                self.aisrv_process_count = 0
                return None
            else:
                self.logger.info(f"train get alloc aisrv_address success, aisrv address: {aisrv_address}")
        else:
            aisrv_default_address = CONFIG.aisrv_default_address
            if aisrv_default_address:
                original_aisrv_address = aisrv_default_address.split(",")
            else:
                original_aisrv_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.aisrv_server_port}"]

            # 注意需要检测配置项
            aisrv_address = [item for item in original_aisrv_address if item]

            self.logger.info(f"train get default aisrv_address success, aisrv address: {aisrv_address}")

        # aisrv是分多个对局的, 每个aisrv * 对局数目
        self.aisrv_process_count = len(aisrv_address) * CONFIG.aisrv_connect_to_kaiwu_env_count

        return aisrv_address

    def save_model_detail(self, ip, path, id):
        """
        处理来自aisrv的save_model请求详细执行过程
        """
        path = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}/"

        # 注意此时是业务调用的
        try:
            self.agent_wrapper.save_param_by_source(
                path=path,
                id=id,
                source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER,
            )
            self.logger.info(f"train learner save_param_by_framework is success, ip is {ip}")
            return True

        except Exception as e:
            self.logger.error(f"train learner save_param_by_framework is failed, ip is {ip}")

            return False

    def learner_process_message_by_aisrv(self):
        """
        收集来自aisrv的zmq请求, 分为下面情况:
        1. 如果是save_model, 则执行save_model操作

        下面是规则:
        1. 针对不同的aisrv来让同一个learner执行save_model的操作
        1.1 在时间间隔内如果获取到第一个即执行
        1.2 在最大时间间隔内如果1.1的aisrv持续的进行save_model时则直接执行, 否则转1.3
        1.3 重新接收第一个需要执行的aisrv, 然后更新时间
        """

        # 用于处理来自aisrv的save_model请求的限制
        last_save_model_aisrv_ip = None
        last_save_model_time = 0

        # 当前接收到aisrv需要退出的进程数量
        current_aisrv_process_stop_count = 0

        # 标志是否有aisrv发送过process_stop的请求, 然后周期性判断超时退出
        had_recv_aisrv_stop_request = False
        last_recv_aisrv_stop_request_time = 0
        had_learner_process_stop = False

        while True:
            # 下面是learner超时退出逻辑
            if had_recv_aisrv_stop_request and had_learner_process_stop:
                now = time.time()
                if now - last_recv_aisrv_stop_request_time > CONFIG.aisrv_process_stop_timeout_seconds:

                    # 达到超时条件
                    error_code = KaiwuDRLDefine.DOCKER_EXIT_CODE_TIMEOUT

                    # 进程退出
                    self.logger.info(
                        f"train learner recv aisrv "
                        f"now {now} - "
                        f"last_recv_aisrv_stop_request_time {last_recv_aisrv_stop_request_time} "
                        f">= {CONFIG.aisrv_process_stop_timeout_seconds}, so exit"
                    )
                    self.learner_process_stop(error_code)

                    # 更新时间, 免得满足超时条件后就进入到一直退出的状态
                    last_recv_aisrv_stop_request_time = now

            try:
                # 收到来自aisrv的请求
                client_id, message = self.zmq_server.recv(block=False, binary=False)
                if message:
                    message_type = message.get(KaiwuDRLDefine.MESSAGE_TYPE)
                    message_value = message.get(KaiwuDRLDefine.MESSAGE_VALUE)
                    if message_type == KaiwuDRLDefine.MESSAGE_SAVE_MODEL:
                        ip = message_value.get("ip")
                        path = message_value.get("path")
                        id = message_value.get("id")
                        # self.logger.info(f"train learner recv save_model from aisrv {ip}")

                        send_data = {
                            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_SAVE_MODEL,
                            KaiwuDRLDefine.MESSAGE_VALUE: True,
                        }
                        # 需要先执行zmq回包, 再进行处理
                        self.zmq_server.send(str(client_id), send_data, binary=False)
                        # self.logger.info(f"train learner send save_model result to aisrv {ip}")

                        now = time.time()

                        # 如果没有上一次的保存模型请求，或者上一次的请求来自同一个ip，执行保存模型
                        if not last_save_model_aisrv_ip or last_save_model_aisrv_ip == ip:
                            # 然后执行保存用户model文件的操作
                            if self.save_model_detail(ip, path, id):
                                last_save_model_aisrv_ip = ip
                                last_save_model_time = now
                                self.logger.info(f"train learner really save_model from aisrv {ip}")

                        # 如果上一次的请求来自不同的ip，但已经超过了最大等待时间，也执行保存模型
                        else:
                            if (
                                now - last_save_model_time
                                >= CONFIG.choose_aisrv_to_load_model_or_save_model_max_time_seconds
                            ):
                                # 然后执行保存用户model文件的操作
                                if self.save_model_detail(ip, path, id):
                                    last_save_model_aisrv_ip = ip
                                    last_save_model_time = now
                                    self.logger.info(f"train learner really save_model from aisrv {ip}")

                    elif message_type == KaiwuDRLDefine.MESSAGE_PROCESS_STOP:
                        ip = message_value.get("ip")
                        error_code = message_value.get("error_code")
                        self.logger.info(f"train learner recv process_stop from aisrv {ip}, error_code {error_code}")
                        send_data = {
                            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_PROCESS_STOP,
                            KaiwuDRLDefine.MESSAGE_VALUE: True,
                        }

                        # 需要先执行zmq回包, 再进行处理
                        self.zmq_server.send(str(client_id), send_data, binary=False)
                        self.logger.info(f"train learner send process_stop result to aisrv {ip}")

                        had_recv_aisrv_stop_request = True
                        last_recv_aisrv_stop_request_time = time.time()

                        # 因为此时能发出来process_stop请求的aisrv是已经启动了的, 故这里重新拉取下地址即得到此时活着的aisrv进程
                        self.learner_get_aisrv_address()

                        """
                        error_code场景:
                        1. 如果是错误退出, 则只要有1个aisrv上报该错误码即需要learner退出
                        2. 如果是正常退出, 则需要按照比例来退出aisrv_process_stop_quantity_ratio, 默认是100%
                        """
                        if error_code > 0:
                            # 进程退出
                            self.logger.info(f"train learner recv error_code {error_code} from {ip}, so exit")
                            self.learner_process_stop(error_code)

                            had_learner_process_stop = True
                        else:
                            current_aisrv_process_stop_count += 1

                            if self.aisrv_process_count == 0:
                                self.logger.info(f"train learner self.aisrv_process_count is 0, so exit")
                                self.learner_process_stop(error_code)

                                had_learner_process_stop = True

                            # 达到比例退出
                            else:
                                if (
                                    current_aisrv_process_stop_count / self.aisrv_process_count
                                    >= CONFIG.aisrv_process_stop_quantity_ratio
                                ):
                                    # 进程退出
                                    self.logger.info(
                                        f"train learner recv aisrv "
                                        f"{current_aisrv_process_stop_count} / {self.aisrv_process_count} "
                                        f">= {CONFIG.aisrv_process_stop_quantity_ratio}, so exit"
                                    )
                                    self.learner_process_stop(error_code)

                                    had_learner_process_stop = True

                    elif message_type == KaiwuDRLDefine.MESSAGE_GET_TRAINING_METRICS:
                        ip = message_value.get("ip")
                        # self.logger.info(f"train learner recv get_training_metrics from aisrv {ip}")

                        send_train_data_detail = self.get_training_metrics_dicts()
                        send_data = {
                            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.MESSAGE_GET_TRAINING_METRICS,
                            KaiwuDRLDefine.MESSAGE_VALUE: send_train_data_detail,
                        }

                        # 需要先执行zmq回包, 再进行处理
                        self.zmq_server.send(str(client_id), send_data, binary=False)
                        # self.logger.info(f"train learner send get_training_metrics result to aisrv {ip}")
                    else:
                        self.logger.error(f"train learner recv unknown message_type {message_type}")

            except Exception as e:
                # sleep下减少CPU损耗
                time.sleep(CONFIG.idle_sleep_second)

    def learner_process_stop(self, error_code):
        """
        learner进程的退出, 包括自己的python3进程, modelpool进程, 但是如果此时就直接退出其他进程如modelpool进程可能导致有其他进程上报model文件失败
        故统一调整到所有的进程是在被动退出时退出
        """

        # 写process_stop文件, 里面写error_code
        process_stop_write_file(error_code, self.logger)

    def train(self):
        """
        训练的规则:
        1. 在线强化学习
        1.1. 当reverb设置最大的size, 采用FIFO模式
        1.2 当满足batch_size即开始训练, 对reverb不做主动清空操作, 从reverb里拿取的数据是随机的, 这样增加了训练次数, 新的数据进来采用FIFO去替换掉旧的
        2. 离线强化学习
        2.1 直接从replay_buffer读取数据训练
        """

        # 标志本次是否真实的train
        is_train_success = False
        if CONFIG.rl_type == KaiwuDRLDefine.ON_LINE:
            if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
                current_size = self.replay_buffer_wrapper._replay_buffer.total_size(
                    self.replay_buffer_wrapper._reverb_client
                )
            elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
                current_size = self.replay_buffer_wrapper._replay_buffer.total_size()
            elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
                current_size = self.replay_buffer_wrapper._replay_buffer.total_size()
            else:
                current_size = 0

            # 步骤1, 判断能否开始训练
            condition = self.strategy.train_condition(current_size)
            if condition:
                # 步骤2, 训练
                self.train_detail()

                is_train_success = True
        else:
            current_size = self.replay_buffer_wrapper._replay_buffer.total_size()
            condition = current_size >= int(CONFIG.train_batch_size)
            if condition:
                self.train_detail()
                is_train_success = True

        return is_train_success

    def periodic_operation(self):
        # on-policy情况下, learner需要知道aisrv/actor地址
        self.strategy.periodic_operation()

    def run_once(self):
        """
        learner的单次流程如下:
        1. 执行定时器操作
        2. 执行训练步骤
        3. on-policy情况下, 执行从aisrv开始的流程
        """

        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()
        self.periodic_operation()

        """
        步骤2, 执行训练, 主要是下面的情况:
        1. 如果是on-policy
        1.1 leaner上直接进行训练
        1.2 其他的情况, 后期扩展, 直接训练
        2. 如果是off-policy, 直接训练
        3. 其他的情况, 后期扩展, 直接训练
        """
        self.train()

        # Model文件保存, 同步已经采用单个进程方式进行

    def after_run(self) -> bool:
        pass

    def run(self):
        if not self.before_run():
            self.logger.info("train before_run failed, so return")
            return

        while not self.agent_wrapper.should_stop():
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(f"train process failed to run trainer. exit. " f"Error is: {e}")
                break

        self.agent_wrapper.close()
        self.logger.info("train self.server.stop success")

        # 策略特定的清理操作
        self.strategy.cleanup()

    def handle_sigterm(self, sig, frame):
        # 已经创建agent_wrapper,并且为主learner进程
        if hasattr(self, "agent_wrapper") and self.agent_wrapper.is_chief:
            self.logger.info(f"trainer {self.current_pid} is starting to handle the SIGTERM signal.")
            self.agent_wrapper.save_param_by_source(source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_BY_SIGTERM)
            # 处理完保存最新模型,等待其他进程工作,避免pod提前退出
            time.sleep(CONFIG.handle_sigterm_sleep_seconds)
        else:
            self.logger.info(f"trainer {self.current_pid} is not chief.")
            time.sleep(CONFIG.handle_sigterm_sleep_seconds)
