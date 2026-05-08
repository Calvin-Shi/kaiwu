#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import json
import multiprocessing
import datetime
import os
import schedule
import time
import copy
import collections
from common_python.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from common_python.alloc.alloc_proxy import AllocProxy
from common_python.alloc.alloc_utils import AllocUtils
from kaiwudrl.common.utils.common_func import (
    is_list_eq,
    list_diff,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
    Context,
    python_exec_shell,
    register_sigterm_handler,
)
from kaiwudrl.common.utils.slots import Slots
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper
from common_python.worker.worker import Worker, WorkerConfig
from kaiwudrl.server.aisrv.aisrv_handle_standard import AiSrvHandle
from common_python.monitor.process_health_monitor import ProcessHealthMonitor, ProcessInfo, ProcessExitReason


class AiServer(Worker):
    def __init__(
        self,
    ) -> None:
        # 进程pid
        self.current_pid = os.getpid()
        worker_config = WorkerConfig(
            worker_name="ai_server",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=True,
        )
        super().__init__(worker_config)

        # 获取本机IP
        self.host = get_host_ip()

        # 初始化进程健康监控器
        self.process_health_monitor = None

    # 业务的上报, aisrv主线程
    def aisrv_main_process_stat(self):
        pass

        """
        下面的监控值上报暂时取消掉
        """

        """
        if int(CONFIG.use_prometheus):
            monitor_data = {}

            # 无论什么场景下都需要上报的监控值
            monitor_data[KaiwuDRLDefine.AISRV_TCP_BATTLESRV] = actor_learner_aisrv_count(self.host, CONFIG.svr_name)

            self.monitor_proxy.put_data({self.current_pid: monitor_data})
        """

    def check_param(self):
        """
        进程启动前配置参数检测
        1. 规则1, 如果是设置了self-play模式, 但是app文件里设置的policy是1个, 则报错
        2. 规则2, 如果是设置了非self-play模式, 但是app文件里设置的policy是2个, 则报错
        3. 规则3, 如果是设置了self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是1个, 则报错
        4. 规则4, 如果是设置了非self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是2个, 则报错
        5. 规则5, 如果是设置了on-policy模式, 但是同时是小规模模式, 则报错
        """

        actor_addrs = CONFIG.actor_addrs
        learner_addrs = CONFIG.learner_addrs
        policies = AppConf.get_app_conf(CONFIG.app, "policies")

        if int(CONFIG.self_play):
            if len(policies) == 1:
                self.logger.error(f"ai_server self-play模式, 但是配置的policy维度为1, 请修改配置后重启进程")
                return False

            if len(actor_addrs) == 1 or len(learner_addrs) == 1:
                self.logger.error(
                    f"ai_server self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为1, 请修改配置后重启进程"
                )
                return False

        else:
            if len(policies) == 2:
                self.logger.error(f"ai_server 非self-play模式, 但是配置的policy维度为2, 请修改配置后重启进程")
                return False

            if len(actor_addrs) == 2 or len(learner_addrs) == 2:
                self.logger.error(
                    f"ai_server 非self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为2, 请修改配置后重启进程"
                )
                return False

        return True

    # aisrv在处理actor和learner的动态扩缩容逻辑
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return

    # aisrv从alloc服务获取kaiwu_env的IP地址
    def get_kaiwu_env_ip_from_alloc(self):
        # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
        retry_num = 0
        kaiwu_env_address = []
        while retry_num < CONFIG.socket_retry_times:
            kaiwu_env_address = self.alloc_util.get_kaiwu_env_ip(CONFIG.set_name, CONFIG.self_play_set_name)
            if not kaiwu_env_address:
                time.sleep(int(CONFIG.socket_timeout))
                retry_num += 1
            else:
                break

        # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
        if retry_num >= CONFIG.socket_retry_times:
            self.logger.error(
                f"ai_server server get kaiwu_env address retry times more than "
                f"{CONFIG.socket_retry_times}, will start with configure file"
            )
            return None

        return kaiwu_env_address

    # aisrv朝alloc服务的注册函数, 需要先注册才能拉取地址
    def aisrv_registry_to_alloc(self):
        # 需要先注册本地aisrv地址后, 再拉取actor, learner地址
        code, msg = self.alloc_util.registry()
        if code:
            self.logger.info(f"ai_server alloc interact registry success")
            return True
        else:
            self.logger.error(f"ai_server alloc interact registry fail, will retry next time, error_code is {msg}")
            return False

    def get_actor_learner_ip_from_alloc(self):
        """
        增加aisrv从alloc获取IP地址的逻辑, 为了和以前从配置文件加载的方式结合, 采用操作步骤如下:
        1. 每隔CONFIG.alloc_process_per_seconds拉取, 最大CONFIG.socket_retry_times次后报错, 当返回有具体的数据则跳出循环
        2. 针对返回的actor和learner地址, 修改内存和配置文件里的值
        """

        if CONFIG.remote_agent_default_runtime_mode == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_ACTOR_PREDICT:
            # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
            retry_num = 0
            while retry_num < CONFIG.socket_retry_times:
                if not int(CONFIG.self_play):
                    (
                        actor_address,
                        learner_address,
                        _,
                        _,
                    ) = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, None)
                    if not actor_address or not learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break
                else:
                    # 对于self-play模式, self_play_set下的learner不是强要求的
                    (
                        self_play_actor_address,
                        self_play_old_actor_address,
                        self_play_learner_address,
                        self_play_old_learner_address,
                    ) = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)

                    if not self_play_actor_address or not self_play_learner_address or not self_play_old_actor_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break

            # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
            if retry_num >= CONFIG.socket_retry_times:
                self.logger.error(
                    f"ai_server server get actor and learner address retry times more than "
                    f"{CONFIG.socket_retry_times}, will start with configure file"
                )
                return

            # 修改配置文件
            if not int(CONFIG.self_play):
                self.change_configure_content(actor_address, learner_address, None, None, None, None)
            else:
                self.change_configure_content(
                    None,
                    None,
                    self_play_actor_address,
                    self_play_learner_address,
                    self_play_old_actor_address,
                    self_play_old_learner_address,
                )
        else:
            # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取learner地址
            retry_num = 0
            while retry_num < CONFIG.socket_retry_times:
                if not int(CONFIG.self_play):
                    learner_address, _ = self.alloc_util.get_learner_ip(CONFIG.set_name, None)
                    if not learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break
                else:
                    # 对于self-play模式, self_play_set下的learner不是强要求的
                    (
                        self_play_learner_address,
                        self_play_old_learner_address,
                    ) = self.alloc_util.get_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)
                    if not self_play_learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break

            # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
            if retry_num >= CONFIG.socket_retry_times:
                self.logger.error(
                    f"ai_server server get actor and learner address retry times more than "
                    f"{CONFIG.socket_retry_times}, will start with configure file"
                )
                return

            # 修改配置文件
            if not int(CONFIG.self_play):
                # 此处需要针对设置值
                actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self.change_configure_content(actor_address, learner_address, None, None, None, None)
            else:
                self_play_actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self_play_old_actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self.change_configure_content(
                    None,
                    None,
                    self_play_actor_address,
                    self_play_learner_address,
                    self_play_old_actor_address,
                    self_play_old_learner_address,
                )

    # C++ 常驻进程进程配置文件修改
    def save_to_file(self, process_name, to_change_key_values):
        if not to_change_key_values or not process_name:
            return

        # 先删除actor_addrs,learner_addrs,self_play, actor_proxy_num, learner_proxy_num
        cmd = (
            f"sed -i '/actor_addrs/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/learner_addrs/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/self_play/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/actor_proxy_num/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/learner_proxy_num/d' {CONFIG.cpp_aisrv_configure};"
        )
        result_code, result_str = python_exec_shell(cmd)
        if result_code:
            self.logger.error(f"ai_server python_exec_shell failed, cmd is {cmd}, error msg is {result_str}")
            return

        # 由于self_play是在main里配置, 这里根据返回的actor_addrs和learner_addrs来决定其值
        actor_addrs_json = json.loads(to_change_key_values.get("actor_addrs"), strict=False)
        self_play = 0
        if len(actor_addrs_json) == 2:
            self_play = 1
        to_change_key_values["self_play"] = self_play

        # 去掉actor_proxy_num和learner_proxy_num参数
        del to_change_key_values["actor_proxy_num"]
        del to_change_key_values["learner_proxy_num"]

        # 追加文件写操作
        with open(CONFIG.cpp_aisrv_configure, "a", encoding=KaiwuDRLDefine.UTF_8) as f:
            for key, value in to_change_key_values.items():
                # gflags严格要求key=value形式, 不能留空格
                f.write(f"--{key}={value}\n")
                self.logger.info(f"AiServer {CONFIG.cpp_aisrv_configure} {key} {value}")

        self.logger.info(f"ai_server {CONFIG.cpp_aisrv_configure} CONFIG save_to_file success")

    def change_configure_content(
        self,
        actor_addrs,
        learner_addrs,
        self_play_actor_address,
        self_play_learner_address,
        self_play_old_actor_address,
        self_play_old_learner_address,
    ):
        """
        修改conf/system/aisrv_system.toml里的配置项目, 如下:
        1. actor_addrs
        2. actor_proxy_num
        3. learner_addrs
        4. learner_proxy_num
        5. self_play_actor_proxy_num
        6. self_play_old_actor_proxy_num
        7. self_play_learner_proxy_num
        8. self_play_old_learner_proxy_num
        """

        # 写回配置文件内容
        to_change_key_values = {}

        # 将当前的配置文件的内容读成json串, 内存修改后, 再写回json内容, 如果解析json串出错, 则提前报错返回
        try:
            old_actor_address_map = copy.deepcopy(CONFIG.actor_addrs)
            old_learner_address_map = copy.deepcopy(CONFIG.learner_addrs)

            # 如果是非self-play, 需要删除掉CONFIG.self_play_old_policy对应的数据
            if not int(CONFIG.self_play):
                if CONFIG.self_play_old_policy in old_actor_address_map:
                    del old_actor_address_map[CONFIG.self_play_old_policy]
                if CONFIG.self_play_old_policy in old_learner_address_map:
                    del old_learner_address_map[CONFIG.self_play_old_policy]

        except Exception as e:
            self.logger.error(
                f"ai_server get actor and learner address from conf failed, error is {str(e)}",
                g_not_server_label,
            )

            return

        """
        处理实例如下:
        actor_addrs = {"train_one": ["127.0.0.1:8001"], "train_two": ["127.0.0.1:8002"]}
        learner_addrs = {"train_one": ["127.0.0.1:9000"], "train_two": ["127.0.0.1:9001"]}
        """

        if not int(CONFIG.self_play):
            if not actor_addrs and not learner_addrs:
                return

            # 如果actor_addrs不空则处理, 否则跳过
            if actor_addrs:
                actor_proxy_num = len(actor_addrs)
                old_actor_address_map[CONFIG.policy_name] = actor_addrs
                to_change_key_values["actor_proxy_num"] = actor_proxy_num
                to_change_key_values["actor_addrs"] = old_actor_address_map

            # 如果learner_addrs不空则处理, 否则跳过
            if learner_addrs:
                learner_proxy_num = len(learner_addrs)
                old_learner_address_map[CONFIG.policy_name] = learner_addrs
                to_change_key_values["learner_proxy_num"] = learner_proxy_num
                to_change_key_values["learner_addrs"] = old_learner_address_map

            # 修改配置文件内容落地
            if actor_addrs or learner_addrs:
                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"ai_server {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

        else:
            if not self_play_actor_address and not self_play_learner_address and not self_play_old_actor_address:
                return

            if self_play_actor_address:
                self_play_actor_proxy_num = len(self_play_actor_address)
                old_actor_address_map[CONFIG.self_play_policy] = self_play_actor_address
                to_change_key_values["self_play_actor_proxy_num"] = self_play_actor_proxy_num

            if self_play_old_actor_address:
                self_play_old_actor_proxy_num = len(self_play_old_actor_address)
                old_actor_address_map[CONFIG.self_play_old_policy] = self_play_old_actor_address
                to_change_key_values["self_play_old_actor_proxy_num"] = self_play_old_actor_proxy_num

            to_change_key_values["actor_addrs"] = old_actor_address_map

            if self_play_learner_address:
                self_play_learner_proxy_num = len(self_play_learner_address)
                CONFIG.self_play_learner_proxy_num = self_play_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_policy] = self_play_learner_address
                to_change_key_values["self_play_learner_proxy_num"] = self_play_learner_proxy_num

            if self_play_old_learner_address:
                self_play_old_learner_proxy_num = len(self_play_old_learner_address)
                CONFIG.self_play_old_learner_proxy_num = self_play_old_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_old_policy] = self_play_old_learner_address
                to_change_key_values["self_play_old_learner_proxy_num"] = self_play_old_learner_proxy_num

            to_change_key_values["learner_addrs"] = old_learner_address_map

            # 修改配置文件内容落地
            if (
                self_play_actor_address
                or self_play_learner_address
                or self_play_old_actor_address
                or self_play_old_learner_address
            ):
                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"ai_server {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"ai_server before_run failed, so return")
            return

        while True:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    # 因为该run函数内只有周期性的操作和on-policy的操作, 故增加sleep时间减少CPU占用
                    time.sleep(CONFIG.idle_sleep_second * 1000)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(f"ai_server failed to run {self.name} . exit. Error is: {e}, ")

    def run_once(self) -> None:
        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, 进程健康监控
        if self.process_health_monitor:
            self.process_health_monitor.check_once()

    def after_run(self) -> bool:
        pass

    def on_process_exit_alert(self, process_info: ProcessInfo):
        """
        进程退出告警回调函数

        当监控到子进程异常退出(特别是OOM)时触发
        """
        if process_info.exit_reason == ProcessExitReason.OOM_KILLED:
            self.logger.error(
                f"ai_server ⚠️  ALERT: Process OOM detected! "
                f"name={process_info.name}, "
                f"pid={process_info.pid}, "
                f"type={process_info.process_type}",
            )
        else:
            self.logger.warning(
                f"ai_server Process exited abnormally: "
                f"name={process_info.name}, "
                f"pid={process_info.pid}, "
                f"type={process_info.process_type}, "
                f"reason={process_info.exit_reason.value}",
            )

    def start_aisrv_handler(self):
        """
        下面是启动逻辑:
        1. 如果是kaiwu_env_proxy或者kaiwu_env, 则
            aisrv在启动时, 从alloc进程获取actor和learner的分配IP地址; 如果不从alloc访问默认是采用配置项, 配置项里会设置具体的地址
                在单个容器里是127.0.0.1
                在多个容器里是需要用户自己设置
            1. actor, learner的分配IP地址
            2. kaiwu_env的分配IP地址
        2. 如果是issac, 默认是单机单进程的, 即数量为1
        3. 如果是direct, 则不能以env地址来区分, 因为此时没有env地址, 而是按照aisrv_connect_to_kaiwu_env_count地址来分配
        """
        if CONFIG.aisrv_framework in [
            KaiwuDRLDefine.AISRV_FRAMEWORK_ENV_TYPE_KAIWU_ENV_PROXY,
            KaiwuDRLDefine.AISRV_FRAMEWORK_ENV_TYPE_ISSAC,
            KaiwuDRLDefine.AISRV_FRAMEWORK_ENV_TYPE_KAIWU_ENV,
        ]:
            kaiwu_env_default_address = CONFIG.kaiwu_env_default_address
            if kaiwu_env_default_address:
                kaiwu_env_address = kaiwu_env_default_address.split(",")
            else:
                kaiwu_env_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.kaiwu_env_svr_port}"]
            if int(CONFIG.use_alloc):
                registry_result = self.aisrv_registry_to_alloc()
                if not registry_result:
                    self.logger.error(f"ai_server aisrv_registry_to_alloc failed")
                else:
                    # 如果需要从alloc服务获取kaiwu_env地址则获取
                    if CONFIG.get_kaiwu_env_by_alloc:
                        kaiwu_env_address = self.get_kaiwu_env_ip_from_alloc()

            if not kaiwu_env_address:
                self.logger.error(f"ai_server fail to get kaiwu_env address")
                return
        else:
            # 此时复用了kaiwu_env_address数据结构
            kaiwu_env_address = [
                f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.kaiwu_env_svr_port}"
            ] * CONFIG.aisrv_connect_to_kaiwu_env_count

        self.logger.info(f"ai_server get kaiwu_env address is {kaiwu_env_address}")
        # 如果是在aisrv处理多agent时采用并行方式则需要启用进程池
        if CONFIG.multi_agent_predict == KaiwuDRLDefine.MULTI_AGENT_PREDICT_PARALLEL:
            num_processes = os.cpu_count()
            self.simu_ctx.pool = multiprocessing.Pool(processes=num_processes)

        # 针对返回来的kaiwu_env的IP列表, 每个IP启动单个进程
        valid_index = 0
        for index, address in enumerate(kaiwu_env_address):
            if address and len(address) > 0:
                self.simu_ctx.index = valid_index
                handler = AiSrvHandle(address, self.simu_ctx)
                handler.start()
                self.logger.info(f"ai_server AiSrvHandle with address: {address} start")
                valid_index += 1

                # 注册workflow进程到健康监控器
                self.process_health_monitor.register_process(
                    pid=handler.pid, name=f"workflow_{valid_index-1}", process_type="workflow"
                )

    def before_run(self):
        # 设置Context
        self.simu_ctx = Context()

        # aisrv进程启动时, 从七彩石获取配置, 然后将该七彩石的操作句柄传给对应的子进程
        if int(CONFIG.use_rainbow):
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            rainbow_wrapper = RainbowWrapper(self.logger)
            # 在本次对局开始前, aisrv看下参数修改情况
            rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
            rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)
            self.simu_ctx.rainbow_wrapper = rainbow_wrapper

        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        # 初始化进程健康监控器(提前初始化,以便后续子进程可以使用)
        self.process_health_monitor = ProcessHealthMonitor(
            logger=self.logger, check_interval=CONFIG.prometheus_stat_per_minutes, log_tag="ai_server"
        )
        # 注册告警回调
        self.process_health_monitor.register_alert_callback(self.on_process_exit_alert)
        # 将监控器传递到context,供子进程注册使用
        self.simu_ctx.process_health_monitor = self.process_health_monitor

        # aisrv handler进程使用
        self.simu_ctx.slots = Slots(int(CONFIG.max_tcp_count), int(CONFIG.max_queue_len))

        # aisrv启动时获取actor和learner地址
        if int(CONFIG.use_alloc):
            # alloc 工具类, aisrv上与alloc交互操作
            self.alloc_util = AllocUtils(self.logger)
            self.aisrv_registry_to_alloc()
            if CONFIG.need_to_start_learner:
                self.get_actor_learner_ip_from_alloc()

        # 无论从七彩石或者其他地方配置完成的配置文件后再开始检测配置文件的有效性
        if not self.check_param():
            self.logger.error(f"ai_server check_param failed, so return")
            return False

        """
        如果在小规模场景下, 因为model_file_sync进程只需要启动1个, 而predictor_local进程是多个的, 故这里需要采用下面步骤:
        1. 如果需要启动learner进程, 即训练进程/预测进程之间需要传递model文件, 则:
        1.1 model_file_sync进程先启动
        1.2 将model_file_sync进程的对象句柄传入到predictor_local进程里进行使用
        又由于会对不同的policy进行AsyncBuilder, 故只有将对model_file_sync的进程启动放在AsyncBuilder调用之前进行
        2. 如果不需要启动learner进程, 即训练进程/预测进程之间不需要传递model文件, 则:
        2.1 不需要启动model_file_sync进程
        2.2 如果是单机单进程的不需要启动model_file_sync进程
        """
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
            if CONFIG.remote_agent_default_runtime_mode in [
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT,
                KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW,
            ]:
                if CONFIG.need_to_start_learner:
                    model_file_sync_wrapper = ModelFileSyncWrapper()
                    model_file_sync_wrapper.init()

                    # 因为在on-policy的情况下存在多个预测进程竞争情况故加上锁的操作
                    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                        lock = multiprocessing.Lock()
                        self.simu_ctx.model_file_sync_wrapper_lock = lock

                    self.simu_ctx.model_file_sync_wrapper = model_file_sync_wrapper

        """
        实例配置如下
        {
            "hero": {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "rl_helper": "app.gorge_walk.environment.gorge_walk_rl_helper.GorgeWalkRLHelper",
                "policies": {
                "train_one": {
                    "policy_builder" : "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                    "algo": "ppo",
                    "state": "app.gym.gym_proto.GymState",
                    "action": "app.gym.gym_proto.GymAction",
                    "reward": "app.gym.gym_proto.GymReward",
                    "actor_network": "app.gym.gym_network.GymDeepNetwork",
                    "learner_network": "app.gym.gym_network.GymDeepNetwork",
                    "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    }
                }
            }
        }
        """
        # 配置相关的传递
        try:
            policies_builder = {}
            policies_conf = AppConf.get_app_conf(CONFIG.app, "policies")
            for policy_name, policy_conf in policies_conf.items():
                policies_builder[policy_name] = policy_conf.policy_builder(policy_name, self.simu_ctx)

            self.simu_ctx.policies_builder = policies_builder

            self.simu_ctx.kaiwu_rl_helper = AppConf.get_app_conf(CONFIG.app, "rl_helper")

        except Exception as e:
            self.logger.exception(f"ai_server server start exception: {str(e)}")
            return False

        """
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        """
        if CONFIG.actor_learner_expansion:
            set_schedule_event(
                int(CONFIG.alloc_process_per_seconds),
                self.aisrv_with_new_actor_learner_change,
            )

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
            set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.aisrv_main_process_stat)

        self.process_run_count = 0

        # 启动aisrv_handler进程
        self.start_aisrv_handler()

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"ai_server is start success at {CONFIG.aisrv_ip_address}:{CONFIG.aisrv_server_port}, "
            f"pid is {self.current_pid}, run_mode is {CONFIG.run_mode}, "
            f"self_play is {CONFIG.self_play}"
        )
        return True
