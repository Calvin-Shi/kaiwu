#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
from kaiwudrl.server.learner.strategy import ITrainerStrategy
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.checkpoint.model_file_sync import ModelFileSync
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
from common_python.ipc.zmq_util import ZmqServer, ZmqConfig, ZmqClient
from common_python.alloc.alloc_utils import AllocUtils
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
    get_uuid,
    register_sigterm_handler,
    stop_process_by_name,
)


class OnPolicyStrategy(ITrainerStrategy):
    """
    OnPolicy的策略实现
    """

    def __init__(self, trainer):
        # 注意因为aisrv/actor会在learner启动后才会启动, 故这里设置为当前时间, 避免第一次调用就报错
        self.last_on_policy_learner_get_and_connect_actor_time = time.time()
        self.on_policy_learner_get_and_connect_actor_count = 0
        # 因为在on-policy里会使用到故提取到这里
        self.current_sync_model_version_from_learner = 0

        # 格式client_id->zmq_client对象
        self.actor_zmq_client_map = {}

        self.model_file_sync_wrapper = ModelFileSync()

        # 下面是统计告警指标
        self.on_policy_push_to_modelpool_error_count = 0
        self.on_policy_push_to_modelpool_success_count = 0
        self.on_policy_learner_recv_aisrv_error_count = 0
        self.on_policy_learner_recv_aisrv_success_count = 0
        self.on_policy_learner_recv_actor_error_cnt = 0
        self.on_policy_learner_recv_actor_success_cnt = 0
        self.on_policy_learner_change_model_version_cnt = 0

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

        # 传递了trainer对象, 则该类里可以复用, 而不是每次调用都传trainer对象
        self.trainer = trainer

    def get_current_sync_model_version_from_learner(self):
        return self.current_sync_model_version_from_learner

    def before_run(self):
        self.model_file_sync_wrapper.make_model_dirs(self.trainer.logger)

        self.alloc_util = AllocUtils(self.trainer.logger)
        self.trainer.logger.info("train OnPolicyStrategy before_run success")

    def train_stat(self):
        """
        on_policy里的特有统计指标
        """
        monitor_data = {
            KaiwuDRLDefine.ON_POLICY_PUSH_TO_MODELPOOL_ERROR_CNT: self.on_policy_push_to_modelpool_error_count,
            KaiwuDRLDefine.ON_POLICY_PUSH_TO_MODELPOOL_SUCCESS_CNT: self.on_policy_push_to_modelpool_success_count,
            KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_AISRV_ERROR_CNT: self.on_policy_learner_recv_aisrv_error_count,
            KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_AISRV_SUCCESS_CNT: self.on_policy_learner_recv_aisrv_success_count,
            KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_ACTOR_ERROR_CNT: self.on_policy_learner_recv_actor_error_cnt,
            KaiwuDRLDefine.ON_POLICY_LEARNER_RECV_ACTOR_SUCCESS_CNT: self.on_policy_learner_recv_actor_success_cnt,
            # on-policy, 样本只是使用1次, 此时样本生产消耗比为1
            KaiwuDRLDefine.SAMPLE_PRODUCTION_AND_CONSUMPTION_RATIO: 1,
        }

        return monitor_data

    def train_condition(self, current_size):
        """
        是否满足训练条件
        """

        """
        这里需要区分下:
        1. 如果是on-policy, 必须等current_size大于batch_size才能进入到self.train_detail逻辑,
            否则因为aisrv在等learner的on-policy响应, learner在等aisrv产生样本, 就出现死锁
        """

        """
        learner满足训练条件的情况:
        1. on-policy
        1.1 大于batch_size, 才开始训练
        """

        condition = current_size >= int(CONFIG.train_batch_size)
        return condition

    def process_policy_specific(self, model_file_id):

        # 如果learner训练成功即开始走on-policy的逻辑
        self.current_sync_model_version_from_learner = model_file_id

        is_train_success = True
        self.learner_on_policy_process(is_train_success)

    def cleanup(self):
        """
        清理时的策略特定操作
        """
        pass

    def periodic_operation(self):
        """
        定时的策略特定操作
        """

        # on-policy情况下, learner需要知道aisrv/actor地址
        now = time.time()
        if now - self.last_on_policy_learner_get_and_connect_actor_time >= CONFIG.prometheus_stat_per_minutes * 60:
            self.on_policy_learner_get_and_connect_actor()
            self.last_on_policy_learner_get_and_connect_actor_time = now

    def on_policy_learner_get_and_connect_actor(self):
        """
        learner获取actor地址并且建立TCP连接, 包括下面的操作:
        1. 获取actor地址, 分是否使用alloc服务
        2. 根据1中获取actor地址情况进行处理
        2.1 如果1中获取actor地址失败, 则下次重试
        2.2 如果1中获取actor地址成功, 则本次执行

        执行on_policy_learner_get_actor_address函数的情况:
        1. self.actor_zmq_client_map为空, 即当前learner没有获取到actor地址
        2. 即使learner当前维护了self.actor_zmq_client_map地址, 因为可能存在部分actor进程是异常重启的, 那么需要周期性的去重新获取
        """

        self.on_policy_learner_get_and_connect_actor_count += 1

        # 如果self.actor_zmq_client_map为空则走获取actor地址流程
        if not self.actor_zmq_client_map:
            self.on_policy_learner_get_actor_address()
        else:
            if (
                self.on_policy_learner_get_and_connect_actor_count
                >= CONFIG.on_policy_learner_get_actor_address_periodic_count
            ):
                self.on_policy_learner_get_actor_address()
                self.on_policy_learner_get_and_connect_actor_count = 0

        # 发送和接收心跳请求
        if self.actor_zmq_client_map:
            self.on_policy_learner_recv_actor_heartbeat_req_resp()

    # learner推送model文件到modelpool去, 加上重试机制
    def learner_push_model_to_modelpool(self):
        all_push_model_success = False
        retry_count = 0

        while not all_push_model_success and retry_count < int(CONFIG.on_policy_error_retry_count_when_modelpool):
            push_model_success = self.model_file_sync_wrapper.push_checkpoint_to_model_pool(self.trainer.logger)
            if not push_model_success:
                # 如果本次失败, 则sleep下再重试, 这里重试的间隔设置大些
                time.sleep(CONFIG.idle_sleep_second * 1000)
            else:
                all_push_model_success = True
                self.trainer.logger.info(f"train learner learner_push_model_to_modelpool success")
                break

            retry_count += 1

        return all_push_model_success

    def recv_model_sync_response(self, zmq_client_map):
        """
        获取发出去的model_version同步请求的响应
        1. learner <--> aisrv
        2. learner <--> actor
        """

        if not zmq_client_map:
            return True

        # learner等待actor确认加载model文件完成通知, 错误情况接入监控告警
        success_recv_cnt = 0
        # 真正完成了model_sync_version操作的计数
        success_model_sync_cnt = 0

        retry_count = 0
        # aisrv/actor会返回结果, 但是结果里有正确和错误的区分, 故采用下面2个变量实现
        response_success_ip = {}
        model_version_change_ip = {}

        """
        重试时间即等于retry_count * CONFIG.idle_sleep_second
        1. actor是在主循环里加载, 采用默认的retry_count * CONFIG.idle_sleep_second即可
        2. aisrv的超时时间设置如下:
        2.1 如果不是按照单局或者单帧的, 采用默认的retry_count * CONFIG.idle_sleep_second即可
        2.2 如果是按照单局或者单帧的, 采用的值需要大于2 * CONFIG.on_policy_timeout_seconds
        """
        while success_recv_cnt != len(zmq_client_map) and retry_count < int(CONFIG.on_policy_error_retry_count):
            for ip, zmq_client in zmq_client_map.items():
                # 如果已经成功的不需要重复获取响应
                if ip not in response_success_ip:
                    try:
                        recv_data = zmq_client.recv(block=False, binary=False)
                        if recv_data:
                            if (
                                recv_data[KaiwuDRLDefine.MESSAGE_TYPE]
                                == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE
                            ):
                                response_success_ip[ip] = ip

                                # 每个aisrv/actor明确返回model_version修改结果
                                if recv_data[KaiwuDRLDefine.MESSAGE_VALUE]:
                                    model_version_change_ip[ip] = ip
                                    success_model_sync_cnt += 1

                                success_recv_cnt += 1

                            else:
                                # 如果这里陷入重试操作, 此时aisrv发送的on-policy流程的请求可能被忽略, 故这里加上日志验证
                                self.trainer.logger.error(
                                    "train process learner model sync recv un support "
                                    f"{recv_data[KaiwuDRLDefine.MESSAGE_TYPE]}"
                                )

                    except Exception as e:
                        # 减少CPU争用
                        time.sleep(CONFIG.idle_sleep_second)

            retry_count += 1

        """
        其返回值的情况如下:
        1. 返回True
        1.1 所有预测进程返回了更新model_version请求成功的响应
        1.2 部分预测进程返回了更新model_version请求成功的响应
        2. 返回False
        2.1 所有预测进程没有返回更新model_version的响应
        2.2 所有预测进程没有返回更新model_version请求成功的响应
        """
        if not success_recv_cnt or not success_model_sync_cnt:
            self.trainer.logger.warning(
                f"train process success_recv_cnt {success_recv_cnt} or success_model_sync_cnt {success_model_sync_cnt} is 0, please check"
            )
            return False

        if success_recv_cnt != len(zmq_client_map):
            keys1 = set(response_success_ip.keys())
            keys2 = set(zmq_client_map.keys())

            self.trainer.logger.warning(f"train process learner model sync not recv resp ips: {keys2-keys1}")
            return True
        else:
            """
            因为此时响应包已经发送回来了, 即使内容表述的是aisrv没有执行成功, 则也返回True
            下次on_policy流程再次执行成功后会让整个on_policy流程正常, 故这里的日志修改为warning级别
            """
            if success_model_sync_cnt != len(zmq_client_map):
                keys1 = set(model_version_change_ip.keys())
                keys2 = set(zmq_client_map.keys())

                self.trainer.logger.warning(
                    f"train process learner model sync recv resp but recv error resp ips: {keys2-keys1}"
                )
                return True

        return True

    # on-policy场景下, learner获取actor地址
    def on_policy_learner_get_actor_address(self):
        """
        1. 如果不使用alloc服务, 则直接使用本地配置, 本地配置为空则使用127.0.0.1
        2. 如果使用alloc服务, 则直接使用alloc服务

        无论是远程预测还是本地预测, 都采用self.actor_zmq_client_map变量里, 注意区分是aisrv还是actor地址
        """
        aisrv_or_actor_address = [KaiwuDRLDefine.LOCAL_HOST_IP]
        if int(CONFIG.use_alloc):
            self.alloc_util.registry()

            # 默认的返回actor地址, 但如果是采用本地预测, 则返回的是aisrv地址
            svr_name = KaiwuDRLDefine.SERVER_ACTOR
            if (
                CONFIG.remote_agent_default_runtime_mode
                == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT
            ):
                svr_name = KaiwuDRLDefine.SERVER_AISRV

            aisrv_or_actor_address = self.alloc_util.get_all_address_by_srv_name(svr_name)
            if not aisrv_or_actor_address:
                self.trainer.logger.warning(f"train get actor_address error, retry next time")
                return
            else:
                self.trainer.logger.info(f"train get actor_address success,  actor address: {aisrv_or_actor_address}")
        else:
            self.trainer.logger.info(f"train set use_alloc False, so actor use {KaiwuDRLDefine.LOCAL_HOST_IP}")

        # 本次没有获取到预测进程地址即返回, 还是使用旧的预测地址进行通信
        if not aisrv_or_actor_address:
            return
        else:
            """
            本次获取到预测进程地址, 采用下面方法进行处理:
            1. 该IP是本次获取到的, 并且在self.actor_zmq_client_map里没有的则新增, 对应场景是新增预测进程
            2. 该IP是本次获取到的, 并且在self.actor_zmq_client_map已经有的不做操作, 对应场景是预测进程不增加也不减少
            3. 该IP本次没有获取到, 但是self.actor_zmq_client_map是存在的需要从self.actor_zmq_client_map删除, 对应场景是减少预测进程
            """
            current_addresses = set()
            for address in aisrv_or_actor_address:
                client_id = get_uuid()
                actor_ip = address.split(":")[0]

                """
                如果是actor地址, 则规则为actor_port = int(CONFIG.zmq_server_port) + 100
                如果是aisrv地址, 则规则为根据policy的数量, 循环下按照actor_port = int(CONFIG.zmq_server_port) + (index + 1) * 100
                由于是不同的policy下都需要调用故需要区分policy
                """
                if (
                    CONFIG.remote_agent_default_runtime_mode
                    == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT
                ):
                    sorted_items = sorted(self.trainer.policy_agent_wrapper_maps.items(), key=lambda item: item[0])
                    sorted_keys = [key for key, value in sorted_items]

                    for key in sorted_keys:
                        key_index = sorted_keys.index(key)
                        for idx in range(1):
                            actor_port = int(CONFIG.zmq_server_port) + (idx + 1) * 100 + key_index
                            address_key = f"{actor_ip}:{actor_port}"
                            if address_key not in self.actor_zmq_client_map:
                                zmq_client = ZmqClient(str(client_id), actor_ip, actor_port, self.zmq_config)
                                zmq_client.connect()

                                self.actor_zmq_client_map[address_key] = zmq_client
                                current_addresses.add(address_key)
                else:
                    actor_port = int(CONFIG.zmq_server_port) + 100
                    address_key = f"{actor_ip}:{actor_port}"
                    if address_key not in self.actor_zmq_client_map:
                        zmq_client = ZmqClient(str(client_id), actor_ip, actor_port, self.zmq_config)
                        zmq_client.connect()

                        self.actor_zmq_client_map[address_key] = zmq_client
                        current_addresses.add(address_key)

            # 现在检查self.actor_zmq_client_map中的地址，如果不在current_addresses中，则删除
            keys_to_remove = [key for key in self.actor_zmq_client_map if key not in current_addresses]
            for key in keys_to_remove:
                del self.actor_zmq_client_map[key]

    # on-policy场景下, learner与actor地址建立连接, 周期性的发送/接收心跳保活请求/响应
    def on_policy_learner_recv_actor_heartbeat_req_resp(self):
        if not self.actor_zmq_client_map:
            return

        # 因为心跳请求的send_data是一致的, 故可以放在循环外面
        send_data = {
            KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
            KaiwuDRLDefine.MESSAGE_VALUE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST,
        }

        learner_send_to_actor_heartbeat_success_count = 0
        for actor_ip, zmq_client in self.actor_zmq_client_map.items():
            zmq_client.send(send_data, binary=False)
            self.trainer.logger.debug(f"train send heartbeat request to actor: {actor_ip} success")

            # 同步等待心跳响应回包
            retry_count = 0
            while retry_count < int(CONFIG.on_policy_error_retry_count):
                try:
                    recv_data = zmq_client.recv(block=False, binary=False)
                    if recv_data:
                        if (
                            recv_data[KaiwuDRLDefine.MESSAGE_TYPE]
                            == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE
                        ):
                            self.trainer.logger.debug(f"train recv heartbeat response to actor: {actor_ip} success")
                            learner_send_to_actor_heartbeat_success_count += 1
                            break
                except Exception as e:
                    # 减少CPU争用
                    time.sleep(CONFIG.idle_sleep_second)

                retry_count += 1

        # 以为心跳的请求频率比较高, 打印日志比较耗时, 故采用debug日志
        if learner_send_to_actor_heartbeat_success_count == len(self.actor_zmq_client_map):
            self.trainer.logger.debug(
                f"train learner recv all actor heartbeat response success, "
                f"count: {learner_send_to_actor_heartbeat_success_count}"
            )

        else:
            # 由于无法收到actor的请求, 那么此时不确定actor的情况是怎么样, 故清空self.actor_zmq_client_map, 重新拉取看下效果
            self.trainer.logger.warning(
                f"train learner not recv all actor heartbeat response, retry next time, "
                f"learner_send_to_actor_heartbeat_success_count "
                f"{learner_send_to_actor_heartbeat_success_count} != "
                f"len(actor_zmq_client_map) {len(self.actor_zmq_client_map)}"
            )

            self.actor_zmq_client_map.clear()
            self.on_policy_learner_get_actor_address()

    # learner朝actor发送model_version请求和收取响应
    def learner_send_and_recv_actor_model_version_request_and_response(self, send_data):
        if not send_data:
            return False

        """
        因为存在learner还没有周期性(1分钟)的获取到预测进程地址而开始了on-policy流程的情况, 处理流程如下:
        1. 如果self.actor_zmq_client_map是空, 则调用self.on_policy_learner_get_and_connect_actor
        2. 如果self.on_policy_learner_get_and_connect_actor调用后还是空, 说明本次获取不到预测进程地址,
        2.1 不需要发送model_version同步请求, 只有等下一次触发on-policy流程
        2.2 model_version减去1
        """
        if not self.actor_zmq_client_map:
            self.on_policy_learner_get_and_connect_actor()

        if not self.actor_zmq_client_map:
            self.learner_reset_model_version()
            return True

        for actor_ip, zmq_client in self.actor_zmq_client_map.items():
            zmq_client.send(send_data, binary=False)
            self.trainer.logger.info(
                "train process learner send model_version sync request to actor: "
                f"{actor_ip}, model_version: {self.current_sync_model_version_from_learner}"
            )

        """
        通知预测进程更新模型版本号, 其容灾情况如下:
        1. 如果超过设置的CONFIG.on_policy_error_max_retry_rounds后, 满足下面的则进程主动退出
        1.1 所有的预测进程没有全部回复
        1.2 所有的预测进程即使全部回复了但是是更新model_version错误的响应
        2. 如果在设置的CONFIG.on_policy_error_max_retry_rounds次数内, 满足下面的则进程继续
        2.1 所有的预测进程全部回复, 并且都是跟新model_version成功的响应, 最佳情况
        2.2 部分的预测进程回复, 并且都是更新model_version成功的响应, 那么依靠剩余的预测进程能继续推动on-policy继续
        """
        learner_recv_all_actor_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            if self.recv_model_sync_response(self.actor_zmq_client_map):
                learner_recv_all_actor_success = True
                break

        if learner_recv_all_actor_success:
            self.trainer.logger.info(f"train process learner recv all the actor newest model sync resp")
            self.on_policy_learner_recv_actor_success_cnt += 1
            return True

        else:
            self.trainer.logger.warning(f"train process learner recv not all the actor newest model sync resp")

            # 增加告警和容灾
            self.on_policy_learner_recv_actor_error_cnt += 1
            return False

    def learner_reset_model_version(self):
        """
        如果失败时, learner主动回退版本号, 此时版本号是按照模型落地文件里的CONFIG.dump_model_freq
        """
        self.current_sync_model_version_from_learner -= CONFIG.dump_model_freq

    def learner_on_policy_process(self, is_train_success):
        """
        on-policy需要启动流程:
        1. 根据是否训练下面操作:
        1.1 训练成功则:
        1.1.1 清空样本池, 一般不会失败
        1.1.2 learner推送model文件到modelpool
        1.1.2.1 如果成功则继续剩余流程
        1.1.2.2 失败则告警指标增加, 本次的训练侧增加的model版本号需要--, 规避样本错误的过滤掉
        1.1.3 learner通知aisrv/actor预测进程从modelpool拉取model文件
        1.1.3.1 如果成功则继续剩余流程
        1.1.3.2 如果不成功则告警, 下一步做容灾
        1.1.4 learner等待aisrv/actor预测进程确认加载model文件完毕通知
        1.1.4.1 如果成功则继续剩余流程
        1.1.4.2 如果不成功则告警, 下一步做容灾
        1.1.5 learner继续训练, 注意会采用样本过滤
        1.2 训练不成功, 即模型版本号没有变化, 不做处理
        """

        # 清空样本池, 如果本次有进行训练才能清空样本池, 否则不需要清空, 如果强制清空, 下次learner可能会卡在reverb读写上面
        if is_train_success:
            self.trainer.replay_buffer_wrapper.reset(self.trainer.cached_local_step, None)
            self.trainer.logger.info(f"train learner have train, so reverb reset success")
        else:
            self.trainer.logger.info(f"train learner not have train, so reverb not need reset")

        # 只有本次明确的训练成功了才会走下面的逻辑
        if is_train_success:
            # learner推送model文件到modelpool, 有重试机制
            learner_push_model_file_success = False
            for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
                if self.learner_push_model_to_modelpool():
                    learner_push_model_file_success = True
                    break

            """
            如果本次leaner推送到modelpool失败时, learner自身来说可以下一次再推送model文件重试, 并且可以下一次再走on-policy流程, 下面处理方法优缺点
            1. 告警指标++, 无需同步预测处最新的模型版本号给预测进程
            """
            if not learner_push_model_file_success:
                # 因为站在learner的角度看, 某次on-policy中推送modelpool文件失败, 下次再进行on-policy即可, 故打印waring日志, 接入告警
                self.trainer.logger.warning(f"train process learner push_checkpoint_to_model_pool failed, so return")
                self.on_policy_push_to_modelpool_error_count += 1

                """
                站在learner的角度看, 因为当前的model文件没有推送到modelpool导致预测进程无法获取新的版本号
                故当前的训练侧的版本号要减去CONFIG.dump_model_freq, 免得导致样本被过滤掉了
                """
                self.trainer.logger.warning(
                    f"train process learner on_policy complete failed on model_version: {self.current_sync_model_version_from_learner}"
                )
                self.learner_reset_model_version()

            else:
                self.on_policy_push_to_modelpool_success_count += 1
                # on_policy_learner_change_model_version_cnt代表是真实的model_version次数, 故只有在真实的同步时计数
                self.on_policy_learner_change_model_version_cnt += 1

                """
                消息格式:
                message_type: xxxx
                message_value: yyyy
                """
                send_data = {
                    KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST,
                    KaiwuDRLDefine.MESSAGE_VALUE: self.current_sync_model_version_from_learner,
                }

                if self.learner_send_and_recv_actor_model_version_request_and_response(send_data):
                    self.trainer.logger.info(
                        f"train process learner on_policy complete success on model_version: "
                        f"{self.current_sync_model_version_from_learner}"
                    )
                else:
                    # 在on-policy情况下主动退出进程, 本次on_policy流程失败, 接入告警, 但是整体on_policy流程继续执行, 单次失败不影响整体流程
                    self.trainer.logger.warning(
                        f"train process learner on_policy complete failed on model_version: "
                        f"{self.current_sync_model_version_from_learner}"
                    )

    def strategy_name(self):
        """
        策略特定的名字
        """
        return "on_policy"
