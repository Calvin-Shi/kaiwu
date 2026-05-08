#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from distutils.dir_util import remove_tree
import hashlib
import os
import shutil
import tempfile
import time
import schedule
import datetime
import random
import multiprocessing
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.checkpoint.model_pool_apis import ModelPoolAPIs
from kaiwudrl.common.utils.common_func import (
    clean_dir,
    insert_any_string,
    make_tar_file,
    tar_file_extract,
    TimeIt,
    get_first_last_line_from_file,
    get_host_ip,
    move_files,
)
from multiprocessing import Manager, RLock
from common_python.logging.kaiwu_logger import g_not_server_label
from kaiwudrl.common.checkpoint.model_path_manager import MODEL_PATH_MANGER
from kaiwudrl.common.checkpoint.model_file_common import get_checkpoint_id_by_re
from contextlib import nullcontext
from common_python.worker.worker import Worker, WorkerConfig

# 如果是tensorrt的加载dump_weights
if CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine().DEEP_LEARNING_FRAMEWORK_TENSORRT:
    from kaiwudrl.server.cpp.tools.dump_tf_weights import dump_weights


class ModelFileSync(Worker):
    """
    actor <--> learner之间传递的Model文件任务重, 单个进程处理
    """

    def __init__(self) -> None:
        # 进程pid
        self.current_pid = os.getpid()
        worker_config = WorkerConfig(
            worker_name="model_file_sync",
            father_pid=self.current_pid,
            use_logger=True,
            use_default_monitor=True,
            use_default_alloc=False,
        )
        super().__init__(worker_config)

        self.exit_flag = multiprocessing.Value("b", False)

        # 调用modelpool, model_pool_addrs 在配置conf/kaiwudrl/configure.toml, 格式形如'127.0.0.1:10013', 以,号分割
        self.remote_addrs = CONFIG.modelpool_remote_addrs

        self.model_pool_apis = ModelPoolAPIs(self.remote_addrs)
        self.model_pool_apis.check_server_set_up()

        # modelpool 相关统计, 需要区分是actor还是learner进程
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            self.push_to_model_pool_succ_cnt = 0
            self.push_to_model_pool_err_cnt = 0
        else:
            self.pull_from_model_pool_succ_cnt = 0
            self.pull_from_model_pool_err_cnt = 0

        """
        由于model同步进程和actor的预测进程之间存在争用的情况, 故这里增加了锁文件
        1. lock_write_file, model_file_sync进程使用
        2. lock_read_file, actor的predict进程使用
        3. actor_predict_count_value, 跨进程通信计数
        """
        self.lock_read_file = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.LOCK_READ_FILE}"
        self.lock_write_file = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.LOCK_WRITE_FILE}"

        # 删除已经存在的lock_read_file和lock_write_file
        if os.path.exists(self.lock_read_file):
            os.remove(self.lock_read_file)

        if os.path.exists(self.lock_write_file):
            os.remove(self.lock_write_file)

        # actor_proxy_local进程进行++, model_file_sync进程进行清零
        self.actor_predict_count_value = multiprocessing.Value("i", 0)

        # model_file_sync进程只有第一次拉下来model文件后, actor_proxy_local进程才能进行load_last_new_model
        self.had_pull_the_first_model_file_success = multiprocessing.Value("b", False)

        """
        主要使用在下面的场景里面, 多进程之间的多读单写操作, 只有在预测进程启动的model_file_sync里需要启动, 训练进程侧启动的model_file_sync不需要启动
        1. model_file_sync从modelpool下载model文件到容器, 并且维护可用的model文件列表, 属于写操作
        2. 预测进程, 无论是哪种模式下都是从可用的model文件列表里读取数据, 而不是从modelpool下载, 属于读操作
        """
        self.manager = None
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            self.manager = Manager()
            self.current_available_model_files = self.manager.dict(
                {"version": 0, "models": self.manager.list(), "lock": self.manager.RLock()}
            )
        # 为了规避多次重复push到modelpool重复的model文件, 故记录了最近一次上传的checkpoint_id
        self.last_success_push_to_modelpool_checkpoint_id = -1

        """
        下面的变量, 由于该类即可能以进程存在, 也可能以对象调用存在, 故这些变量先定义下
        """
        self.plugins_path = None
        self.models_path = None
        self.convert_model_dir = None

        # 存放到modelpool的当前index
        self.current_modelpool_save_model_index = 0

    def make_local_model_dirs(self, logger):
        """
        预测放在workflow里时需要调用下生成临时目录, 因为有多个workflow进程故这里的plugins_path和models_path都需要带上pid
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            ip = get_host_ip()
            slot_id = self.current_pid
            self.plugins_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{ip}_{slot_id}/plugins/"
            if not os.path.exists(self.plugins_path):
                os.makedirs(self.plugins_path)

            self.models_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{ip}_{slot_id}/models/"
            if not os.path.exists(self.models_path):
                os.makedirs(self.models_path)

            self.models_path_new = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{ip}_{slot_id}/models_new/"
            if not os.path.exists(self.models_path_new):
                os.makedirs(self.models_path_new)

            """
            采用flag指示当前是写入哪个model文件夹:
            0: model_path_new
            1: model_path
            """
            self.current_modes_path_flag = 0

            logger.info(
                f"model_file_sync mkdir {self.models_path} {self.models_path_new} {self.plugins_path} success",
                g_not_server_label,
            )

    def make_model_dirs(self, logger):
        """
        创建上传和下载model文件的临时目录
        1. learner上的上传目录为/tmp下临时目录, 需要代码里删除, 形如/tmp/tmpyiu2tmhp/
        2. actor的下载目录为ckpt_dir下的业务名_算法名的plugins, 代码里不需要删除, 形如/data/ckpt/sgame_ppo/plugins/
        3. actor的加载model的目录为ckpt_dir下的业务名_算法名的models, 代码里不需要删除, 形如/data/ckpt/sgame_ppo/modles/
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            self.plugins_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/plugins"
            if not os.path.exists(self.plugins_path):
                os.makedirs(self.plugins_path)
            logger.info(f"model_file_sync mkdir {self.plugins_path} success", g_not_server_label)

            self.models_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/models"
            if not os.path.exists(self.models_path):
                os.makedirs(self.models_path)

            self.models_path_new = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/models_new/"
            if not os.path.exists(self.models_path_new):
                os.makedirs(self.models_path_new)

            """
            采用flag指示当前是写入哪个model文件夹:
            0: model_path_new
            1: model_path
            """
            self.current_modes_path_flag = 0

            logger.info(
                f"model_file_sync mkdir {self.models_path} {self.models_path_new} {self.plugins_path} success",
                g_not_server_label,
            )

        # convert_models, 作为actor和learner都需要创建的
        self.convert_model_dir = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/convert_models_{CONFIG.svr_name}"
        if not os.path.exists(self.convert_model_dir):
            os.makedirs(self.convert_model_dir)
        logger.info(
            f"model_file_sync mkdir {self.convert_model_dir} success",
            g_not_server_label,
        )

    # 调用函数获取最新的model文件
    def get_last_model_file(self, logger):
        (
            pull_model_success,
            current_available_model_files,
        ) = self.pull_checkpoint_from_model_pool(logger, self.models_path, self.plugins_path, return_model_files=True)
        if pull_model_success:
            logger.info(f"model_file_sync get_last_model_file success, model_path: {self.models_path}")
        else:
            # 因为这个会多次拉取, 训练进程还没有上传model文件而预测进程正在拉取, 故采用休息1s
            time.sleep(1)

        return pull_model_success, current_available_model_files

    def stop(self):
        self.exit_flag.value = True
        self.join()
        if self.manager:
            self.manager.shutdown()

        self.logger.info("model_file_sync ModelFileSync stop success", g_not_server_label)

    def after_run(self) -> bool:
        pass

    def before_run(self):
        # 先调用基类初始化
        if not super().before_run():
            return False

        # fork后重新获取子进程pid
        self.current_pid = os.getpid()

        self.logger.info(
            f"{CONFIG.svr_name} model_file_sync, use {CONFIG.ckpt_sync_way}, "
            f"use_which_deep_learning_framework is {CONFIG.use_which_deep_learning_framework}",
            g_not_server_label,
        )

        self.logger.info(f"model_file_sync process pid is {self.current_pid}", g_not_server_label)

        self.make_model_dirs(self.logger)

        self.process_run_count = 0

        t = time.time()
        self.last_run_schedule_time_by_push_pull = t
        self.last_run_schedule_time_by_sync_stat = t

        # 注册定时器任务
        # set_schedule_event(CONFIG.model_file_sync_per_minutes, self.push_and_pull_model_file)
        # set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.model_file_sync_stat)

        return True

    def clear_current_available_model_files(self):
        """
        清空队列, 在model_file_sync进程每次进行拉取时如果modelpool有文件时则需要清空, 否则保留的是旧数据的
        """
        self.current_available_model_files["models"][:] = []

    def get_current_available_model_files(self):
        """
        获取到可用的model文件列表, 直接返回self.current_available_model_files数据结构, 由调用者使用
        """
        with self.current_available_model_files["lock"]:
            return self.current_available_model_files["models"]

    def get_current_available_model_file(self, model_id, logger):
        """
        获取到可用的model文件, 在model_file_sync进程里处理self.current_available_model_files, 再返回具体的model_file和model_id
        """
        with self.current_available_model_files["lock"]:
            current_available_model_files_list = self.current_available_model_files["models"]
            if not current_available_model_files_list:
                logger.info(f"model_file_sync current_available_model_files is empty")
                return -1, None

            # 直接操作共享列表，无需转移数据
            if model_id == KaiwuDRLDefine.ID_LATEST:
                model_file = current_available_model_files_list[-1]
                current_id = get_checkpoint_id_by_re(model_file)

            elif model_id == KaiwuDRLDefine.ID_RANDOM:
                model_file = random.choice(current_available_model_files_list)
                current_id = get_checkpoint_id_by_re(model_file)

            else:
                try:
                    model_id = int(model_id)
                    found = False
                    for model_file in current_available_model_files_list:
                        if model_id == get_checkpoint_id_by_re(model_file):
                            current_id = model_id
                            found = True
                            break
                    if not found:
                        logger.error(f"model_file_sync model_id {model_id} not found in available files")
                        return -1, None

                except Exception as e:
                    logger.error(f"model_file_sync id is error, id is {model_id}, e is {e}")
                    return -1, None

        path = os.path.dirname(model_file)
        return current_id, path

    def get_modelpool_key(self):
        """
        在modelpool里保存的模型文件, 采用FIFO的模式, 比如要保存50个文件, 则key是从key_0, key_1, ......, key49, key的规范如下:
        model.ckpt_app_algo_index
        """
        key = f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}_{CONFIG.app}_{CONFIG.algo}_{self.current_modelpool_save_model_index}"
        self.current_modelpool_save_model_index = (
            self.current_modelpool_save_model_index + 1
        ) % CONFIG.modelpool_max_save_model_count
        return key

    def get_all_modelpool_keys(self):
        """
        获取设置的modelpool的keys, 用于在拉取modelpool文件时使用
        """
        return [
            f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}_{CONFIG.app}_{CONFIG.algo}_{index}"
            for index in range(CONFIG.modelpool_max_save_model_count)
        ]

    def model_file_sync_stat(self):
        if int(CONFIG.use_prometheus):
            monitor_data = {}
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
                monitor_data[KaiwuDRLDefine.PULL_FROM_MODEL_POOL_SUCC_CNT] = self.pull_from_model_pool_succ_cnt
                monitor_data[KaiwuDRLDefine.PULL_FROM_MODEL_POOL_ERR_CNT] = self.pull_from_model_pool_err_cnt
            else:
                monitor_data[KaiwuDRLDefine.PUSH_TO_MODEL_POOL_SUCC_CNT] = self.push_to_model_pool_succ_cnt
                monitor_data[KaiwuDRLDefine.PUSH_TO_MODEL_POOL_ERR_CNT] = self.push_to_model_pool_err_cnt

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

            # 由于是一直朝上增长的指标, 不需要指标复原, 能看见代码在正常运行, 可以根据周期间隔计算出时间段内的执行次数

    def push_and_pull_model_file(self):

        """
        如果是eval模式不需要learner/actor之间的model文件同步
        """
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_EXAM:
            return

        # 有些场景不需要使用modelpool, 比如train_test
        if not CONFIG.use_ckpt_sync:
            return

        """
        下面是actor/learner与modelpool交互情况:
        1. 如果是tensorflow, learner推送checkpoint文件, actor拉取checkpoint文件
        2. 如果是tensorrt, learner推送wight文件, actor拉取wight文件
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            if (
                CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE
                or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX
                or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH
            ):
                self.push_checkpoint_to_model_pool(self.logger)
            elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT:
                self.push_wight_to_model_pool()
            else:
                self.logger.error(
                    f"model_file_sync un support use_which_deep_learning_framework: "
                    f"{CONFIG.use_which_deep_learning_framework}",
                    g_not_server_label,
                )
                return

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR or CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            if CONFIG.self_play_actor:
                if (
                    CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE
                    or CONFIG.use_which_deep_learning_framework
                    == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX
                    or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH
                ):
                    self.pull_old_checkpoint_from_model_pool()
                elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT:
                    self.pull_old_wight_from_model_pool()
                else:
                    self.logger.error(
                        f"model_file_sync un support use_which_deep_learning_framework: "
                        f"{CONFIG.use_which_deep_learning_framework}",
                        g_not_server_label,
                    )
                    return
            else:
                if (
                    CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_SIMPLE
                    or CONFIG.use_which_deep_learning_framework
                    == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORFLOW_COMPLEX
                    or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH
                ):
                    self.pull_checkpoint_from_model_pool(self.logger)
                elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_TENSORRT:
                    self.pull_wight_from_model_pool()
                else:
                    self.logger.error(
                        f"model_file_sync un support use_which_deep_learning_framework: "
                        f"{CONFIG.use_which_deep_learning_framework}",
                        g_not_server_label,
                    )
                    return

        else:
            self.logger.error(
                f"model_file_sync un support svr_name: {CONFIG.svr_name}",
                g_not_server_label,
            )
            return

    def make_model_tar_file(self):
        """
        流程如下:
        1. 根据形如/data/ckpt/hero_ppo/checkpoint找到最新的step生成的checkpoint文件, 形如:
        all_model_checkpoint_paths: "/data/ckpt//hero_ppo/model.ckpt-3247"
        2. 在/data/ckpt/hero_ppo/ | grep 3247找出满足需求的
            model-3247.data-00000-of-00001, model-3247.index, model-3247.meta, checkpoint
        3. 对2制作成tar文件, 生成tar文件路径
        """

        # 获取到Model文件路径所在的路径
        model_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}"
        checkpoint_file = f"{model_path}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"

        last_line = None
        try:
            first_line, last_line = get_first_last_line_from_file(checkpoint_file)
        except Exception as e:
            pass

        if last_line is None:
            return False, f"last_line is None", None, -1

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = get_checkpoint_id_by_re(last_line)
        # 只是判断checkpoint_id为None的情形, 而checkpoint_id为0是正常的
        if checkpoint_id is None:
            return False, f"checkpoint_id is None, last_line is {last_line}", None, -1

        # 增加判断如果该checkpoint_id没有上传, 那么就上传, 否则不用上传
        if self.last_success_push_to_modelpool_checkpoint_id == checkpoint_id:
            return (
                True,
                f"self.last_success_push_to_modelpool_checkpoint_id "
                f"{self.last_success_push_to_modelpool_checkpoint_id} == checkpoint_id {checkpoint_id}",
                None,
                checkpoint_id,
            )

        target_dir = tempfile.mkdtemp()

        # 寻找包含checkpoint_id的meta, data, index
        for root, dirs, file_list in os.walk(model_path):
            # 排除指定目录
            dirs[:] = [d for d in dirs if d not in MODEL_PATH_MANGER.exclude_directories()]
            for file_name in file_list:
                # 这里要求文件名格式必须是model.ckpt-340.这种, 目前tensorflow和pytorch都满足
                if f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{checkpoint_id}." in file_name:
                    shutil.copy(os.path.join(root, file_name), target_dir)

        # 需要增加checkpoint文件, 此时因为仅仅上传一个checkpoint_id的model文件， 故不需要全量的上传checkpoint文件
        with open(target_dir + f"/{KaiwuDRLDefine.CHECK_POINT_FILE}", "w") as file:
            file.write(f"{first_line}\n{last_line}\n")

        with open(target_dir + f"/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}", "w") as file:
            file.write(f"{first_line}\n{last_line}\n")

        # shutil.copy(checkpoint_file, target_dir)

        # 放在/tmp目录下生成tar文件
        output_file_name = (
            f"{model_path}/{KaiwuDRLDefine.KAIWU_CHECK_POINT_FILE}_{CONFIG.app}_{CONFIG.algo}_{checkpoint_id}.tar.gz"
        )
        make_tar_file(output_file_name, target_dir)

        # 删除/tmp的临时文件
        remove_tree(target_dir)

        return True, None, output_file_name, checkpoint_id

    # 获取checkpoint里最新的文件, 形如model.ckpt-0.meta,model.ckpt-0.index,model.ckpt-0.data
    def get_newest_checkpoint_file_id(self):
        # 获取到Model文件路径所在的路径
        model_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}"

        checkpoint_file = f"{model_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}"

        last_line = None
        try:
            _, last_line = get_first_last_line_from_file(checkpoint_file)
        except Exception as e:
            pass

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = get_checkpoint_id_by_re(last_line)

        return checkpoint_id

    def push_wight_to_model_pool(self):
        """
        流程如下:
        1. 根据当前learner的checkpoint文件生产wight文件
        2. 采用modelpool_api发送给modelpool
        """
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_LEARNER:
            return False

        try:
            """
            learner在启动时就会产生checkpoint文件, 即可以走dump_weights逻辑
            """

            # 当前最新的checkpoint文件
            newest_checkpoint_id = self.get_newest_checkpoint_file_id()
            if int(newest_checkpoint_id) < 0:
                return False

            ckpt_prefix = (
                f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-"
                f"{newest_checkpoint_id}"
            )
            output_filename = f"{self.convert_model_dir}/trt_weights.wts2"
            shareFCWeights = True

            # 生成wight文件, 耗时较多, 采用日志打印
            with TimeIt() as ti:
                dump_weights(ckpt_prefix, output_filename, shareFCWeights)
            self.logger.info(
                f"model_file_sync success dump weights cost {ti.interval} s, "
                f"ckpt_prefix {ckpt_prefix}, shareFCWeights {shareFCWeights}",
                g_not_server_label,
            )

            wight_file = output_filename

            # 获取当前的key
            key = self.get_modelpool_key()

            # push 到modelpool
            with open(wight_file, "rb") as fin:
                model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(
                    model=model,
                    hyperparam=None,
                    key=key,
                    md5sum=local_md5,
                    save_file_name=wight_file.split("/")[-1],
                )

            self.push_to_model_pool_succ_cnt += 1

            self.logger.info(
                f"model_file_sync push to modelpool success, \
            total push to modelpool succ cnt is {self.push_to_model_pool_succ_cnt} \
            total push to modelpool err cnt is {self.push_to_model_pool_err_cnt}",
                g_not_server_label,
            )
            return True

        except Exception as e:
            self.push_to_model_pool_err_cnt += 1
            self.logger.exception(
                f"model_file_sync push wight to modelpool error, \
                 as {str(e)}",
                g_not_server_label,
            )
            return False

    def push_checkpoint_to_model_pool(self, logger=None):
        """
        流程如下:
        1. make_model_tar_file 制作生成的tar文件
        2. 采用modelpool_api发送给modelpool

        因为该函数可能在on-policy下调用, 故日志句柄由外界传入, 在model_file_sync进程内则直接使用logger
        """
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_LEARNER:
            return False

        try:
            # 生成checkpoint的tar文件
            (
                success,
                reason,
                output_file_name,
                checkpoint_id,
            ) = self.make_model_tar_file()
            if not success:
                logger.warning(
                    f"model_file_sync make_model_tar_file is failed. reason is {reason}",
                    g_not_server_label,
                )
                self.push_to_model_pool_err_cnt += 1

                return False
            else:
                if not output_file_name:
                    logger.info(
                        f"model_file_sync make_model_tar_file is success. reason is {reason}",
                        g_not_server_label,
                    )
                    return True

            # 获取当前的key
            key = self.get_modelpool_key()

            # push 到modelpool
            with open(output_file_name, "rb") as fin:
                model = fin.read()
                local_md5 = hashlib.md5(model).hexdigest()
                self.model_pool_apis.push_model(
                    model=model,
                    hyperparam=None,
                    key=key,
                    md5sum=local_md5,
                    save_file_name=output_file_name.split("/")[-1],
                )

            self.push_to_model_pool_succ_cnt += 1

            # 删除output_file_name
            os.remove(output_file_name)

            self.last_success_push_to_modelpool_checkpoint_id = checkpoint_id

            logger.info(
                f"model_file_sync push output_file_name {output_file_name} key {key} to modelpool success, \
            total push to modelpool succ cnt is {self.push_to_model_pool_succ_cnt} \
            total push to modelpool err cnt is {self.push_to_model_pool_err_cnt}",
                g_not_server_label,
            )

            return True

        except Exception as e:
            self.push_to_model_pool_err_cnt += 1
            logger.exception(
                f"model_file_sync push model file to modelpool error, \
                 as {str(e)}",
                g_not_server_label,
            )
            return False

    def change_checkpoint_file_content(self, checkpoint_path, model_path):
        """
        修改checkpoint里文件内容, 采用方式是先清空文件内容, 再接着重新写入内容
        """

        first_line, last_line = get_first_last_line_from_file(checkpoint_path)
        checkpoint_id = get_checkpoint_id_by_re(last_line)
        if checkpoint_id is None:
            return

        with open(checkpoint_path, "w") as file:
            pass

        with open(checkpoint_path, "w") as file:
            file.write(f'model_checkpoint_path: "{model_path}/{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{checkpoint_id}"\n')
            file.write(
                f'all_model_checkpoint_paths: "{model_path}/{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{checkpoint_id}"\n'
            )

    def rename_checkpoint_file(self, checkpoint_path):
        """
        流程如下:
        1. 读取当前的checkpoint文件内容, 因为是根据当前的model文件来加载model文件的, 故只是需要修改当前的model文件相关内容而不是整个checkpoint
        2. 修改
        3. 写回当前文件
        """

        if not checkpoint_path:
            return

        updated_lines = []
        to_insert_str = "models/"

        # 获取checkpoint的开始和结束的2行
        first_line, last_line = get_first_last_line_from_file(checkpoint_path)
        if first_line:
            updated_lines.append(first_line + "\n")

        # 兼容新旧的格式
        model_file_ckpt = KaiwuDRLDefine.KAIWU_MODEL_CKPT

        if last_line:
            updated_line = insert_any_string(last_line, to_insert_str, model_file_ckpt, "before")
            updated_lines.append(updated_line + "\n")

        """
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                updated_line  = insert_any_string(line, to_insert_str, model_file_ckpt, 'before')
                updated_lines.append(updated_line)
        """

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

    def pull_wight_from_model_pool(self):
        """
        流程如下:
        1. 采用modelpool_api从modelpool获取wight文件, 放到形如/data/ckpt/app_algo/convert_models_actor下, 文件名带上当前时间戳
        2. 将1中的文件, 拷贝到/data/ckpt/app_algo下
        3. actor从2中加载wight文件
        4. 清空/data/ckpt/app_algo/convert_models_actor目录
        """
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR and CONFIG.svr_name != KaiwuDRLDefine.SERVER_AISRV:
            return False

        try:
            # 拉取wight文件
            model_name = f"{KaiwuDRLDefine.KAIWU_MODEDL_WIGHT}_{CONFIG.app}_{CONFIG.algo}"
            model_name_list = self.model_pool_apis.pull_keys()
            if model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    return False

                model_file_name = model_info._file_name

                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    return False

                # 删除convert_model_dir下的文件, 采用删除原文件夹, 新增文件夹方式
                clean_dir(self.convert_model_dir)

                model_file_tar_path = f"{self.convert_model_dir}/{model_file_name}"

                # 写入二进制内容到文件里
                with open(model_file_tar_path, "wb+") as file:
                    file.write(model)

                # 将wight文件拷贝到/data/ckpt目录下, 并且生成文件FINISH标志该可用
                shutil.copy(
                    model_file_tar_path,
                    f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}",
                )

                with open(
                    f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.FILE_FINISH_NAME}",
                    "w+",
                ) as file:
                    file.write(KaiwuDRLDefine.FILE_FINISH_NAME)

                self.pull_from_model_pool_succ_cnt += 1

                self.logger.info(
                    f"model_file_sync pull wight {model_file_name} from modelpool to {self.models_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}",
                    g_not_server_label,
                )
                return True

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            self.logger.exception(
                f"model_file_sync pull weight from modelpool error, as {str(e)}, ",
                g_not_server_label,
            )
            return False

    def get_read_file_lock(self):
        open(self.lock_read_file, "a").close()

    def get_write_file_lock(self):
        open(self.lock_write_file, "a").close()

    def release_read_file_lock(self):
        os.remove(self.lock_read_file)

    def release_write_file_lock(self):
        os.remove(self.lock_write_file)

    def get_lock_read_file(self):
        return self.lock_read_file

    def get_lock_write_file(self):
        return self.lock_write_file

    def get_had_pull_the_first_model_file_success_value(self):
        return self.had_pull_the_first_model_file_success.value

    def pull_checkpoint_from_model_pool_by_on_policy(self, logger, model_file_sync_wrapper_lock):
        """
        on-policy场景下调用, 如果同一个容器里有多个预测进程, 其中model_file_sync_wrapper_lock锁的意义主要是在预测进程需要单独的进行调用
        典型的是小规模场景下aisrv的预测进程在一个容器里就可能导致出现问题
        """
        with model_file_sync_wrapper_lock:
            return self.pull_checkpoint_from_model_pool(logger=logger, return_model_files=True)

    def pull_checkpoint_from_model_pool(self, logger=None, model_path=None, plugin_path=None, return_model_files=False):
        """
        流程如下:
        1. 采用modelpool_api从modelpool获取tar文件, 解压到目录ckpt_dir下的业务名_算法名的plugins
        2. 遍历ckpt_dir下的业务名_算法名的plugins, 拷贝model-ckpt的开头的文件到ckpt_dir下的业务名_算法名的models
        3. actor从ckpt_dir下的业务名_算法名的models下加载model文件
        4. 清空ckpt_dir下的业务名_算法名的plugins

        参数配置:
        因为该函数可能在on-policy下调用, 故日志句柄由外界传入, 在model_file_sync进程内则直接使用logger
        因为该函数可能在off-policy下调用, 由调用者指定具体的model_path位置, 故支持传入model_path
        因为如果直接调用该函数, 则可以返回具体的model文件, 不需要通过进程之间通信来返回具体的model文件, 故支持传入return_model_files
        """
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR and CONFIG.svr_name != KaiwuDRLDefine.SERVER_AISRV:
            return False, None

        try:
            # 拉取checkpoint的tar文件, 一次性拉取完成, 总的耗时 = 单次耗时 * 次数
            model_name_list = self.model_pool_apis.pull_keys()
            if model_name_list:
                """
                下面分场景:
                1. 由于需要支持保留最近N个model文件的特性, 故进程间通信的manager.list在写入时如果有多个model文件则加锁耗时较长,
                   故先放在current_available_model_files, 等完成后集中的处理, 目的是减少锁的占用耗时
                2. 如果只是返回普通的队列, 那么可以直接放在current_available_model_files里
                """
                current_available_model_files = []
            else:
                # 由于刚开始时modelpool里暂时没有model文件是正常的, 故设置为info日志
                logger.info(f"model_file_sync self.model_pool_apis.pull_keys() is None or empty, so return")
                return False, None

            """
            由于存在保存多个模型文件modelpool_max_save_model_count, 支持需要采用先后顺序来获取模型文件, 故这里需要排序, 
            不过增加了一次访问, 耗时可控
            """
            sorted_model_files = []
            for model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    logger.warning(f"model_file_sync model_name {model_name} model_info is None, so continue")
                    continue

                model_file_name = model_info._file_name
                time_stamp_nano = model_info._time_stamp_nano

                sorted_model_files.append(
                    {
                        "time_stamp_nano": time_stamp_nano,
                        "model_file_name": model_file_name,
                        "model_name": model_name,
                    }
                )

            # 按照time_stamp_nano从小到大排序
            sorted_model_files.sort(key=lambda x: x["time_stamp_nano"])
            logger.info(f"model_file_sync after sorted, sorted_model_files is {sorted_model_files}")

            for sorted_model_file in sorted_model_files:
                model_name = sorted_model_file["model_name"]
                model_file_name = sorted_model_file["model_file_name"]

                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    logger.warning(f"model_name {model_name} model is None, so continue")
                    continue

                # 如果有plugin_path传入则使用传入的plugin_path, 否则使用self.plugins_path
                plugins_path = self.plugins_path
                if plugin_path:
                    plugins_path = plugin_path

                model_file_tar_path = f"{plugins_path}/{model_file_name}"

                # 写入二进制内容到文件里
                with open(model_file_tar_path, "wb+") as file:
                    file.write(model)

                # 解压缩tar文件, 放在对应目录
                tar_file_extract(model_file_tar_path, plugins_path)

                """
                需要实现不同进程之间model文件同步的情况:
                1. on-policy是需要actor/learner进程主动调用的, 不存在多进程争用
                2. off-policy分下面情况:
                2.1 如果是remote模式, 即采用actor进程预测, 则predict进程和model_file_sync进程需要同步
                2.2 如果是local模式, 即采用actor_proxy_local进程预测, 则actor_proxy_local进程和model_file_sync进程需要同步, 采用的是获取的model文件放在进程间队列方式
                2.3 如果是local模式, 即采用aisrv的workflow进程预测, 则workflow进程和model_file_sync进程需要同步, 采用的是获取的model文件放在进程间队列方式
                """
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                    if (
                        CONFIG.remote_agent_default_runtime_mode
                        == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT
                    ):
                        pass

                # 如果有model_path传入则使用传入的model_path, 否则使用self.models_path_new
                if self.current_modes_path_flag == 0:
                    models_path = self.models_path_new
                else:
                    models_path = self.models_path

                if model_path:
                    models_path = model_path

                # 因为下载多个model文件时则需要放在多个文件夹里
                single_model_path = f"{models_path}/{model_name}"

                # 清空models下的文件夹, 采用删除原文件夹, 新增文件夹方式
                clean_dir(single_model_path)

                # 兼容新旧的格式
                model_file_ckpt = KaiwuDRLDefine.KAIWU_MODEL_CKPT

                # 遍历文件夹拷贝文件到models下去
                for root, dirs, file_list in os.walk(plugins_path):
                    for file_name in file_list:
                        file_path = os.path.join(root, file_name)

                        # 需要修改checkpoint内容, 使其在self.modes_path下面能找到引擎文件
                        if KaiwuDRLDefine.CHECK_POINT_FILE == file_name:

                            if (
                                CONFIG.remote_agent_default_runtime_mode
                                == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW
                            ):
                                self.change_checkpoint_file_content(file_path, single_model_path)

                            else:
                                self.rename_checkpoint_file(file_path)

                            shutil.copy(file_path, single_model_path)

                        # 拷贝id_list文件
                        if KaiwuDRLDefine.KAIWU_MODEL_ID_LIST == file_name:
                            shutil.copy(file_path, single_model_path)

                        if f"{model_file_ckpt}" in file_name or KaiwuDRLDefine.CHECK_POINT_FILE == file_name:
                            shutil.copy(file_path, single_model_path)

                            # 因为pytorch和tensorflow的加载形式不一样, 故这里的设置不一样的
                            if (
                                KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH
                                == CONFIG.use_which_deep_learning_framework
                            ):
                                if f"{model_file_ckpt}" in file_name:
                                    current_available_model_files.append(f"{single_model_path}/{file_name}")
                            else:
                                current_available_model_files.append(single_model_path)

                # 删除plugins下的文件, 采用删除原文件夹, 新增文件夹方式
                clean_dir(plugins_path)

                """
                需要实现不同进程之间model文件同步的情况:
                1. on-policy是需要actor/learner进程主动调用的, 不存在多进程争用
                2. off-policy分下面情况:
                2.1 如果是remote模式, 即采用actor进程预测, 则predict进程和model_file_sync进程需要同步
                2.2 如果是local模式, 即采用predict_local进程预测, 则predict_local进程和model_file_sync进程需要同步, 采用的是获取的model文件放在进程间队列方式
                2.3 如果是local模式, 即采用aisrv的workflow进程预测, 则workflow进程和model_file_sync进程需要同步, 采用的是获取的model文件放在进程间队列方式
                """

                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                    # 设置第一个model文件已经被成功拉取下来
                    self.had_pull_the_first_model_file_success.value = True

                self.pull_from_model_pool_succ_cnt += 1

                logger.info(
                    f"model_file_sync pull {model_file_name} from modelpool to {single_model_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}",
                    g_not_server_label,
                )

            if return_model_files:
                return True, current_available_model_files
            else:
                """
                多进程通信需要的操作如下:
                1. 加锁
                2. 更新self.current_available_model_files数据结构, 注意此时需要更新下目录名字
                3. 解锁
                4. 如果本次有, 则才更新下一次写入的models文件夹
                """
                if current_available_model_files:
                    with self.current_available_model_files["lock"]:
                        self.clear_current_available_model_files()
                        for model_file in current_available_model_files:
                            self.current_available_model_files["models"].append(model_file)

                        self.current_modes_path_flag = 1 - self.current_modes_path_flag

                return True, None

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            logger.exception(
                f"model_file_sync pull checkpoint from modelpool error, as {str(e)}, ",
                g_not_server_label,
            )
            return False, None

    # 加载旧模型wight
    def pull_old_wight_from_model_pool(self):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR and CONFIG.svr_name != KaiwuDRLDefine.SERVER_AISRV:
            return False

        try:
            # 拉取wight文件
            model_name = f"{KaiwuDRLDefine.KAIWU_MODEDL_WIGHT}_{CONFIG.app}_{CONFIG.algo}_old"
            model_name_list = self.model_pool_apis.pull_keys()
            """
            获取旧的模型
            1. 如果返回的model_name_list的长度为1, 即进程启动时只有1个wight文件, 则新旧采用同一个wight文件
            2. 如果返回的model_name_list的长度大于1, 则进程启动了并且开始生成多个wight文件, 则采用列表为-2,-1的wight文件
            """
            if model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    return False

                # 因为都是为trt_weights.wts2, 故需要设置下别名带上_old下标
                model_file_name = f"{model_info._file_name}"

                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    return False

                model_file_tar_path = f"{self.convert_model_dir}/{model_file_name}"

                # 写入二进制内容到文件里
                with open(model_file_tar_path, "wb+") as file:
                    file.write(model)

                # 同时计算下文件的md5sum写入到文件里, 便于对账, 不同进程之间对文件的校验

                # 将wight文件拷贝到/data/ckpt目录下
                shutil.copy(
                    model_file_tar_path,
                    f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}",
                )

                with open(
                    f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.FILE_FINISH_OLD_NAME}",
                    "w+",
                ) as file:
                    file.write(KaiwuDRLDefine.FILE_FINISH_OLD_NAME)

                self.pull_from_model_pool_succ_cnt += 1

                self.logger.info(
                    f"model_file_sync pull wight {model_file_name} from modelpool to {self.models_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}",
                    g_not_server_label,
                )
                return True

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            self.logger.exception(
                f"model_file_sync pull weight from modelpool error, as {str(e)}, ",
                g_not_server_label,
            )
            return False

    # 加载旧模型checkpoint
    def pull_old_checkpoint_from_model_pool(self, logger=None, model_path=None, plugin_path=None):
        if CONFIG.svr_name != KaiwuDRLDefine.SERVER_ACTOR and CONFIG.svr_name != KaiwuDRLDefine.SERVER_AISRV:
            return False

        try:
            # 拉取checkpoint的tar文件
            model_name = f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}_{CONFIG.app}_{CONFIG.algo}_old"
            model_name_list = self.model_pool_apis.pull_keys()
            if model_name in model_name_list:
                # 获取model文件名字
                model_info = self.model_pool_apis.pull_model_info(model_name)
                if not model_info:
                    return False

                model_file_name = model_info._file_name
                # 获取model文件内容
                model = self.model_pool_apis.pull_model(model_name)
                if not model:
                    return False

                # 如果有plugin_path传入则使用传入的plugin_path, 否则使用self.plugins_path
                plugins_path = self.plugins_path
                if plugin_path:
                    plugins_path = plugin_path

                model_file_tar_path = f"{plugins_path}/{model_file_name}"

                # 写入二进制内容到文件里
                with open(model_file_tar_path, "wb+") as file:
                    file.write(model)

                # 解压缩tar文件, 放在对应目录
                tar_file_extract(model_file_tar_path, plugins_path)

                """
                需要实现不同进程之间model文件同步的情况:
                1. on-policy是需要actor/learner进程主动调用的, 不存在多进程争用
                2. off-policy分下面情况:
                2.1 如果是remote模式, actor预测, 则predict进程和model_file_sync进程需要同步
                2.2 如果是local模式, actor_proxy_local预测, 则actor_proxy_local进程和model_file_sync进程需要同步
                2.3 如果是local模式, aisrv预测, 则不存在多进程争用
                """
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                    if (
                        CONFIG.remote_agent_default_runtime_mode
                        == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT
                    ):
                        # 获取可写锁文件操作
                        self.get_write_file_lock()

                # 如果有model_path传入则使用传入的model_path, 否则使用self.models_path
                models_path = self.models_path
                if model_path:
                    models_path = model_path

                # 清空models下的文件夹, 采用删除原文件夹, 新增文件夹方式
                clean_dir(models_path)

                # 兼容新旧的格式
                model_file_ckpt = KaiwuDRLDefine.KAIWU_MODEL_CKPT

                # 遍历文件夹拷贝文件到models下去
                for root, dirs, file_list in os.walk(plugins_path):
                    for file_name in file_list:
                        file_path = os.path.join(root, file_name)

                        # 需要修改checkpoint内容, 使其在self.modes_path下面能找到引擎文件
                        if KaiwuDRLDefine.CHECK_POINT_FILE == file_name:
                            if (
                                CONFIG.remote_agent_default_runtime_mode
                                == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_LOCAL_AISRV_WORKFLOW
                            ):
                                self.change_checkpoint_file_content(file_path, models_path)

                            else:
                                self.rename_checkpoint_file(file_path)

                            shutil.copy(file_path, models_path)

                        # 拷贝id_list文件
                        if KaiwuDRLDefine.KAIWU_MODEL_ID_LIST == file_name:
                            shutil.copy(file_path, models_path)

                        if f"{model_file_ckpt}" in file_name or KaiwuDRLDefine.CHECK_POINT_FILE == file_name:
                            shutil.copy(file_path, models_path)

                # 删除plugins下的文件, 采用删除原文件夹, 新增文件夹方式
                clean_dir(plugins_path)

                """
                需要实现不同进程之间model文件同步的情况:
                1. on-policy是需要actor/learner进程主动调用的, 不存在多进程争用
                2. off-policy分下面情况:
                2.1 如果是remote模式, actor预测, 则predict进程和model_file_sync进程需要同步
                2.2 如果是local模式, actor_proxy_local预测, 则actor_proxy_local进程和model_file_sync进程需要同步
                2.3 如果是local模式, aisrv预测, 则不存在多进程争用
                """
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_OFF_POLICY:
                    if (
                        CONFIG.remote_agent_default_runtime_mode
                        == KaiwuDRLDefine.REMOTE_AGENT_RUNTIME_MODE_REMOTE_AISRV_PREDICT
                    ):
                        # 释放写锁文件操作
                        self.release_write_file_lock()

                        # 创建读锁文件操作
                        self.get_read_file_lock()

                        # 设置第一个model文件已经被成功拉取下来
                        self.had_pull_the_first_model_file_success.value = True

                        # 等待读取文件被删除
                        while os.path.exists(self.lock_read_file):
                            time.sleep(CONFIG.idle_sleep_second)

                        # 重置计数器
                        with self.actor_predict_count_value.get_lock():
                            self.actor_predict_count_value.value = 0

                self.pull_from_model_pool_succ_cnt += 1

                logger.info(
                    f"model_file_sync pull {model_file_name} from modelpool to {models_path} success \
                                total pull from modelpool succ cnt is {self.pull_from_model_pool_succ_cnt} \
                                total pull from modelpool err cnt is {self.pull_from_model_pool_err_cnt}",
                    g_not_server_label,
                )
                return True

        except Exception as e:
            self.pull_from_model_pool_err_cnt += 1
            logger.exception(
                f"model_file_sync pull from modelpool error, as {str(e)}, ",
                g_not_server_label,
            )
            return False

    def run_once(self):
        """
        启动定时器操作, 定时器里执行具体的操作
        1. learner --> modelpool, push
        2. modelpool --> actor, pull
        """
        # schedule.run_pending()

        now = time.time()
        # 两个独立的 if，避免 elif 导致时间戳更新延迟
        if now - self.last_run_schedule_time_by_push_pull >= CONFIG.model_file_sync_per_minutes * 60:
            self.last_run_schedule_time_by_push_pull = now
            self.push_and_pull_model_file()

        if now - self.last_run_schedule_time_by_sync_stat >= CONFIG.prometheus_stat_per_minutes * 60:
            self.last_run_schedule_time_by_sync_stat = now
            self.model_file_sync_stat()

    def run(self):
        if not self.before_run():
            self.logger.error(f"model_file_sync before_run failed, so return", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(
                    f"model_file_sync failed to run {self.name} trainer. exit. ",
                    g_not_server_label,
                )
                break
