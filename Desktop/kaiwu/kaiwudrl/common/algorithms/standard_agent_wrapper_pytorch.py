#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import glob
import numpy as np
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.torch_utils import *
from kaiwudrl.common.utils.common_func import TimeIt

from kaiwudrl.common.checkpoint.model_file_common import (
    before_save_model,
    after_save_model,
    update_id_list,
    check_path_id_valid,
    find_id_in_id_list,
)


class StandardAgentWrapperPytorch:
    """
    StandardAgentWrapperPytorch类, aisrv, actor, learner都会使用, 不同的进程采用不同的方法
    """

    def __init__(self, agent, logger, server=None) -> None:
        self.agent = agent
        self.logger = logger

        # 统计值
        self.train_count = 0
        self.preload_model_train_count = 0
        self.predict_count = 0
        self.load_model_count = 0

        # 按照频率来保存控制参数
        self.save_model_count = 0
        # 按照总数来控制参数
        self.save_model_all_count = 0

        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            self.local_rank = 0

        # 主learner
        self.is_chief = CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER and self.local_rank == 0

        # 最后一次保存模型的时间
        self.last_save_model_time = 0

        # 进程启动时间
        self.process_start_time = time.monotonic()

        # 给业务设置下日志接口
        self.set_logger()

        # 因为pytorch需要用户确保保存最多多少model文件
        self.file_queue = []

        # 被删除的文件列表, 主要是用于对账使用, 每隔100个打印下列表
        self.delete_files = []
        self.delete_file_count = 0

        # on-policy情况下过滤样本列表和数量
        self.filter_sample_count = 0
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.filter_sample_list = []

        # 保存文件成功的最后id, 即如果该id已经保存
        self.last_success_save_model_id = -1

        # 获取数据和真实训练的耗时, 单位秒
        self.data_fetch_cost_time = 0
        self.real_train_cost_time = 0

    def should_stop(self):
        # 需要业务提供下该方法
        # return self.agent.should_stop()
        return False

    def set_logger(self):
        # 由于已经在init时传递了logger对象, 故这里不需要再传递
        if hasattr(self.agent, "set_logger"):
            self.agent.set_logger(self.logger)

    def close(self):
        if hasattr(self.agent, "stop"):
            return self.agent.stop()

    def before_train(self):
        pass

    def add_file_to_queue(self):
        id = self.train_count

        # 采用模糊匹配的方法来操作
        model_file_names = glob.glob(
            f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{id}.*"
        )
        if not model_file_names:
            return

        self.file_queue.append(model_file_names)
        if len(self.file_queue) >= CONFIG.max_save_model_file_count:
            model_file_names_to_delete = self.file_queue.pop(0)
            if model_file_names_to_delete:
                for to_model_file_name in model_file_names_to_delete:
                    if os.path.exists(to_model_file_name):
                        os.remove(to_model_file_name)
                        self.delete_files.append(to_model_file_name)
                        self.delete_file_count += 1

        # 减少日志打印, 避免日志打印的比较多, 导致其他日志无法正常打印
        if self.delete_file_count % 100 == 0:
            # self.logger.info(f"model file {self.delete_files} deleted success")
            self.delete_files.clear()
            self.delete_file_count = 0

    def after_train(self):
        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False
        self.train_count += 1
        if self.train_count % CONFIG.dump_model_freq == 0:
            if getattr(CONFIG, f"{CONFIG.svr_name}_device_type") == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
                torch.npu.set_device(torch.device(self.agent.model.device))

            # 框架落模型文件
            self.save_param_by_source(
                path=f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/",
                id=self.train_count,
                source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK,
            )

            # 放入队列控制占用大小以免磁盘无限增加被驱逐
            self.add_file_to_queue()

            # 维护id_list列表
            update_id_list(self.train_count, framework=True)

            has_model_file_changed = True

        return has_model_file_changed, self.train_count

    def before_save_param(self):
        # 保存模型前的操作
        before_save_model()

    def after_save_param(self, id):
        """
        业务侧调用生成model文件后, 需要做下面工作:
        1. 生成json文件
        2. 生成tar.gz文件
        3. 清空类似下面的文件/data/user_ckpt_dir/gorge_walk_dp, 这样会导致该目录下只是保存最新的model文件, 历史的采用tar.gz包放置
        """
        after_save_model(self.process_start_time, id)

    def do_save_param(self, path, id):
        """
        保存模型文件
        """

        # 保存模型前的操作
        self.before_save_param()

        # 直接调用业务层的 save_model（带标记）
        self.agent.save_model(path=path, id=id, framework=True)

        # 保存模型后的操作
        self.after_save_param(id)

        self.save_model_count += 1
        self.save_model_all_count += 1

    def save_param_by_source(self, path=None, id=None, source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK):
        """
        统一的保存模型函数，支持三种 source:
        1. FRAMEWORK: 框架内部调用，直接保存，不需要频率限制
        2. SIGTERM: 优雅退出时调用，直接保存，不需要频率限制
        3. USER: 用户调用，需要频率限制、次数限制等检查
        """
        if getattr(CONFIG, f"{CONFIG.svr_name}_device_type") == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
            torch.npu.set_device(torch.device(self.agent.model.device))

        # id取值为self.train_count
        id = self.train_count

        # 根据 source 类型选择不同的处理方式
        if source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK:
            # 框架内部调用：直接调用业务层方法，不需要频率限制等检查
            path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}"
            # 直接调用业务层（带标记）
            self.agent.save_model(path=path, id=id, framework=True)

        elif source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_BY_SIGTERM:
            path = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}"
            # 优雅退出时如果id为0, 即一步也没有训练成功则不需要保存模型文件, 该模型文件也是随机的
            if not id:
                self.logger.info(f"train_step is 0, so not save_model")
            else:
                # 优雅退出时主动调用一次用户侧保存模型函数，不需要频率限制等检查
                if not find_id_in_id_list(id, framework=False):
                    # 直接调用业务层（带标记）
                    self.do_save_param(path, id)
                else:
                    self.logger.info(f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{id} already exists")

        elif source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER:
            # 非框架调用下默认的保存目录和框架已经有的文件目录不一样
            path = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}"

            try:
                # id为0时, 即一步也没有训练成功, 在平台时不需要保存模型
                if not id:
                    if CONFIG.deployment_platforms != KaiwuDRLDefine.DEPLOYMENT_PLATFORMS_CLIENT:
                        self.logger.info(f"train_step is 0, so not save_model")
                        return

                if self.last_success_save_model_id == int(id):
                    self.logger.info(
                        f"self.last_success_save_model_id is {self.last_success_save_model_id}, as the same as {id}, so return"
                    )
                    return

                if CONFIG.user_save_mode_max_count > 0:
                    if self.save_model_all_count >= CONFIG.user_save_mode_max_count:
                        self.logger.error(
                            f" save_param_by_source() self.save_model_all_count {self.save_model_all_count} "
                            f"> CONFIG.user_save_mode_max_count {CONFIG.user_save_mode_max_count}, "
                            f"please check your code for any error"
                        )
                        # 在不保存文件的基础上还需要增加该计数, 目的是为用户提示
                        self.save_model_all_count += 1
                        return

                if CONFIG.user_save_model_max_frequency_per_min > 0:
                    # 获取当前时间
                    current_time = time.time()
                    if current_time - self.last_save_model_time >= 60:
                        self.save_model_count = 0
                        self.last_save_model_time = current_time

                    if self.save_model_count <= CONFIG.user_save_model_max_frequency_per_min:
                        self.do_save_param(path, id)
                        self.last_success_save_model_id = id
                    else:
                        self.logger.error(
                            f" save_param_by_source() user_save_model_max_frequency_per_min > "
                            f"CONFIG.user_save_model_max_frequency_per_min "
                            f"{CONFIG.user_save_model_max_frequency_per_min}, so return "
                        )
                else:
                    self.do_save_param(path, id)
                    self.last_success_save_model_id = id

            except RuntimeError:
                self.logger.exception(f" save_param_by_source() RuntimeError Exception")
                raise RuntimeError(f" save_param_by_source() RuntimeError Exception")
            except Exception as e:
                self.logger.exception(f" save_param_by_source() Exception {str(e)}")
                raise RuntimeError(f" save_param_by_source() Exception {str(e)}")
        else:
            self.logger.exception(f"save_param_by_source() un support source: {source}")
            raise RuntimeError(f"save_param_by_source() un support source: {source}")

    def before_predict(self, predict_data):
        return isinstance(predict_data, dict)

    def after_predict(self, batch_size):
        self.predict_count += batch_size

    # predict/exploit 需要计数
    _COUNTING_OBS_METHODS = {"predict", "exploit"}

    def _obs_dispatch(self, method_name, data):
        """obs 类方法的统一分发实现

        - predict/exploit: 包含 before_predict + update_predict_count + after_predict(计数)
        - reset/init_config: 纯透传
        """
        try:
            if method_name in self._COUNTING_OBS_METHODS:
                self.before_predict(data)

                if hasattr(self.agent, "update_predict_count"):
                    self.agent.update_predict_count(self.predict_count)

            values = getattr(self.agent, method_name)(data, framework=True)

            if method_name in self._COUNTING_OBS_METHODS:
                batch_size = len(data) if isinstance(data, list) else 1
                self.after_predict(batch_size)

            return values

        except RuntimeError:
            self.logger.exception(f"{method_name}() RuntimeError Exception")
            return None

        except Exception as e:
            self.logger.exception(f"{method_name}() Exception {str(e)}")
            return None

    def predict(self, predict_data):
        return self._obs_dispatch("predict", predict_data)

    def exploit(self, predict_data):
        return self._obs_dispatch("exploit", predict_data)

    def reset(self, reset_data):
        return self._obs_dispatch("reset", reset_data)

    def init_config(self, config_data):
        return self._obs_dispatch("init_config", config_data)

    # train函数, 单机单进程版本调用
    def train_local(self, data, extra_tensors=None):
        try:
            self.before_train()

            # 具体的训练流程
            values = self.agent.learn(data, framework=True)

            # 返回是否更新了model文件, 更新的model文件的ID
            has_model_file_changed, model_file_id = self.after_train()

            return values, has_model_file_changed, model_file_id

        except RuntimeError:
            self.logger.exception(f" train_local() RuntimeError Exception")
            return None, None, None

        except Exception as e:
            self.logger.exception(f" train_local() Exception, {str(e)}")
            return None, None, None

    # train 函数, 集群版本调用
    def train(self, current_sync_model_version_from_learner=-1):
        try:
            self.before_train()

            train_success = True

            # 获取数据阶段(含on-policy过滤), 统计获取数据耗时
            with TimeIt() as ti_fetch:
                data = self.get_data_from_replay_buffer()

                """
                在on-policy的情况下, 进行样本过滤, 规则如下:
                1. 只保留等于current_sync_model_version_from_learner的样本
                2. 满足batch_size的才去训练, 否则需要等待batch_size个样本
                """
                if data is not None and CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                    # data 是一个 (batch_size, data_len) 的tensor
                    batch_tensor = data

                    # 原始的数据量
                    origin_sample_count = batch_tensor.shape[0] if hasattr(batch_tensor, "shape") else len(batch_tensor)

                    # 过滤：保留model_version匹配的样本
                    # batch_tensor[:, -1] 是最后一列，即model_version
                    if isinstance(batch_tensor, torch.Tensor):
                        # PyTorch tensor情况 - 高性能向量化操作
                        mask = batch_tensor[:, -1].long() == current_sync_model_version_from_learner
                        filtered_tensor = batch_tensor[mask]  # (filtered_count, data_len)
                        filter_sample_count = filtered_tensor.shape[0]

                        # 累积到缓冲区
                        self.filter_sample_list.append(filtered_tensor)

                        # 统计被过滤掉的样本数
                        self.filter_sample_count += origin_sample_count - filter_sample_count

                        # 计算当前累积的样本总数
                        accumulated_count = sum(t.shape[0] for t in self.filter_sample_list)

                        # 样本不足，标记data为None
                        if accumulated_count < CONFIG.train_batch_size:
                            data = None
                        else:
                            # 样本充足，进行训练
                            # 拼接所有tensor，然后取前train_batch_size个样本
                            all_samples = torch.cat(self.filter_sample_list, dim=0)  # 高效拼接
                            data = all_samples[: CONFIG.train_batch_size]  # 切片取batch_size

                            # 保留剩余样本
                            remaining = all_samples[CONFIG.train_batch_size :]
                            if remaining.shape[0] > 0:
                                self.filter_sample_list = [remaining]
                            else:
                                self.filter_sample_list = []

                    else:
                        # numpy array或其他情况（保持向后兼容）
                        # 优化：转为tensor后统一处理
                        if isinstance(batch_tensor, np.ndarray):
                            batch_tensor = torch.from_numpy(batch_tensor).float()

                        # 向量化过滤
                        mask = batch_tensor[:, -1].long() == current_sync_model_version_from_learner
                        filtered_tensor = batch_tensor[mask]
                        filter_sample_count = filtered_tensor.shape[0]

                        # 累积到缓冲区
                        self.filter_sample_list.append(filtered_tensor)

                        # 统计被过滤掉的样本数
                        self.filter_sample_count += origin_sample_count - filter_sample_count

                        # 计算当前累积的样本总数
                        accumulated_count = sum(t.shape[0] for t in self.filter_sample_list)

                        # 样本不足，标记data为None
                        if accumulated_count < CONFIG.train_batch_size:
                            data = None
                        else:
                            # 样本充足，进行训练
                            all_samples = torch.cat(self.filter_sample_list, dim=0)
                            data = all_samples[: CONFIG.train_batch_size]

                            # 保留剩余样本
                            remaining = all_samples[CONFIG.train_batch_size :]
                            if remaining.shape[0] > 0:
                                self.filter_sample_list = [remaining]
                            else:
                                self.filter_sample_list = []

            self.data_fetch_cost_time = ti_fetch.interval

            # data为None说明获取数据失败或on-policy样本不足, 提前返回
            if data is None:
                self.real_train_cost_time = 0
                return train_success, None, False, -1

            # 真实训练阶段, 统计训练耗时
            with TimeIt() as ti_train:
                values = self.agent.learn(data, framework=True)
            self.real_train_cost_time = ti_train.interval

            # 返回是否更新了model文件, 更新的model文件的ID
            has_model_file_changed, model_file_id = self.after_train()

            return train_success, values, has_model_file_changed, model_file_id

        except RuntimeError:
            self.logger.exception(f" train() RuntimeError Exception")
            train_success = False
            return train_success, None, False, -1

        except Exception as e:
            self.logger.exception(f" train() Exception, {str(e)}")
            train_success = False
            return train_success, None, False, -1

    def get_global_step(self):
        return self.train_count

    @property
    def train_stat(self):
        return self.train_count, self.preload_model_train_count

    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def load_model_stat(self):
        return self.load_model_count

    @property
    def name(self):
        return "StandardModelWrapperPytorch"

    @property
    def tf_sess(self):
        return self.sess

    def load_model_by_source(self, path=None, id="1", source=KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK):
        """
        统一的加载模型函数，支持三种 source:
        1. FRAMEWORK: 框架内部调用（需要根据 run_mode 判断是否加载）
        2. USER: 用户调用（带 load_model 标记）
        3. SIGTERM: 暂不使用（保留用于未来扩展）

        Args:
            path: 模型路径
            id: 模型ID
            source: 调用来源

        Returns:
            bool/None: FRAMEWORK 模式返回 bool，USER 模式返回 None
        """
        try:
            # 根据 source 类型选择不同的处理方式
            if source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK:
                # 框架调用：需要根据 run_mode 判断是否加载
                is_to_load_model = False

                if CONFIG.run_mode in [
                    KaiwuDRLDefine.RUN_MODE_EVAL,
                    KaiwuDRLDefine.RUN_MODE_EXAM,
                ]:
                    is_to_load_model = True
                elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
                    if int(CONFIG.preload_model):
                        is_to_load_model = True

                    # on_policy情况下, KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK是主动加载的
                    if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                        is_to_load_model = True

                if not is_to_load_model:
                    return False

                # 判断参数是否合法
                if not check_path_id_valid(path, id):
                    self.logger.error(f"load_model_by_source from models_path {path}, id {id} failed, please check")
                    return False

                # 直接调用业务侧的 load_model（不带标记）
                self.agent.load_model(path=path, id=id, framework=True)
                self.load_model_count += 1
                self.logger.info(
                    f"{CONFIG.run_mode} mode load_model_by_source from models_path {path}, "
                    f"checkpoint_id {id} success"
                )
                return True

            elif source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_USER:
                # 用户调用：直接调用业务层方法（带 load_model 标记）
                if CONFIG.run_mode not in [
                    KaiwuDRLDefine.RUN_MODE_TRAIN,
                    KaiwuDRLDefine.RUN_MODE_EVAL,
                    KaiwuDRLDefine.RUN_MODE_EXAM,
                ]:
                    return False

                self.agent.load_model(path=path, id=id, framework=True)
                self.load_model_count += 1
                self.logger.info(
                    f"{CONFIG.run_mode} mode load_model_by_source from models_path {path}, "
                    f"checkpoint_id {id} success"
                )
                return True

            else:
                # 其他 source（如 SIGTERM）暂不支持
                self.logger.warning(f"load_model_by_source: unsupported source {source}")
                return False

        except RuntimeError:
            self.logger.exception(f"load_model_by_source() RuntimeError Exception")
            if source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK:
                return False
            else:
                raise RuntimeError(f"load_model_by_source() RuntimeError Exception")

        except Exception as e:
            self.logger.exception(f"load_model_by_source() Exception {str(e)}")
            if source == KaiwuDRLDefine.SAVE_OR_LOAD_MODEL_By_FRAMEWORK:
                return False
            else:
                raise RuntimeError(f"load_model_by_source() Exception {str(e)}")

    def preload_model_file(self, preload_model_dir, preload_model_id):
        """
        预加载模型文件, 直接调用业务类, 步骤如下:
        1. 不需要清空以前的checkpoint文件, 因为以前的checkpoint文件会被很快覆盖掉
        2. 调用业务类的load_model
        3. 调用业务类的save_model
        """
        if not check_path_id_valid(preload_model_dir, preload_model_id):
            self.logger.error(
                f"preload_model_file failed, but preload_model_dir {preload_model_dir} or "
                f"preload_model_id {preload_model_id} not valid, please check"
            )
            return False

        try:
            # 调用业务的load_model，框架内部直接调用业务层方法
            self.agent.load_model(path=preload_model_dir, id=preload_model_id, framework=True)
            self.train_count = preload_model_id

            # 需要记录下预加载时设置的已经训练的次数, 用于计算样本生成消耗比, 否则会导致实际的值偏高
            self.preload_model_train_count = preload_model_id

            self.logger.info(f" preload_model_file success, path is {preload_model_dir}, id is {preload_model_id}")
            return True
        except Exception as e:
            self.logger.exception(f" preload_model_file() Exception {str(e)}")
            return False

    def set_dataset(self, replay_buffer_wrapper):
        self.replay_buffer_wrapper = replay_buffer_wrapper

    def is_chief(self):
        return self.is_chief

    def get_model_object(self):
        return self.agent

    def get_data_from_replay_buffer(self):
        # 采用pytorch方法获取样本数据
        return self.replay_buffer_wrapper.dataset_from_generator_by_pytorch()
