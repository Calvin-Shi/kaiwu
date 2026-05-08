#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from common_python.config.config_control import CONFIG
from kaiwudrl.common.config.cluster_conf import ClusterConf

# from kaiwudrl.common.utils.common_func import get_local_rank, get_local_size


# rerf: https://github.com/horovod/horovod, need pip install horovod, gcc 7.3.1
try:
    import horovod.tensorflow as hvd
except ImportError:
    hvd = None


class StandardAgentWrapperTensorRT:
    """
    StandardAgentWrapperTensorRT, actor和learner都会使用, aisrv, actor, learner都会使用, 不同的进程采用不同的方法
    主要是加载业务的实现
    """

    def __init__(self, agent, logger, server=None) -> None:

        self.agent = agent
        self.logger = logger
        self.chief_only_hooks = []
        self.train_hooks = []
        self.predict_hooks = []

        # mpi_local_rank = get_local_rank()

        # 加载TensorFlow需要的配置文件
        """
        self.cluster_conf = ClusterConf(
        CONFIG.learner_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.actor_ip_addrs.split(','), # 注意配置项是字符串
        CONFIG.learner_grpc_ports.split(','),
        CONFIG.actor_grpc_ports.split(','),
        # 下面配置项目每个进程启动时, 进程配置文件加载
        CONFIG.svr_name,
        CONFIG.svr_index,
        CONFIG.svr_ports,
        mpi_local_rank
        )
        """

        """
        根据加载的配置文件里进程不同名字来处理, 主要是actor和learner
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            hvd.init()
            self.local_rank = hvd.local_rank()

        # 主learner
        self.is_chief = CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER and hvd.rank() == 0

        # learner_device
        self.learner_device = "/job:learner/task:0"

        # actor_device
        # self.actor_device = '/job:localhost/task:%d' % self.cluster_conf.task_index

        # config
        cpu_num = int(CONFIG.cpu_num)
        self.config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            intra_op_parallelism_threads=cpu_num,
            inter_op_parallelism_threads=cpu_num,
        )

        self.sess = None
        self.local_step = -1
        self.global_step = -1

        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            self.config.gpu_options.allow_growth = True
            self.cached_local_step = -1

        """
        目前KaiwuDRL通信情况:
        1. 采用的horovod进行learner间的通信
        2. 采用modelpool进行learner/actor之间的模型同步
        """

        """
        if not server:
            self.server = tf.distribute.Server(
                    self.cluster_conf.cluster_spec(),
                    job_name=self.cluster_conf.job_name,
                    task_index=self.cluster_conf.task_index,
                    config=self.config)
        else:
            self.server = server
        """

        self.predict_count = 0
        self.train_count = 0

        # 给业务设置下日志接口
        self.set_logger()

    # 设置predict input_tensors入口函数
    def build_predict_input_tensors(self, input_fn, *args, **kwargs):

        with tf.device(f"{self.actor_device}/cpu:0"):
            input_tensors = input_fn(*args, **kwargs)

        with tf.compat.v1.variable_scope("%s/global" % self.agent.name):
            self.global_spec = self.agent.build_model("predict", input_tensors)

        with tf.compat.v1.variable_scope("%s/local" % self.agent.name):
            self.local_spec = self.agent.build_model("predict", input_tensors)

    # 直接调用业务类
    def build_model(self):
        with tf.Graph().as_default():
            self.agent.build_model()

            self.agent.init_model()

    def set_dataset(self, dataset):
        self.agent.set_dataset(dataset)

    def should_stop(self):
        return self.agent.should_stop()

    def set_logger(self):
        # 由于已经在init时传递了logger对象, 故这里不需要再传递
        pass
        # self.agent.set_logger(self.logger)

    def close(self):
        if hasattr(self.agent, "stop"):
            return self.agent.stop()

    def before_train(self):
        pass

    def after_train(self):
        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False
        self.train_count += 1

        return has_model_file_changed, -1

    # 校验需要检测的数据格式
    def before_predict(self, predict_data):
        return isinstance(predict_data, dict)

    # 计数
    def after_predict(self, batch_size):
        self.predict_count += batch_size

    # train 函数, 调用业务类的train函数
    def train(self):
        self.before_train()

        train_success = True

        try:
            with tf.Graph().as_default():
                values = self.agent.train()

            has_model_file_changed, model_file_id = self.after_train()

            return train_success, values, has_model_file_changed, model_file_id
        except Exception as e:
            self.logger.exception(f" train() Exception, {str(e)}")
            train_success = False
            return train_success, None, False, -1

    # predict函数, 调用业务类的predict函数
    def predict(self, predict_data, batch_size):

        values = None
        if self.before_predict(predict_data):

            # 部分场景需要更新predict_count
            if hasattr(self.agent, "update_predict_count"):
                self.agent.update_predict_count(self.predict_count)

            values = self.agent.predict(predict_data)

            self.after_predict(batch_size)

        return values

    def predict_pipeline(self):

        values = self.agent.predict_pipeline()

        self.after_predict()

        return values

    def add_chief_only_hooks(self, hooks):
        if hvd.rank() == 0 and hooks:
            self.chief_only_hooks.extend(hooks)

    def add_train_hooks(self, hooks):
        if hooks:
            self.train_hooks.extend(hooks)

    def add_predict_hooks(self, hooks):
        if hooks:
            self.predict_hooks.extend(hooks)

    def get_local_step(self):
        return self.sess._tf_sess().run(self.local_step)

    # 直接返回业务定义的
    def get_global_step(self):
        return self.agent.get_global_step()

    @property
    def train_stat(self):
        return self.train_count

    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def tf_sess(self):
        return self.agent.tf_sess

    @property
    def name(self):
        return "ModelWrapperTensorRT"

    def is_chief(self):
        return self.is_chief

    # 直接调用业务类的load_last_new_model
    def load_last_new_model(self, models_path):
        if not models_path:
            return

        return self.agent.load_last_new_model(models_path)
