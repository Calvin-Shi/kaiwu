#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import threading
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
from kaiwudrl.common.utils.common_func import get_host_ip


# 因为下面的ReverbDataset用到了torch故这里需要专门import下
if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
    if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
        # 根据配置动态加载下需要的类
        if CONFIG.pytorch_read_data_from_reverb_type == 1:
            from kaiwudrl.common.replay_buffer.reverb_dataset_v1 import ReverbDataset
        elif CONFIG.pytorch_read_data_from_reverb_type == 2:
            from kaiwudrl.common.replay_buffer.reverb_dataset_v2 import ReverbDataset
        else:
            from kaiwudrl.common.replay_buffer.reverb_dataset_v1 import ReverbDataset


# 定义一个类似于 TensorFlow 的 TensorSpec 的类
class TensorSpec:
    def __init__(self, dtype, name):
        self.dtype = dtype
        self.name = name

    def __repr__(self):
        return f"TensorSpec(dtype={self.dtype}, name={self.name})"


class ReplayBufferWrapper(object):
    def __init__(self, tensor_names, tensor_dtypes, logger=None):
        self._tensor_names = tensor_names
        self._tensor_dtypes = tensor_dtypes
        self._sorted_names = None
        self._sorted_dtypes = None

        # 针对replaybuffer 统计信息
        self.proc_sample_cnt = 0
        self.skip_sample_cnt = 0

        # 在使用pytorch场景里, 从reverb里获取数据时的ReverbDataset对象, 全局唯一
        self.reverb_dataset = None

        self.logger = logger

    def init(self, mem_buffer=None):
        """
        初始化 replay buffer

        Args:
            mem_buffer: 可选的共享 MemBuffer 对象，用于跨进程共享（仅 ZMQ 类型）
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            from kaiwudrl.common.replay_buffer.reverb_replay_buffer import (
                ReverbReplayBuffer,
            )

            # 采用pytorch方式
            if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
                data_spec = None
            # 采用tensorflow方式或者其他
            else:
                data_spec = tuple(
                    [tf.TensorSpec(dtype, name) for name, dtype in zip(*self.sorted_tensor_spec_tensorflow())]
                )

            self._replay_buffer = ReverbReplayBuffer(data_spec)
            self._replay_buffer.init()

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            self._replay_buffer = NotImplemented
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:

            # 采用pytorch方式
            if KaiwuDRLDefine.DEEP_LEARNING_FRAMEWORK_PYTORCH == CONFIG.use_which_deep_learning_framework:
                data_spec = None
            # 采用tensorflow方式或者其他
            else:
                data_spec = tuple(
                    [tf.TensorSpec(dtype, name) for name, dtype in zip(*self.sorted_tensor_spec_tensorflow())]
                )

            from kaiwudrl.common.replay_buffer.zmq_replay_buffer import ZmqReplayBuffer

            # 创建 ZmqReplayBuffer，传入 mem_buffer（如果提供）
            self._replay_buffer = ZmqReplayBuffer(data_spec, self.logger, mem_buffer=mem_buffer)

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            from kaiwudrl.common.replay_buffer.shared_memory_replay_buffer import SharedMemoryReplayBuffer

            ip = get_host_ip()
            self._replay_buffer = SharedMemoryReplayBuffer(
                base_name=KaiwuDRLDefine.SHARED_MEMORY_NAME,
                is_producer=False,
                producer_identifier=ip,
                discovery_interval=10.0,
                size=4 * 1024 * 1024,
                create=False,
                unlink_on_close=False,
                logger=self.logger,
            )
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_FILE_MMAP:
            from kaiwudrl.common.replay_buffer.file_mmap_replay_buffer import (
                FileMMAPReplayBuffer,
            )

            self._replay_buffer = FileMMAPReplayBuffer(CONFIG.file_mmap_name, self.logger)
        else:
            raise ValueError("ReplayBuffer currently only support reverb or tf_uniform or zmq!")

        self.logger.info(f"train replay_buff, use {CONFIG.replay_buffer_type}")

    def init_with_shared_mem_buffer(self, shared_mem_buffer):
        """
        使用共享的 mem_buffer 进行初始化，用于跨进程共享 replay buffer

        Args:
            shared_mem_buffer: 从 trainer 进程传递过来的共享 MemBuffer 对象
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            # 直接调用 init() 方法，传入共享的 mem_buffer
            self.init(mem_buffer=shared_mem_buffer)
            self.logger.info("ReplayBufferWrapper initialized with shared mem_buffer for cross-process communication")
        else:
            # 其他类型的 replay buffer 不支持此方法
            raise ValueError(
                f"init_with_shared_mem_buffer only supports ZMQ replay buffer type, current type: {CONFIG.replay_buffer_type}"
            )

    def get_mem_buffer(self):
        """
        获取底层的 mem_buffer 对象，用于跨进程共享

        Returns:
            底层的 MemBuffer 对象（仅在 ZMQ 类型时有效）
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.mem_buffer
        else:
            return None

    def train_hooks(self, local_step_tensor=None):
        return []

    # 定义一个方法来生成自定义的 TensorSpec
    def sorted_tensor_spec_pytorch(self):
        if self._sorted_names is None or self._sorted_dtypes is None:
            tensor_infos = list(zip(self._tensor_names, self._tensor_dtypes))
            sorted_tensors_infos = sorted(tensor_infos, key=lambda x: x[0])
            tmp_uniq_names, names, dtypes = set(), [], []
            # uniq
            for item in sorted_tensors_infos:
                if item[0] not in tmp_uniq_names:
                    tmp_uniq_names.add(item[0])
                    names.append(item[0])
                    dtypes.append(item[1])
            for i, name in enumerate(list(zip(names))):
                only_keep_first = True if CONFIG.use_rnn and name in CONFIG.rnn_states else False
                self.logger.info(f"train tensor spec: {name}")

            # Replay buffer hooker needs `step` to filter expired samples.
            if CONFIG.replay_buffer_type == "TF_UNIFORM":  # 你需要定义这个常量
                names += ["s"]
                dtypes += [torch.int64]

            self._sorted_dtypes = dtypes
            self._sorted_names = names

        return self._sorted_names, self._sorted_dtypes

    def sorted_tensor_spec_tensorflow(self):
        if self._sorted_names is None or self._sorted_dtypes is None:
            tensor_infos = list(zip(self._tensor_names, self._tensor_dtypes))
            sorted_tensors_infos = sorted(tensor_infos, key=lambda x: x[0])
            tmp_uniq_names, names, dtypes = set(), [], []
            # uniq
            for item in sorted_tensors_infos:
                if item[0] not in tmp_uniq_names:
                    tmp_uniq_names.add(item[0])
                    names.append(item[0])
                    dtypes.append(item[1])

            for i, name in enumerate(list(zip(names))):
                only_keep_first = True if CONFIG.use_rnn and name in CONFIG.rnn_states else False
                self.logger.info(f"train tensor spec: {name}")

            # Replay buffer hooker needs `step` to filter expired samples.
            if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
                names += ["s"]
                dtypes += [tf.int64]

            self._sorted_dtypes = dtypes
            self._sorted_names = names

        return self._sorted_names, self._sorted_dtypes

    def dataset_from_generator(self):
        """
        dataset_from_generator
        1. reverb里, 采用dataset.from_generator来进行构造数据, 获取到具体数据, 再进行run_session
        2. zmq里, 共享内存的不做操作
        """
        if (
            CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB
            or CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM
        ):
            if CONFIG.reverb_data_cache:
                dataset = self._replay_buffer.as_dataset_by_cache()
            else:
                dataset = self._replay_buffer.as_dataset()

            self._dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)

            if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
                next_tensors = self._dataset_iter.get_next()[1]
            elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
                next_tensors = self._dataset_iter.get_next()
            else:
                assert False

            return next_tensors

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            assert False

        else:
            assert False

    def dataset_from_generator_by_pytorch(self):
        """
        该方案从reverb里, 采用遍历方式获取到数据
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            # ReverbDataset对象只是第一次调用时创建
            if self.reverb_dataset is None:
                self.reverb_dataset = ReverbDataset(self.logger)

            datas = next(iter(self.reverb_dataset))

            return datas

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.next_by_batch_size()

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            return self._replay_buffer.recv_batch(CONFIG.train_batch_size)

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_FILE_MMAP:
            return self._replay_buffer.read_from_mmap_file()

        else:
            pass

    def input_tensors(self):
        """
        该方案是采用tf.compat.v1.placeholder_with_default占位符 + 业务自定义网络结构生成的流水线设计, 推荐
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            pass

        if CONFIG.reverb_data_cache:
            dataset = self._replay_buffer.as_dataset_by_cache()
        else:
            dataset = self._replay_buffer.as_dataset()

        self._dataset_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            next_tensors = self._dataset_iter.get_next()[1]
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            next_tensors = self._dataset_iter.get_next()
        else:
            assert False

        tensors = [
            (
                tf.compat.v1.placeholder_with_default(d, shape=[None, None] + d.get_shape().as_list()[2:])
                if CONFIG.use_rnn
                else tf.compat.v1.placeholder_with_default(d, shape=[None] + d.get_shape().as_list()[1:])
            )
            for d in next_tensors
        ]

        return dict(zip(self._sorted_names, tensors))

    def extra_initializer_ops(self):
        return [self._dataset_iter.initializer]

    def extra_threads(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            self._reverb_server = self._replay_buffer.build_reverb_server()
            self._reverb_client = self._replay_buffer.build_reverb_client()

            """
            server的wait是常驻线程
            """

            def start_reverb_server():
                self._reverb_server.wait()

            thread = threading.Thread(target=start_reverb_server)
            thread.daemon = True
            thread.start()

        else:
            pass

    def reset(self, step, tf_sess):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            self._replay_buffer.clear(self._reverb_client, step)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            tf_sess.run(self._tf_replay_buffer_clear)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            pass
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            pass
        else:
            assert False

    def input_ready(self, tf_sess):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            current_size = self._replay_buffer.total_size(self._reverb_client)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            # 这里暂时没有实现
            current_size = tf_sess.run(self._tf_replay_buffer_total_size)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            current_size = 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            current_size = 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_FILE_MMAP:
            current_size = 0
        else:
            assert False

        # self.logger.debug(f"train current_size: {current_size}, CONFIG.train_batch_size: {CONFIG.train_batch_size}")
        return current_size >= int(CONFIG.train_batch_size)

    def add_sample(self, sample):
        """
        新增1条记录, 在zmq场景下使用, 即朝共享内存插入单条记录
        """
        if not sample:
            return False

        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.add_sample(sample)
        else:
            assert False

    # 获取样本接收速度
    def get_recv_speed(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            return 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            return 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.get_insert_speed()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            return 0
        else:
            assert False

    # 获取目前样本池里的数目
    def get_current_size(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            return self._replay_buffer.total_size(self._reverb_client)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            # 这里暂时没有实现
            return 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return CONFIG.replay_buffer_capacity
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            return CONFIG.replay_buffer_capacity
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_FILE_MMAP:
            return CONFIG.replay_buffer_capacity

        else:
            assert False

    # 获取目前样本池里插入的数目
    def get_insert_stats(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            return self._replay_buffer.insert_stats(self._reverb_client)
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.get_insert_speed()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            return self._replay_buffer.get_insert_speed()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            # 这里暂时没有实现
            return 0
        else:
            assert False

    # 获取ReplayBuffer的监控情况
    def get_replay_buffer_monitor(self):
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            if self.reverb_dataset is None:
                return None

            return self.reverb_dataset.get_metrics()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_TF_UNIFORM:
            # 这里暂时没有实现
            return 0
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_SHARED_MEMORY:
            return self._replay_buffer.get_monitor_info()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            return self._replay_buffer.get_insert_speed()
        else:
            assert False
