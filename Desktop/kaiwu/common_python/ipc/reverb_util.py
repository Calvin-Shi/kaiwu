#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import traceback
import reverb


"""
采用配置类方式, 支持多个配置项

使用实例:
config = ReverbUtilConfig(
    reverb_sampler='reverb.selectors.Uniform',
    reverb_client_max_sequence_length=1,
    reverb_client_chunk_length=1,
)

使用方式reverb_utils = ReverbUtil(learner_addr=learner_addr, config=config, logger=logger)

"""


class ReverbUtilConfig:
    def __init__(
        self,
        reverb_sampler="reverb.selectors.Uniform",
        reverb_client_max_sequence_length=1,
        reverb_client_chunk_length=1,
    ):
        self.reverb_sampler = reverb_sampler
        self.reverb_client_max_sequence_length = reverb_client_max_sequence_length
        self.reverb_client_chunk_length = reverb_client_chunk_length


class ReverbUtil:
    def __init__(self, learner_addr, config, logger) -> None:

        self.logger = logger

        self.reverb_sampler = config.reverb_sampler
        self.reverb_client_max_sequence_length = config.reverb_client_max_sequence_length
        self.reverb_client_chunk_length = config.reverb_client_chunk_length

        self.reverb_client = reverb.Client(learner_addr)
        self.logger.info(f"learner_proxy reverb client {self.reverb_client} {learner_addr} connect to reverb server")

        # 统计情况, 因为该类是高频调用, 不建议打印太多日志, 采用统计项来跟进
        self.send_to_reverb_server_succ_cnt = 0
        self.send_to_reverb_server_error_cnt = 0

        # 如果采样的是reverb_sampler = reverb.selectors.Prioritized, 则需要自动随机生成每条样本的优先级
        self.has_prioritized = self.reverb_sampler == "reverb.selectors.Prioritized"

        self.max_sequence_length = int(self.reverb_client_max_sequence_length)
        self.chunk_length = int(self.reverb_client_chunk_length)

        # 性能优化：复用writer对象，避免频繁创建销毁context
        self._writer = None

    def get_send_to_reverb_server_stat(self):
        tmp_send_to_reverb_server_succ_cnt = self.send_to_reverb_server_succ_cnt
        tmp_send_to_reverb_server_error_cnt = self.send_to_reverb_server_error_cnt

        # 指标周期性的复原
        self.send_to_reverb_server_succ_cnt = 0
        self.send_to_reverb_server_error_cnt = 0

        return tmp_send_to_reverb_server_succ_cnt, tmp_send_to_reverb_server_error_cnt

    def _get_writer(self):
        """
        性能优化：复用writer对象，避免频繁with context开销
        """
        if self._writer is None:
            # 关闭旧writer
            if self._writer is not None:
                try:
                    self._writer.__exit__(None, None, None)
                except:
                    pass

            # 创建新writer
            self._writer = self.reverb_client.writer(
                max_sequence_length=self.max_sequence_length,
                chunk_length=self.chunk_length,
            ).__enter__()

        return self._writer

    """
    如果设置了reverb.selectors.Prioritized, 则优先级是业务自定义的prioritized_data是个数组
    prioritized_data其长度需要和train_data的长度一致
    """

    def write_to_reverb_server_simple(self, reverb_table_names, train_data, prioritized_data=None):
        if train_data is None or len(train_data) == 0:
            return

        if self.has_prioritized:
            if not prioritized_data or len(train_data) != len(prioritized_data):
                return

        try:
            # 性能优化：复用writer，减少context创建开销
            writer = self._get_writer()

            # 性能优化：批量处理减少网络往返
            num_samples = len(train_data)
            priorities = prioritized_data if self.has_prioritized else [1.0] * num_samples

            # 批量append + create_item
            for i, sample in enumerate(train_data):
                writer.append(sample)
                for table in reverb_table_names:
                    writer.create_item(table, 1, priorities[i])

            # 统一flush，减少网络开销
            writer.flush()

            self.send_to_reverb_server_succ_cnt += num_samples

        except Exception as e:
            self.send_to_reverb_server_error_cnt += len(train_data)
            self.logger.error(
                f"learner_proxy send one data to reverb server error as {str(e)}, traceback.print_exc() is {traceback.format_exc()}"
            )
            # 出错时重置writer
            self._writer = None
