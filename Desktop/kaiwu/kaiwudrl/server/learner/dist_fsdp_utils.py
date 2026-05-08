#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import torch
import torch.distributed as dist
from common_python.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


def setup_distributed():
    """
    设置分布式训练环境, 支持CPU和GPU
    返回: (rank, world_size, device)
    """
    if CONFIG.svr_name != KaiwuDRLDefine.SERVER_LEARNER:
        return 0, 1, torch.device("cpu")

    # 获取环境变量，设置默认值
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        # GPU环境 - 使用nccl后端（GPU分布式训练的最佳选择）
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")

        # 设置当前GPU设备
        torch.cuda.set_device(device)

        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("MASTER_ADDR", "env://"),
                timeout=torch.distributed.default_pg_timeout,
            )

        # 额外的GPU环境检查
        if dist.is_initialized():
            print(
                f"GPU Distributed training initialized: "
                f"rank={dist.get_rank()}, world_size={dist.get_world_size()}, "
                f"device={device}, backend={backend}"
            )
    else:
        # CPU环境 - 使用gloo后端
        backend = "gloo"
        device = torch.device("cpu")

        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("MASTER_ADDR", "env://"),
                timeout=torch.distributed.default_pg_timeout,
            )

        if dist.is_initialized():
            print(
                f"CPU Distributed training initialized: "
                f"rank={dist.get_rank()}, world_size={dist.get_world_size()}, "
                f"backend={backend}"
            )

    return dist.get_rank(), dist.get_world_size(), device


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed training environment cleaned up")


def is_distributed_available():
    """
    检查分布式训练是否可用
    """
    return dist.is_available() and dist.is_initialized()


def get_global_rank():
    """
    获取全局rank
    """
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    """
    获取world size
    """
    return dist.get_world_size() if dist.is_initialized() else 1


def distributed_barrier():
    """
    分布式屏障同步
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


# 使用示例
if __name__ == "__main__":
    try:
        # 初始化分布式训练
        rank, world_size, device = setup_distributed()

        print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")

        # 在这里进行分布式训练操作...

    finally:
        # 清理资源
        cleanup_distributed()
