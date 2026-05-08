#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import fcntl
import pyarrow as pa


class ArrowUtils:
    def __init__(self, logger):
        # 默认的共享内存地址
        self.shm_path = "/dev/shm/arrow_data"
        self.logger = logger

    def write_data(self, raw_data):
        """
        写入数据到共享内存
        param raw_data: 字典格式数据，如 {'id': [1, 2], 'value': [10.5, 20.3]}
        return: 成功返回 True, 失败返回 False
        """
        if not raw_data:
            self.logger.warning("写入数据为空")
            return False

        fd = None
        try:
            # 创建 Arrow Table
            data = pa.table(raw_data)

            # 1. 创建或打开文件描述符
            fd = os.open(self.shm_path, os.O_CREAT | os.O_RDWR, 0o666)
            # 2. 加独占锁（基于文件描述符）
            fcntl.flock(fd, fcntl.LOCK_EX)

            # 3. 将描述符转为文件对象并写入数据
            with os.fdopen(fd, "wb") as f:
                with pa.ipc.new_file(f, data.schema) as writer:
                    writer.write_table(data)

            return True
        except (IOError, ValueError, pa.ArrowInvalid) as e:
            self.logger.error(f"写入数据失败: {str(e)}", exc_info=True)
            return False
        finally:
            # 确保文件锁释放
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception as e:
                    pass

    def read_data(self):
        """
        # 从共享内存读取
        :return: 成功返回 pyarrow.Table，失败返回 None
        """
        try:
            if not os.path.exists(self.shm_path):
                raise FileNotFoundError(f"共享内存文件不存在: {self.shm_path}")

            fd = os.open(self.shm_path, os.O_RDONLY)
            # 加共享锁
            fcntl.flock(fd, fcntl.LOCK_SH)

            # 读取数据
            with os.fdopen(fd, "rb") as f:
                mmap = pa.memory_map(self.shm_path)
                reader = pa.ipc.open_file(mmap)
                return reader.read_all()

        except (IOError, pa.ArrowInvalid) as e:
            self.logger.error(f"读取数据失败: {str(e)}")
            return None
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception as e:
                    pass
