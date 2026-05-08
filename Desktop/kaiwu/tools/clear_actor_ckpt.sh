#!/bin/bash

# 清理机器上actor进程的tensorflow的checkpoint文件的功能

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh

# 直接删除对应的文件夹
rm -rf /data/ckpt/*
judge_succ_or_fail $? "/data/ckpt/ clear"