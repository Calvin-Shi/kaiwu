#!/bin/bash

# 清理机器上日志文件目录

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh

log_confile_file=/data/projects/kaiwu-fwk/kaiwudrl/conf/kaiwudrl/configure.toml

# 获取配置的日志文件目录
log_dir=`grep 'log_dir' /data/projects/kaiwu-fwk/kaiwudrl/conf/kaiwudrl/configure.toml | sed 's/.*= //'`

rm -rf $log_dir/*
judge_succ_or_fail $? "delet log_dir $log_dir"