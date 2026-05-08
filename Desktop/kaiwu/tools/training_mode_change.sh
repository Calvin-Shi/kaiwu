#!/bin/bash
# 一键切换训练模式(wrapper_type)

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/training_mode_change.sh local|remote|none, e.g. sh tools/training_mode_change.sh local \033[0m"
    exit -1
fi

wrapper_type=$1

# 校验入参只能是 local、remote、none
if [ "$wrapper_type" != "local" ] && [ "$wrapper_type" != "remote" ] && [ "$wrapper_type" != "none" ];
then
    echo -e "\033[31m Error: wrapper_type must be one of: local, remote, none. Got: $wrapper_type \033[0m"
    exit -1
fi

# 修改 tools/conf/project_default.toml 中的 wrapper_type
config_file="tools/conf/project_default.toml"
if grep -q 'wrapper_type' $config_file; then
    # 已存在 wrapper_type，直接替换
    sed -i 's/wrapper_type = .*/wrapper_type = "'"$wrapper_type"'"/' $config_file
    judge_succ_or_fail $? "change $config_file wrapper_type to $wrapper_type"
else
    # 不存在 wrapper_type，追加到文件末尾
    echo "" >> $config_file
    echo "# Wrapper type" >> $config_file
    echo "# 采用的wrapper形式" >> $config_file
    echo "wrapper_type = \"$wrapper_type\"" >> $config_file
    judge_succ_or_fail $? "add wrapper_type = $wrapper_type to $config_file"
fi
