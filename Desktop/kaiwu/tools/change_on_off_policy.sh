#!/bin/bash
# 更新配置文件里的on-policy/off-policy的配置



chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh


if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_on_off_policy.sh on-policy|off-policy \
    such as: sh tools/change_on_off_policy.sh off-policy  \033[0m"

    exit -1
fi

on_off_policy=$1

# 同时修改下面的配置文件
configure_file=kaiwudrl/conf/kaiwudrl/configure.toml
project_default_configure_file="/root/tools/conf/project_default.toml"

if [ $on_off_policy == "on-policy" ] || [ $on_off_policy == "off-policy" ];
then
    # 修改掉algorithm_on_policy_or_off_policy
    sed -i "s/algorithm_on_policy_or_off_policy = .*/algorithm_on_policy_or_off_policy = \"$on_off_policy\"/g" $configure_file
    sed -i "s/algorithm_on_policy_or_off_policy = .*/algorithm_on_policy_or_off_policy = \"$on_off_policy\"/g" $project_default_configure_file

    # 修改掉dump_model_freq, 在on-policy只能设置为1
    sed -i "s/dump_model_freq = .*/dump_model_freq = 1/g" $configure_file
    sed -i "s/dump_model_freq = .*/dump_model_freq = 1/g" $project_default_configure_file

    # 修改掉predict_batch_size, 在on-policy只能设置为1
    sed -i "s/predict_batch_size = .*/predict_batch_size = 1/g" $configure_file
    sed -i "s/predict_batch_size = .*/predict_batch_size = 1/g" $project_default_configure_file

else
    echo -e "\033[31m useage: sh tools/change_on_off_policy.sh on-policy|off-policy \
    such as: sh tools/change_on_off_policy.sh off-policy  \033[0m"

    exit -1
fi

judge_succ_or_fail $? "$on_off_policy change $configure_file $project_default_configure_file success"
