#!/bin/bash
# 更新配置文件里的on-line/off-line的配置



chmod +x tools/common.sh
. tools/common.sh


if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_on_off_line.sh on-line|off-line \
    such as: sh tools/change_on_off_line.sh off-line  \033[0m"

    exit -1
fi

on_off_line=$1

# 同时修改下面的配置文件
configure_file=conf/kaiwudrl/configure.toml
project_default_configure_file="/root/tools/conf/project_default.toml"

if [ $on_off_line == "on-line" ] || [ $on_off_line == "off-line" ];
then
    # 修改掉algorithm_on_policy_or_off_policy
    sed -i "s/change_on_off_line = .*/change_on_off_line = \"$on_off_line\"/g" $configure_file
    sed -i "s/change_on_off_line = .*/change_on_off_line = \"$on_off_line\"/g" $project_default_configure_file

else
    echo -e "\033[31m useage: sh tools/change_on_off_line.sh on-line|off-line \
    such as: sh tools/change_on_off_line.sh off-line  \033[0m"

    exit -1
fi

judge_succ_or_fail $? "$on_off_line change $configure_file $project_default_configure_file success"
