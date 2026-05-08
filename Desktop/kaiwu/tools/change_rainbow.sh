#!/bin/bash

# 一键切换七彩石配置信息


chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh


if [ $# -ne 3 ];
then
    echo -e "\033[31m useage: sh tools/change_rainbow.sh rainbow_env_name rainbow_user_id rainbow_secret_key \033[0m"

    exit -1
fi

rainbow_env_name=$1
rainbow_user_id=$2
rainbow_secret_key=$3

# 下面是具体的修改配置文件的操作
config_file="kaiwudrl/conf/kaiwudrl/configure.toml"
project_default_configure_file="/root/tools/conf/project_default.toml"
sed -i 's/use_rainbow = .*/use_rainbow = true/' $config_file
sed -i 's/^rainbow_env_name = .*/rainbow_env_name = "'"$rainbow_env_name"'"/' $config_file
sed -i 's/^rainbow_env_name = .*/rainbow_env_name = "'"$rainbow_env_name"'"/' $project_default_configure_file
sed -i 's/^rainbow_user_id = .*/rainbow_user_id = "'"$rainbow_user_id"'"/' $config_file
sed -i 's/^rainbow_user_id = .*/rainbow_user_id = "'"$rainbow_user_id"'"/' $project_default_configure_file
sed -i 's/^rainbow_secret_key = .*/rainbow_secret_key = "'"$rainbow_secret_key"'"/' $config_file
sed -i 's/^rainbow_secret_key = .*/rainbow_secret_key = "'"$rainbow_secret_key"'"/' $project_default_configure_file

judge_succ_or_fail $? "change rainbow $rainbow_env_name $rainbow_user_id $rainbow_secret_key"
