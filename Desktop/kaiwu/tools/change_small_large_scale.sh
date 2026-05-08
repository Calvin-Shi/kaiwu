#!/bin/bash
# 一键切换小规模和大规模场景下的配置

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/change_small_large_scale.sh small|medium|large, sh tools/change_small_large_scale.sh small  \033[0m"

    exit -1
fi


small_large_scale=$1

# 下面是具体的修改配置文件的操作
project_default_configure_file="/root/tools/conf/project_default.toml"
framework_config_file="conf/kaiwudrl/configure.toml"
framework_config_file_v2="kaiwudrl/conf/kaiwudrl/configure.toml"
if [ $small_large_scale == "small" ];
then
    sed -i 's/predict_batch_size = .*/predict_batch_size = 1/' $project_default_configure_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "local_aisrv_workflow"/' $project_default_configure_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "local_aisrv_workflow"/' $framework_config_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "local_aisrv_workflow"/' $framework_config_file_v2

elif [ $small_large_scale == "medium" ];
then
    sed -i 's/predict_batch_size = .*/predict_batch_size = 1/' $project_default_configure_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "remote_aisrv_predict"/' $project_default_configure_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "remote_aisrv_predict"/' $framework_config_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "remote_aisrv_predict"/' $framework_config_file_v2

elif [ $small_large_scale == "large" ];
then
    sed -i 's/predict_batch_size = .*/predict_batch_size = 1/' $project_default_configure_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "remote_actor_predict"/' $project_default_configure_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "remote_actor_predict"/' $framework_config_file
    sed -i 's/remote_agent_default_runtime_mode = .*/remote_agent_default_runtime_mode = "remote_actor_predict"/' $framework_config_file_v2

else
    echo -e "\033[31m useage: sh tools/change_small_large_scale.sh small|large, sh tools/change_small_large_scale.sh small  \033[0m"

    exit -1

fi

judge_succ_or_fail $? "change $project_default_configure_file $framework_config_file $small_large_scale"
