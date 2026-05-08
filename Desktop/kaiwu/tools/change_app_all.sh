#!/bin/bash

# 在各个场景打镜像时, 采用一键来修改配置

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh kaiwudrl/utils/change_app_all.sh app_name, sh kaiwudrl/utils/change_app_all.sh gorge_walk  \033[0m"

    exit -1
fi

app_name=$1

# 下面是具体的修改配置文件的操作
tools_start_bash_file="tools/start.sh"
utils_start_bash_file="kaiwudrl/utils/start.sh"
produce_config="kaiwudrl/utils/produce_config.sh"
run_mulit_learner_by_horovodrun_config="kaiwudrl/utils/run_mulit_learner_by_horovodrun.sh"
run_mulit_learner_by_openmpirun_config="kaiwudrl/utils/run_mulit_learner_by_openmpirun.sh"
main_configure="kaiwudrl/conf/kaiwudrl/configure.toml"
app_configure="conf/configure_app.toml"
modelpool_stop_file="kaiwudrl/utils/modelpool_stop.sh"
modelpool_start_file="kaiwudrl/utils/modelpool_start.sh"
gpu_iplist_file="kaiwudrl/thirdparty/model_pool_go/config/gpu.iplist"
clear_aisrv_log_file="kaiwudrl/utils/clear_aisrv_log.sh"
clear_actor_log_file="kaiwudrl/utils/clear_actor_log.sh"
clear_learner_log_file="kaiwudrl/utils/clear_learner_log.sh"
project_default_configure_file="/root/tools/conf/project_default.toml"
deployment_kit_constant_file="/root/tools/constant.sh"

files=("$utils_start_bash_file" "$tools_start_bash_file" "$produce_config" "$run_mulit_learner_by_horovodrun_config" \
       "$run_mulit_learner_by_openmpirun_config" "$main_configure" "$app_configure" "$modelpool_stop_file" \
       "$modelpool_start_file" "$gpu_iplist_file" "$clear_aisrv_log_file" "$clear_actor_log_file" "$clear_learner_log_file" "$project_default_configure_file")

for file in "${files[@]}";
do
    sed -i "s|/data/projects/kaiwu-fwk|/data/projects/${app_name}|g" "$file"
    judge_succ_or_fail $? "change $file $app_name"

done

if [ -f "$deployment_kit_constant_file" ]; then
    sed -i "s|NEED_TO_CHANGE|${app_name}|g" "$deployment_kit_constant_file"
    judge_succ_or_fail $? "change $deployment_kit_constant_file $app_name"
else
    echo "文件不存在: $deployment_kit_constant_file"
fi
