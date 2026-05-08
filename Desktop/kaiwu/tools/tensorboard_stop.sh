#!/bin/bash


# tensorboard 程停止脚本

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh

service_name='tensorboard'

judge_process_exist_and_kill $service_name
judge_succ_or_fail $? "$service_name stop"