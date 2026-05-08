#!/bin/bash


# git的branch操作

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh


git branch

judge_succ_or_fail $? "git branch"