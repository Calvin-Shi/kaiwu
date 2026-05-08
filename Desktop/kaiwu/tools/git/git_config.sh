#!/bin/bash


# git的config操作

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh


git config -l

judge_succ_or_fail $? "git config"