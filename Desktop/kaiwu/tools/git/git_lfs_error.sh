#!/bin/bash


# 针对git lfs error时报错的处理

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh

git rm .gitattributes
git reset --hard HEAD

judge_succ_or_fail $? "git lfs error"