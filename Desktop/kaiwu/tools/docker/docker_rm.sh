#!/bin/bash

# 删除机器上废弃的docker镜像

chmod +x kaiwudrl/utils/common.sh
. kaiwudrl/utils/common.sh

docker system prune -af
