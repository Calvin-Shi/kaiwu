#!/bin/bash

# 一键删除掉共享内存下的文件

# 删除/dev/shm下相关的, 其余的不能删除
rm -rf /dev/shm/sem.*env_to_entity* /dev/shm/*env_to_entity*shm \
      /dev/shm/sem.*entity_to_env* /dev/shm/*entity_to_env*shm \
      /dev/shm/sem.*aisrv* /dev/shm/*aisrv*shm
