#!/bin/bash
# GPU显存清理脚本
#
# 用法:
#   sh tools/clean_gpu.sh [target_pid]
#
# 参数:
#   target_pid  可选。需要无条件kill的主进程PID。通常传入调用方自己的$$,
#               用于解决debugpy场景下主进程进入Z/回收中状态导致fd扫描漏杀的问题。
#
# 设计目标:
#   杀掉所有仍持有/dev/nvidia*句柄的进程, 释放GPU显存。
#   以独立会话(setsid)在后台运行, 调用方立即返回, 不阻塞。
#
# 关键特性:
#   1. fire-and-forget: setsid nohup & 启动后台独立会话, 脱离调用方进程组
#   2. 延迟执行: sleep 3 给调用方自身退出留足窗口期
#   3. 双重策略:
#      a) 如果传入target_pid, 先无条件kill -9它(钉死目标, 避免fd扫描时机问题)
#      b) 扫/proc/*/fd/nvidia*, 杀所有仍持有GPU句柄的残留进程(兜底)

target_pid=${1:-""}

setsid nohup bash -c "
  sleep 3
  # a) 如果传入了target_pid, 无条件kill它(钉死目标)
  if [ -n \"$target_pid\" ]; then
    kill -9 $target_pid 2>/dev/null
  fi
  sleep 1
  # b) 兜底: 扫 /proc/*/fd, 杀所有仍持有 /dev/nvidia* 句柄的进程
  for pid in \$(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    if ls -l /proc/\$pid/fd 2>/dev/null | grep -q nvidia; then
      kill -9 \$pid 2>/dev/null
    fi
  done
" >/dev/null 2>&1 </dev/null &

# 立即返回, 不等待后台任务
exit 0
