#!/bin/bash
# Train plugin - 启用选择召唤师技能（enable_select_skill）

set +e

GAME_CONFIG_FILE="/data/projects/kaiwu_arena/environments/${app}/conf/game.yaml"

log() {
    local msg="$1"

    if type log_info >/dev/null 2>&1; then
        log_info "$msg"
    else
        echo "[train-plugin] ${msg}"
    fi
}

update_enable_select_skill() {
    local config_file="$1"

    if [ ! -f "$config_file" ]; then
        log "配置文件不存在，跳过更新：${config_file}"
        return 0
    fi

    if grep -qE "^[[:space:]]*enable_select_skill:" "$config_file" 2>/dev/null; then
        if sed -i -E "s|^([[:space:]]*)enable_select_skill:.*$|\1enable_select_skill: true|g" "$config_file" 2>/dev/null; then
            log "✓ 已更新 enable_select_skill 为 true"
            return 0
        fi

        log "✗ 更新 enable_select_skill 失败，但继续执行"
        return 0
    fi

    if printf "\nenable_select_skill: true\n" >>"$config_file" 2>/dev/null; then
        log "✓ 已追加 enable_select_skill 为 true"
        return 0
    fi

    log "✗ 追加 enable_select_skill 失败，但继续执行"
    return 0
}

main() {
    log "准备更新 ${GAME_CONFIG_FILE} 中的 enable_select_skill 为 true"
    update_enable_select_skill "$GAME_CONFIG_FILE"
    log "Train plugin 执行完成"
    return 0
}

main "$@" || true
