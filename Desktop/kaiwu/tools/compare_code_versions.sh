#!/bin/bash
# 对比指定算法的简配版本与基线的差异
# 用法:
#   sh tools/compare_code_versions.sh <agent> <profile>                 # 对比 profile 与基线
#   sh tools/compare_code_versions.sh <agent> <profile_a> <profile_b>   # 对比两个 profile
#
# 示例:
#   sh tools/compare_code_versions.sh agent_ppo computer
#   sh tools/compare_code_versions.sh agent_diy computer no_computer

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROFILES_DIR="$PROJECT_DIR/agent_profiles"

if [ ! -d "$PROFILES_DIR" ]; then
    echo "错误: agent_profiles 目录不存在 ($PROFILES_DIR)"
    exit 1
fi

show_usage() {
    echo "用法:"
    echo "  $0 <agent> <profile>                 # 对比 profile 与基线"
    echo "  $0 <agent> <profile_a> <profile_b>   # 对比两个 profile"
    echo ""
    echo "示例:"
    echo "  $0 agent_ppo computer"
    echo "  $0 agent_diy computer no_computer"
    echo ""

    # 列出所有可用的算法和 profiles
    echo "可用算法及 profiles:"
    for d in "$PROFILES_DIR"/*/; do
        profile_name=$(basename "$d")
        [ "$profile_name" = "default" ] && continue
        # 检查该 profile 下有哪些算法子目录
        has_agents=false
        for agent_dir in "$d"/*/; do
            [ ! -d "$agent_dir" ] && continue
            agent_name=$(basename "$agent_dir")
            count=$(find "$agent_dir" -type f ! -name '.gitkeep' | wc -l)
            if [ "$count" -gt 0 ]; then
                echo "  $agent_name / $profile_name  ($count 个差异文件)"
                has_agents=true
            fi
        done
        if [ "$has_agents" = false ]; then
            # 兼容旧格式: profile 下直接放文件(无算法子目录)
            count=$(find "$d" -type f ! -name '.gitkeep' | wc -l)
            [ "$count" -gt 0 ] && echo "  (flat) $profile_name  ($count 个差异文件)"
        fi
    done
}

if [ $# -lt 2 ]; then
    show_usage
    exit 0
fi

agent="$1"
shift

BASELINE_DIR="$PROJECT_DIR/$agent"
if [ ! -d "$BASELINE_DIR" ]; then
    echo "错误: 算法目录 '$agent' 不存在 ($BASELINE_DIR)"
    exit 1
fi

if [ $# -eq 1 ]; then
    profile="$1"
    profile_dir="$PROFILES_DIR/$profile/$agent"

    if [ ! -d "$profile_dir" ]; then
        echo "错误: profile '$profile' 下没有算法 '$agent' 的差异文件 ($profile_dir)"
        exit 1
    fi

    echo "========================================="
    echo " [$agent] Profile [$profile] vs 基线"
    echo "========================================="
    echo ""

    files=$(find "$profile_dir" -type f ! -name '.gitkeep' | sort)
    if [ -z "$files" ]; then
        echo "(无差异文件)"
        exit 0
    fi

    for f in $files; do
        rel="${f#$profile_dir/}"
        base_file="$BASELINE_DIR/$rel"
        echo "--- [$rel] ---"
        if [ ! -f "$base_file" ]; then
            echo "  [新增文件] 基线中不存在"
        else
            diff_output=$(diff -u "$base_file" "$f" 2>/dev/null || true)
            if [ -z "$diff_output" ]; then
                echo "  (与基线相同，无差异)"
            else
                echo "$diff_output"
            fi
        fi
        echo ""
    done

elif [ $# -eq 2 ]; then
    profile_a="$1"
    profile_b="$2"
    dir_a="$PROFILES_DIR/$profile_a/$agent"
    dir_b="$PROFILES_DIR/$profile_b/$agent"

    if [ ! -d "$dir_a" ]; then
        echo "错误: profile '$profile_a' 下没有算法 '$agent' 的差异文件"
        exit 1
    fi
    if [ ! -d "$dir_b" ]; then
        echo "错误: profile '$profile_b' 下没有算法 '$agent' 的差异文件"
        exit 1
    fi

    echo "========================================="
    echo " [$agent] Profile [$profile_a] vs [$profile_b]"
    echo "========================================="
    echo ""
    diff -r -u "$dir_a" "$dir_b" --exclude='.gitkeep' || true
fi
