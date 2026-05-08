#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import re
import time
import signal
import shutil
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
from common_python.config.config_control import CONFIG
from distutils.dir_util import copy_tree, remove_tree
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    base64_decode,
    write_json_to_file,
    make_tar_file,
    make_single_dir,
    stop_process_by_pid,
    load_private_key_by_data,
    public_signature_verify_by_data,
    generate_private_signature_by_data,
    python_exec_shell,
    compute_directory_hash,
    get_map_content,
    load_public_key_by_data,
)
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.common_func import TimeIt, set_schedule_event


def format_reward_config():
    """
    将 KAIWU_REWARD_CONFIG 的 json 字符串，按字典序对 key 进行排序，
    拼接成 "key1_value1_key2_value2_..." 的形式。
    """
    reward_config_str = os.environ.get("KAIWU_REWARD_CONFIG", "")
    if not reward_config_str:
        return ""

    try:
        config = json.loads(reward_config_str)
        if not isinstance(config, dict):
            return ""

        parts = []
        for key in sorted(config.keys()):
            value = config[key]
            parts.append(f"{key}_{value}")

        return "_".join(parts)
    except Exception:
        return ""


def before_save_model():
    # 文件夹如果没有则新建
    make_single_dir(CONFIG.user_ckpt_dir)

    # 清空原有文件内容
    file_path = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}"
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    make_single_dir(file_path)


def check_id_valid(id):
    """
    因为有些场景只是检测ID是否正常即可, 不需要检测path
    """
    if not id:
        return False

    # 排除特别的ID类型
    if id == KaiwuDRLDefine.ID_LATEST or id == KaiwuDRLDefine.ID_RANDOM:
        return True

    try:
        if int(id) < 0:
            return False

    except ValueError:
        return False

    return True


def check_path_id_valid(path, id):
    """
    判断规则:
    1. path是文件目录并且是存在的
    2. id是大于等于0的
    """
    if not path or not os.path.isdir(path) or not os.path.exists(path):
        return False

    # 排除特别的ID类型
    if id == KaiwuDRLDefine.ID_LATEST or id == KaiwuDRLDefine.ID_RANDOM:
        return True

    try:
        if int(id) < 0:
            return False

    except ValueError:
        return False

    return True


def clear_user_ckpt_dir():
    """
    清空/data/user_ckpt/下的文件夹内容, 在进程启动时进行操作
    为软链接时不做操作
    """
    model_path = CONFIG.user_ckpt_dir
    if os.path.islink(model_path):
        return

    try:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
    except FileNotFoundError:
        # 多进程并发启动时, 可能已被其他进程删除, 忽略即可
        pass


def clear_id_list_file(framework=False):
    """
    清空id_list文件, 否则id_list会包含旧的数据, 可能导致磁盘占用增加的问题
    """
    if framework:
        id_lst_file = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"
    else:
        id_lst_file = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"

    if os.path.exists(id_lst_file):
        os.remove(id_lst_file)


def update_save_id_list(file_name):
    """
    由于落地model文件在learner进程, 而上传model文件在model_file_save进程, 那么就存在进程之间的同步问题
    """
    if not file_name:
        return

    id_lst_file = f"{CONFIG.user_ckpt_dir}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"
    if not os.path.exists(id_lst_file):
        with open(id_lst_file, "w") as file:
            # 在文件中写入内容或执行其他操作
            file.write(f"{file_name}\n")
    else:
        with open(id_lst_file, "a") as file:
            # 在文件中写入内容或执行其他操作
            file.write(f"{file_name}\n")


def update_id_list(train_step, framework=False):
    """
    维护的ID_LIST列表操作:
    1.如果KAIWU_MODEL_ID_LIST文件不存在则新增文件
    2.如果KAIWU_MODEL_ID_LIST文件存在则新增内容, 内容为id列表
    """
    if framework:
        id_lst_file = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"
    else:
        id_lst_file = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"

    if not os.path.exists(id_lst_file):
        with open(id_lst_file, "w") as file:
            file.write(f"all id list\n")
            # 在文件中写入内容或执行其他操作
            file.write(f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{train_step}\n")
    else:
        with open(id_lst_file, "a") as file:
            # 在文件中写入内容或执行其他操作
            file.write(f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{train_step}\n")


def find_id_in_id_list(train_step, framework=False):
    """
    检查当前id是否在ID_LIST列表:
    """
    if framework:
        id_lst_file = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"
    else:
        id_lst_file = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}"

    if not os.path.exists(id_lst_file):
        return False
    else:
        model_file = f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-{train_step}"
        model_file_list = []
        with open(id_lst_file, "r") as file:
            tmp_model_file_list = file.readlines()
            for tmp_model_file in tmp_model_file_list:
                model_file_list.append(tmp_model_file.strip())
        return model_file in model_file_list


def after_save_model(process_start_time, id):
    # 将时间转换为RFC3339格式
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # 使用本地时区
    now = datetime.now().astimezone()
    created_at = now.isoformat()

    local_file = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}_{CONFIG.algo}"

    # 更新update_id_list
    update_id_list(id)

    # 通识课奖励配置数据生成用的
    reward_config = format_reward_config()
    train_time = os.environ.get("KAIWU_TRAIN_TIME", "")
    if reward_config:  # 只有非空字符串才成立
        output_file_name = f"{CONFIG.user_ckpt_dir}/{CONFIG.app}-{CONFIG.algo}-{reward_config}_runtime_{train_time}-{id}.{KaiwuDRLDefine.TAR_GZ}"
    # 正常训练模式：kaiwudrl打包模型文件, 在平台, 单机, 客户端上命名不一样的
    elif CONFIG.deployment_platforms == KaiwuDRLDefine.DEPLOYMENT_PLATFORMS_CLIENT:
        output_file_name = (
            f"{CONFIG.user_ckpt_dir}/{CONFIG.app}-{CONFIG.algo}-{id}-"
            f"{time_str}-{CONFIG.kaiwu_project_version}.{KaiwuDRLDefine.TAR_GZ}"
        )
    else:
        output_file_name = (
            f"{CONFIG.user_ckpt_dir}/{CONFIG.app}-{CONFIG.task_id}-{CONFIG.algo}-{id}-"
            f"{time_str}-{CONFIG.kaiwu_project_version}.{KaiwuDRLDefine.TAR_GZ}"
        )

    make_tar_file(output_file_name, local_file)

    # 注意存在信号退出和正常退出时不同的train_step相互覆盖问题, 故这里的名字需要带上train_step
    json_file_name = f"{KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_JSON_FILE_NAME}_{id}"
    write_json_to_file(
        {
            "created_at": created_at,
            "train_time": int(time.monotonic() - process_start_time),
            "train_step": id,
            "platform": KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_MAGIC,
            "business": CONFIG.business,
            "user_id": CONFIG.user_id,
            "team_id": CONFIG.team_id,
            "project_code": CONFIG.app,
            "project_version": CONFIG.kaiwu_project_version,
            "task_id": CONFIG.task_id,
            "algorithm": CONFIG.algo,
        },
        json_file_name,
        CONFIG.user_ckpt_dir,
    )

    # 由于落地model文件在learner进程, 而上传model文件在model_file_save进程, 那么就存在进程之间的同步问题
    update_save_id_list(output_file_name)


def model_file_signature_verify(target_dir):
    """
    因为评估时会调用该函数, 注意此时的target_dir其实是在ckpt这一层, 参考下图就是/workspace/train/backup_model/tmp/ckpt/
    [root@a402271b9651 /data/projects/gorge_walk]# ll -rt /workspace/train/backup_model/tmp/ckpt/
    总用量 140
    -rw-r--r-- 1 root root 131200 4月  19 20:42 model.ckpt-1.npy
    -rw-r--r-- 1 root root     25 4月  19 20:42 id_list
    -rw-r--r-- 1 root root    823 4月  19 20:43 kaiwu.json
    [root@a402271b9651 /data/projects/gorge_walk]# ll -rt /workspace/train/backup_model/tmp/
    总用量 96
    drwxr-xr-x 5 root root  4096 4月  19 20:43 sarsa
    drwxr-xr-x 5 root root  4096 4月  19 20:43 q_learning
    drwxr-xr-x 5 root root  4096 4月  19 20:43 monte_carlo
    -rw-r--r-- 1 root root   283 4月  19 20:43 kaiwu.json
    drwxr-xr-x 5 root root  4096 4月  19 20:43 dynamic_programming
    drwxr-xr-x 4 root root  4096 4月  19 20:43 conf
    drwxr-xr-x 2 root root  4096 4月  19 20:43 ckpt
    -rw-r--r-- 1 root root 67124 4月  19 20:45 gorge_walk-uuid10-dynamic_programming-1-2024_04_19_20_42_36-1.1.1.zip
    [root@a402271b9651 /data/projects/gorge_walk]#
    """
    if not target_dir or not os.path.exists(target_dir):
        print(f"target_dir or not os.path.exists(target_dir) target_dir {target_dir}, please check")
        return False

    # 注意是把kaiwu.json写入到ckpt文件夹下的
    kaiwu_json_file = f"{target_dir}/{KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_JSON_FILE_NAME}.json"

    if not os.path.exists(kaiwu_json_file):
        print(f"kaiwu_json_file {kaiwu_json_file} is not exists, please check")
        return False

    # 计算模型文件的hash值
    with open(
        kaiwu_json_file,
        "r",
    ) as f:
        data = json.load(f)

    # 判断字典里的部分数据和配置项是否一致
    if (
        data["user_id"] != CONFIG.user_id
        or data["team_id"] != CONFIG.team_id
        or data["platform"] != KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_MAGIC
        or data["business"] != CONFIG.business
        or data["project_code"] != CONFIG.app
        or data["project_version"] != CONFIG.kaiwu_project_version
        or data["algorithm"] != CONFIG.algo
    ):
        return False

    # 因为要现场计算public_signature, 故在计算时需要去掉该值
    private_signature = base64_decode(data["signature"])
    del data["signature"]

    # 比较model文件hash值
    target_directory_hash = data["model_file_hash"]
    model_file_hash, model_files = compute_directory_hash(
        f"{target_dir}",
        exclude_files=[f"{KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_JSON_FILE_NAME}.json"],
    )

    if target_directory_hash != model_file_hash:
        print(f"target_directory_hash {target_directory_hash} != model_file_hash {model_file_hash}")

        return False

    # 采用公钥去验证
    json_dumps = get_map_content(data)
    public_key = load_public_key_by_data(CONFIG.public_key_content)
    if not public_signature_verify_by_data(public_key, json_dumps, private_signature):
        print(
            f"public_signature_verify_by_data failed, "
            f"private_signature {private_signature}, json_dumps {json_dumps}, private_key {public_key}"
        )
        return False

    return True


def model_file_signature_verify_by_file(model_zip_file):
    """
    针对数字签名后的包进行校验, 确认当时进行数字签名的内容, 操作如下:
    1. 解压缩zip包
    2. 根据zip包里的ckpt文件夹下的内容计算数字签名
    3. 2中计算的结果与json文件里的model_file_hash值进行匹配, 如果匹配上则继续4, 否则报错
    4. 针对json文件内容进行数字签名, 与json文件里的private_signature进行匹配, 如果匹配上则返回True, 否则返回False

    该函数因为是外界调用的, 故无法采用logger打印错误日志, 即采用printf
    """
    if not model_zip_file or not os.path.exists(model_zip_file):
        print(f"model_zip_file or not os.path.exists(model_zip_file) model_zip_file {model_zip_file}, please check")
        return False

    target_dir = tempfile.mkdtemp()
    shutil.copy(model_zip_file, target_dir)

    python_exec_shell(f"cd {target_dir} && unzip * ")

    success = model_file_signature_verify(f"{target_dir}/ckpt/")

    remove_tree(target_dir)

    return success


def get_checkpoint_id_by_re(last_line):
    """
    根据last_line来获取checkpoint_id
    """
    if not last_line:
        return None

    """
    1. standard模式下, checkpoint文件内容:
    all_model_checkpoint_paths: "/data/ckpt//sgame_1v1_ppo/model.ckpt-66700861"
    all_model_checkpoint_paths: "/data/ckpt//sgame_1v1_ppo/model.ckpt-66703420"

    """
    checkpoint_id = re.search(r"(?<={}-)\d+".format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_line)
    if not checkpoint_id:
        return None
    checkpoint_id = int(checkpoint_id.group())
    if checkpoint_id < 0:
        return None

    return checkpoint_id


# 进程正常退出时预留文件
def process_stop_write_file(error_code, logger):
    file_name = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/process_stop.done"

    # 需要考虑到文件夹没有建立的情况
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    # 如果有旧的文件则删除
    if Path(file_name).exists():
        Path(file_name).unlink()

    with open(file_name, "w") as f:
        f.write(str(error_code))

    # 需要考虑没有传logger的场景
    if logger:
        logger.info(f"process_stop_write_file, error_code is {error_code}")
    else:
        print(f"process_stop_write_file, error_code is {error_code}")
