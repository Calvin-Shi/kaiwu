# -*- coding: utf-8 -*-
"""
Author: Tencent AI Arena Authors
"""

from kaiwudrl.common.monitor.metrics_utils import collect_training_metrics


def get_training_metrics():
    """Fetch training metrics for hok1v1 project."""
    metrics_dict = {
        "basic": {
            "train_global_step": "sum",
            "actor_predict_succ_cnt": "sum",
            "sample_production_and_consumption_ratio": "avg",
            "episode_cnt": "sum",
            "actor_load_last_model_succ_cnt": "sum",
            "sample_receive_cnt": "sum",
        },
        "algorithm": {
            "reward": "avg",
            "total_loss": "avg",
            "policy_loss": "avg",
            "value_loss": "avg",
            "entropy_loss": "avg",
        },
        "env": {
            "win_rate": "avg",
            "self_tower_hp": "avg",
            "enemy_tower_hp": "avg",
            "frame": "avg",
            "kill": "avg",
            "death": "avg",
            "money_per_frame": "avg",
            "hurt_to_hero": "avg",
            "hurt_by_hero": "avg",
        },
    }
    return collect_training_metrics(
        metrics_dict,
        group_by_label={"env": "model_id"},
    )
