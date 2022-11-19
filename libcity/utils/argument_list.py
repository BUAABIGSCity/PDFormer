import argparse

general_arguments = {
    "gpu": "bool",
    "batch_size": "int",
    "train_rate": "float",
    "part_train_rate": "float",
    "eval_rate": "float",
    "learning_rate": "float",
    "max_epoch": "int",
    "gpu_id": "list of int",
    "seed": "int",
    "dataset_class": "str",
    "executor": "str",
    "evaluator": "str",

    "input_window": "int",
    "output_window": "int",
    "scaler": "str",
    "load_external": "bool",
    "normal_external": "bool",
    "ext_scaler": "str",
    "add_time_in_day": "bool",
    "add_day_in_week": "bool",
    "use_trend": "bool",
    "len_closeness": "int",
    "len_period": "int",
    "len_trend": "int",
    "interval_period": "int",
    "interval_trend": "int",
    "data_col": "str",
    "bidir": "bool",
    "far_mask_delta": "float",
    "dtw_delta": "int",

    "learner": "str",
    "weight_decay": "float",
    "lr_decay": "bool",
    "lr_scheduler": "str",
    "lr_eta_min": "float",
    "lr_decay_ratio": "float",
    "lr_warmup_epoch": "int",
    "lr_warmup_init": "float",
    "use_early_stop": "bool",
    "patience": "int",
    "clip_grad_norm": "bool",
    "max_grad_norm": "int",
    "random_flip": "bool",
    "use_curriculum_learning": "bool",
    "grad_accmu_steps": "int",
    "set_loss": "str",
    "huber_delta": "float",
    "quan_delta": "float",

    "step_size": "int",
    "embed_dim": "int",
    "skip_dim": "int",
    "lape_dim": "int",
    "geo_num_heads": "int",
    "sem_num_heads": "int",
    "t_num_heads": "int",
    "mlp_ratio": "int",
    "qkv_bias": "bool",
    "drop": "float",
    "attn_drop": "float",
    "drop_path": "float",
    "s_attn_size": "int",
    "t_attn_size": "int",
    "enc_depth": "int",
    "dec_depth": "int",
    "type_ln": "str",
    "type_short_path": "str",
    "cand_key_days": "int",
    "n_cluster": "int",
    "cluster_max_iter": "int",
    "cluster_method": "str",

    "mode": "str",
    "mask_val": "int"
}

hyper_arguments = {
    "gpu": {
        "type": "bool",
        "default": None,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    }
}


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x
