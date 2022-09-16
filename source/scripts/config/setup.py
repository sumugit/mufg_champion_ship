import os
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def setup(cfg):
    """ confirm path """
    # GPU 設定
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dir 設定
    cfg.OUTPUT = os.path.join(cfg.CURRENT_PATH, "Output")
    cfg.EXP = os.path.join(cfg.OUTPUT, "OUT_EX001")  # 適宜修正
    cfg.PRETRAIN = os.path.join(cfg.CURRENT_PATH, "pretrain")
    cfg.INPUT = os.path.join(cfg.CURRENT_PATH, "Input")
    cfg.DATASET = os.path.join(cfg.CURRENT_PATH, "Dataset")

    cfg.OUTPUT_EXP = os.path.join(cfg.OUTPUT, cfg.EXP)
    cfg.FINAL = os.path.join(cfg.OUTPUT, "final")
    cfg.EXP_PRETRAIN = os.path.join(
        cfg.PRETRAIN, "roberta_large_5fold_42")  # 適宜修正
    cfg.EXP_MODEL = os.path.join(cfg.EXP, "model")
    cfg.EXP_FIG = os.path.join(cfg.EXP, "fig")
    cfg.EXP_PREDS = os.path.join(cfg.EXP, "preds")

    # Dir 作成
    for d in [cfg.INPUT, cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
    return cfg

def set_seed(seed=2022):
    """ config seed number """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_stratifiedkfold(train, target_col, n_splits, seed):
    """ StratifiedkFold """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    generator = kf.split(train, train[target_col])
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series


def collatte(inputs, labels=None):
    """ input_ids, mask_len の数合わせ """
    # texts に含まれる各文書 の 最大 token 数を取得
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    # 訓練, 検証データ
    if not labels is None:
        # 各文書の最大 token 数までを取得
        inputs = {
            "input_ids": inputs["input_ids"][:, :mask_len],
            "attention_mask": inputs["attention_mask"][:, :mask_len],
        }
        labels = labels[:, :mask_len]
        return inputs, labels, mask_len
    # テストデータ
    else:
        inputs = {
            "input_ids": inputs["input_ids"][:, :mask_len],
            "attention_mask": inputs["attention_mask"][:, :mask_len],
        }
        return inputs, mask_len
