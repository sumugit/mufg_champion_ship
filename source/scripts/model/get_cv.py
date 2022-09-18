import os
import sys
sys.path.append('../')
import config.setup as setup
import numpy as np
import pandas as pd
from config.config import Config
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


# 構成のセットアップ
cfg = setup.setup(Config)
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, "processed/processed_train.csv"))
oof_pred = np.load(os.path.join(cfg.EXP_PREDS, 'oof_pred.npy'))

score = f1_score(np.argmax(oof_pred, axis=1),
                dataset_train[cfg.target])

print(f'CV: {round(score, 5)}')