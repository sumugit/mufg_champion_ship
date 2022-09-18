import os
import sys
sys.path.append('../')
import config.setup as setup
import prediction
import numpy as np
import pandas as pd
from glob import glob
from config.config import Config
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


# 構成のセットアップ
cfg = setup.setup(Config)
cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)

# データの読込
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_test.csv'))
dataset_test['html_content'] = dataset_test['html_content'].astype(str)
submission = pd.read_csv(os.path.join(
    cfg.INPUT, 'raw/sample_submit.csv'), header=None)

sub_pred = np.zeros(shape=(dataset_test.shape[0], cfg.num_class))

ensemble_list = ['OUT_EX001', 'OUT_EX002', 'OUT_EX003']
for out in ensemble_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    sub_pred += np.load(os.path.join(expxxx_model, 'sub_pred.npy'))

lgbm_list = ['OUT_EX003']
for out in lgbm_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    sub_pred += np.load(os.path.join(expxxx_model, 'lgbm_sub_pred.npy'))

submission[1] = np.argmax(sub_pred, axis=1)
submission[1] = submission[1].astype(int)

# 提出用ファイル
submission.to_csv(os.path.join(cfg.FINAL, 'submission_1_2.csv'),
                index=False, header=False)
