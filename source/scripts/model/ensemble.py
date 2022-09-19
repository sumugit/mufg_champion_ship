import os
import sys
sys.path.append('../')
import config.setup as setup
import prediction
import numpy as np
import pandas as pd
from glob import glob
from config.config import Config
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


# 構成のセットアップ
cfg = setup.setup(Config)
cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)

# データの読込
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_train.csv'))
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_test.csv'))
dataset_train['html_content'] = dataset_train['html_content'].astype(str)
dataset_test['html_content'] = dataset_test['html_content'].astype(str)
submission = pd.read_csv(os.path.join(cfg.INPUT, 'raw/sample_submit.csv'), header=None)

oof_pred = np.zeros(shape=(dataset_train.shape[0], cfg.num_class))
sub_pred = np.zeros(shape=(dataset_test.shape[0], cfg.num_class))

ensemble_list = ['OUT_EX001', 'OUT_EX002', 'OUT_EX003', 'OUT_EX004', 'OUT_EX005']
for out in ensemble_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    oof_pred += np.load(os.path.join(expxxx_model, 'oof_pred.npy'))
    sub_pred += np.load(os.path.join(expxxx_model, 'sub_pred.npy'))

lgbm_list = ['OUT_EX003']
for out in lgbm_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    oof_pred += np.load(os.path.join(expxxx_model, 'lgbm_oof_pred.npy'))
    sub_pred += np.load(os.path.join(expxxx_model, 'lgbm_sub_pred.npy'))


np.save(os.path.join(cfg.FINAL, 'oof_pred.npy'), oof_pred)
np.save(os.path.join(cfg.FINAL, 'sub_pred.npy'), sub_pred)
score = f1_score(np.argmax(oof_pred, axis=1), dataset_train[cfg.target])
print('CV:', round(score, 5))

submission[1] = np.argmax(sub_pred, axis=1)
submission[1] = submission[1].astype(int)

# 提出用ファイル
submission.to_csv(os.path.join(cfg.FINAL, 'submission_12345.csv'),
                index=False, header=False)
