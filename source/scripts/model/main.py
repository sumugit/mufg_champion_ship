import os
import sys
sys.path.append('../')
import config.setup as setup
import prediction
import numpy as np
import pandas as pd
from glob import glob
import model.bert_train as model
from config.config import Config
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


# 構成のセットアップ
cfg = setup.setup(Config)

# データの読込
dataset_train = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_train.csv'))
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_test.csv'))
dataset_train['html_content'] = dataset_train['html_content'].astype(str)
dataset_test['html_content'] = dataset_test['html_content'].astype(str)
submission = pd.read_csv(os.path.join(cfg.INPUT, 'raw/sample_submit.csv'), header=None)

# tokenizerの読み込み
cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)


# validationデータの設定
cfg.folds = setup.get_stratifiedkfold(dataset_train, cfg.target, cfg.num_fold, cfg.seed)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, 'folds.csv'), header=False)  # fold の index 保存

# deBERTa-v3-large の学習
score = model.training(cfg, dataset_train, dataset_test)

# BERTの推論
cfg.model_weights = [p for p in sorted(
    glob(os.path.join(cfg.EXP_MODEL, 'fold*.pth')))]
sub_pred = prediction.inferring(cfg, dataset_test)
submission[1] = np.argmax(sub_pred, axis=1)

# 提出用ファイル
submission.to_csv(os.path.join(cfg.EXP_PREDS, 'submission.csv'),
        index=False, header=False)