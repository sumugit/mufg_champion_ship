import os
import lightgbm as lgb
from glob import glob
import numpy as np
import sys
sys.path.append('../')
from config.config import Config
import config.setup as setup
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

cfg = setup.setup(Config)
setup.set_seed(cfg.seed)

# データの読込
test_processed = pd.read_csv(os.path.join(cfg.INPUT, "processed/processed_test.csv"))
test_processed = test_processed.drop(['id', 'goal', 'html_content', cfg.target], axis=1)
submission = pd.read_csv(os.path.join(cfg.INPUT, 'raw/sample_submit.csv'), header=None)

sub_pred = np.zeros((len(test_processed), cfg.num_class), dtype=np.float32)
cfg.lgbm_models = [p for p in sorted(glob(os.path.join(cfg.EXP_MODEL, 'lgbm_fold*.pth')))]

for fold, model_path in enumerate(cfg.lgbm_models):
    model = pickle.load(open(model_path, 'rb'))
    # テストデータで予測
    fold_pred = [[1-pred, pred] for pred in model.predict(test_processed)]
    np.save(os.path.join(cfg.EXP_PREDS, f'lgbm_sub_pred_fold{fold}.npy'), fold_pred)
    # 各 Fold の予測値をアンサンブル
    sub_pred += np.array(fold_pred) / len(cfg.lgbm_models)

np.save(os.path.join(cfg.EXP_PREDS, f'lgbm_sub_pred.npy'), sub_pred)
submission[1] = np.argmax(sub_pred, axis=1)
submission.to_csv(os.path.join(cfg.EXP_PREDS, 'lgbm_submission.csv'), index=False, header=False)
