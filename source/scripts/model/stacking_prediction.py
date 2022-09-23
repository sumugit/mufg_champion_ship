import os
import lightgbm as lgb
from sklearn.metrics import f1_score
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

# 構成のセットアップ
cfg = setup.setup(Config)
setup.set_seed(cfg.seed)
df_train = pd.DataFrame()
df_test = pd.DataFrame()

dataset_train = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_train.csv'))
dataset_train['html_content'] = dataset_train['html_content'].astype(str)
submission = pd.read_csv(os.path.join(cfg.INPUT, 'raw/sample_submit.csv'), header=None)


ensemble_list = ['OUT_EX001', 'OUT_EX002', 'OUT_EX003', 'OUT_EX004', 'OUT_EX005']
for out in ensemble_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    oof_pred = np.load(os.path.join(expxxx_model, 'oof_pred.npy'))
    sub_pred = np.load(os.path.join(expxxx_model, 'sub_pred.npy'))
    df_train[out] = [pred[1]/sum(pred) for pred in oof_pred]
    df_test[out] = [pred[1]/sum(pred) for pred in sub_pred]

lgbm_list = ['OUT_EX003']
for out in lgbm_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    oof_pred = np.load(os.path.join(expxxx_model, 'lgbm_oof_pred.npy'))
    sub_pred = np.load(os.path.join(expxxx_model, 'lgbm_sub_pred.npy'))
    df_train['lgbm'] = [pred[1] for pred in oof_pred]
    df_test['lgbm'] = [pred[1] for pred in sub_pred]

# 目的変数
df_train[cfg.target] = dataset_train[cfg.target]
print(df_train.head())
print(df_test.head())

# stratifiedKFold
cfg.folds = setup.get_stratifiedkfold(df_train, cfg.target, cfg.num_fold, cfg.seed)
# cfg.folds.to_csv(os.path.join(cfg.STACK, "folds_stacking.csv"), header=False)  # fold の index 保存

oof_pred = np.zeros((len(df_train), cfg.num_class), dtype=np.float32)
sub_pred = np.zeros((len(df_test), cfg.num_class), dtype=np.float32)
cfg.lgbm_models = [p for p in sorted(glob(os.path.join(cfg.STACK, 'lgbm_fold*.pth')))]

for fold, model_path in enumerate(cfg.lgbm_models):
    model = pickle.load(open(model_path, 'rb'))
    # テストデータで予測
    fold_train_pred = [[1-pred, pred] for pred in model.predict(df_train.drop([cfg.target], axis=1))]
    fold_test_pred = [[1-pred, pred] for pred in model.predict(df_test)]
    np.save(os.path.join(cfg.STACK, f'lgbm_sub_pred_fold{fold}.npy'), fold_test_pred)
    # 各 Fold の予測値をアンサンブル
    oof_pred += np.array(fold_train_pred) / len(cfg.lgbm_models)
    sub_pred += np.array(fold_test_pred) / len(cfg.lgbm_models)

np.save(os.path.join(cfg.STACK, f'lgbm_oof_pred.npy'), oof_pred)
np.save(os.path.join(cfg.STACK, f'lgbm_sub_pred.npy'), sub_pred)
score = f1_score(dataset_train[cfg.target], np.argmax(oof_pred, axis=1))
print(f'CV : {round(score, 5)}')
submission[1] = np.argmax(sub_pred, axis=1)
submission.to_csv(os.path.join(cfg.STACK, 'stacking_submission.csv'), index=False, header=False)
