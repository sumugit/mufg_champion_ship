import os
from sklearn.metrics import f1_score
import lightgbm as lgb
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

# データの読込
train_processed = pd.read_csv(os.path.join(cfg.INPUT, "processed/processed_train.csv"))
test_processed = pd.read_csv(os.path.join(cfg.INPUT, "processed/processed_test.csv"))
train_processed = train_processed.drop(['id', 'goal', 'html_content'], axis=1)
test_processed = test_processed.drop(['id', 'goal', 'html_content'], axis=1)
# train_processed['country'] = train_processed['country'].astype('category')
# test_processed['country'] = test_processed['country'].astype('category')

# stratifiedKFold
cfg.folds = setup.get_stratifiedkfold(
    train_processed, cfg.target, cfg.num_fold, cfg.seed)
cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, "folds_lightgbm.csv"), header=False)  # fold の index 保存


# scores = []
cat = ['country', 'category1', 'category2']
oof_pred = np.zeros((len(train_processed), cfg.num_class), dtype=np.float32)
sub_pred = np.zeros((len(test_processed), cfg.num_class), dtype=np.float32)

for fold in cfg.trn_fold:
    train_df = train_processed.loc[cfg.folds != fold]
    valid_df = train_processed.loc[cfg.folds == fold]
    # カテゴリ変数の変換
    valid_idx = list(valid_df.index)
    train_idx = list(train_df.index)
    X_train, X_valid = train_df.drop(
        [cfg.target], axis=1), valid_df.drop([cfg.target], axis=1)
    y_train, y_valid = train_df[cfg.target], valid_df[cfg.target]
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature=cat, free_raw_data=False)

    params = {
        'num_leaves': 64,
        'max_depth': 5,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'verbose': -1,
        'num_iteration': 100
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=1000,
        verbose_eval=-1,
        early_stopping_rounds=50
    )
    y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    y_valid_pred = [1 if val > 0.5 else 0 for val in y_valid_pred]
    score = f1_score(y_valid, y_valid_pred)
    print(f'fold{fold} CV : {round(score,5)}')
    
    file = os.path.join(cfg.EXP_MODEL, f'lgbm_fold{fold}.pth')
    pickle.dump(model, open(file, 'wb'))
    # scores.append(score)
    # 各 Fold の予測値をアンサンブル
    # テストデータで予測
    fold_train_pred = [[1-pred, pred] for pred in model.predict(train_processed.drop([cfg.target], axis=1))]
    fold_test_pred = [[1-pred, pred] for pred in model.predict(test_processed.drop([cfg.target], axis=1))]
    # 各 Fold の予測値をアンサンブル
    oof_pred += np.array(fold_train_pred) / cfg.num_fold
    sub_pred += np.array(fold_test_pred) / cfg.num_fold

np.save(os.path.join(cfg.EXP_PREDS, f'lgbm_oof_pred.npy'), oof_pred)
np.save(os.path.join(cfg.EXP_PREDS, f'lgbm_sub_pred.npy'), sub_pred)
score = f1_score(train_processed[cfg.target], np.argmax(oof_pred, axis=1))
print(f'CV : {round(score, 5)}')
# print(f'CV : {round(np.mean(scores),5)}')

