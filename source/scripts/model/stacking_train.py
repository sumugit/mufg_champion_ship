import os
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
import sys
sys.path.append('../')
import pickle
from config.config import Config
import config.setup as setup
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 構成のセットアップ
cfg = setup.setup(Config)
setup.set_seed(cfg.seed)
df_train = pd.DataFrame()
df_test = pd.DataFrame()

dataset_train = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_train.csv'))
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_test.csv'))
dataset_train = dataset_train.drop(['id', 'goal', 'html_content'], axis=1)
dataset_test = dataset_test.drop(['id', 'goal', 'html_content'], axis=1)
submission = pd.read_csv(os.path.join(cfg.INPUT, 'raw/sample_submit.csv'), header=None)
# dataset_train['html_content'] = dataset_train['html_content'].astype(str)
# dataset_test['html_content'] = dataset_test['html_content'].astype(str)

ensemble_list = ['OUT_EX001', 'OUT_EX002', 'OUT_EX003', 'OUT_EX004', 'OUT_EX005']
for out in ensemble_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    oof_pred = np.load(os.path.join(expxxx_model, 'oof_pred.npy'))
    sub_pred = np.load(os.path.join(expxxx_model, 'sub_pred.npy'))
    df_train[out] = [pred[1]/sum(pred) for pred in oof_pred]
    df_test[out] = [pred[1]/sum(pred) for pred in sub_pred]

"""
lgbm_list = ['OUT_EX003']
for out in lgbm_list:
    expxxx = os.path.join(cfg.OUTPUT, out)
    expxxx_model = os.path.join(expxxx, 'preds')
    oof_pred = np.load(os.path.join(expxxx_model, 'lgbm_oof_pred.npy'))
    sub_pred = np.load(os.path.join(expxxx_model, 'lgbm_sub_pred.npy'))
    df_train['lgbm'] = [pred[1] for pred in oof_pred]
    df_test['lgbm'] = [pred[1] for pred in sub_pred]
"""

# 結合
df_train = pd.concat([dataset_train, df_train], axis=1)
df_test = pd.concat([dataset_test, df_test], axis=1)


# stratifiedKFold
cfg.folds = setup.get_stratifiedkfold(df_train, cfg.target, cfg.num_fold, cfg.seed)
cfg.folds.to_csv(os.path.join(cfg.STACK, "folds_stacking.csv"), header=False)  # fold の index 保存

scores = []
cat = ['country', 'category1', 'category2']
sub_pred = np.zeros((len(df_test), cfg.num_class), dtype=np.float32)

for fold in cfg.trn_fold:
    train_df = df_train.loc[cfg.folds != fold]
    valid_df = df_train.loc[cfg.folds == fold]
    # カテゴリ変数の変換
    valid_idx = list(valid_df.index)
    train_idx = list(train_df.index)
    X_train, X_valid = train_df.drop([cfg.target], axis=1), valid_df.drop([cfg.target], axis=1)
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
    # 検証データで予測
    y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    y_valid_pred = [1 if val > 0.5 else 0 for val in y_valid_pred]
    score = f1_score(y_valid, y_valid_pred)
    print(f'fold{fold} CV: {score}')
    file = os.path.join(cfg.STACK, f'lgbm_fold{fold}.pth')
    pickle.dump(model, open(file, 'wb'))
    scores.append(score)
    # テストデータで予測
    fold_test_pred = [[1-pred, pred] for pred in model.predict(df_test.drop([cfg.target], axis=1))]
    sub_pred += np.array(fold_test_pred) / cfg.num_fold

print(sub_pred)
print(f'CV: {round(np.mean(scores), 5)}')
submission[1] = np.argmax(sub_pred, axis=1)
submission.to_csv(os.path.join(cfg.STACK, 'stacking2_submission.csv'), index=False, header=False)
