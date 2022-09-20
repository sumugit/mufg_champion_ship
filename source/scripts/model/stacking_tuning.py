import os
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
import sys
sys.path.append('../')
from config.config import Config
import config.setup as setup
import pandas as pd
from hyperopt import hp, tpe, Trials, fmin
import warnings
warnings.filterwarnings('ignore')

# 構成のセットアップ
cfg = setup.setup(Config)
setup.set_seed(cfg.seed)
df_train = pd.DataFrame()
df_test = pd.DataFrame()

dataset_train = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_train.csv'))
dataset_test = pd.read_csv(os.path.join(cfg.INPUT, 'processed/processed_test.csv'))
dataset_train['html_content'] = dataset_train['html_content'].astype(str)
dataset_test['html_content'] = dataset_test['html_content'].astype(str)

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


# stratifiedKFold
cfg.folds = setup.get_stratifiedkfold(df_train, cfg.target, cfg.num_fold, cfg.seed)
cfg.folds.to_csv(os.path.join(cfg.STACK, "folds_stacking.csv"), header=False)  # fold の index 保存

def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'max_depth': int(params['max_depth']),
        'reg_lambda': params['reg_lambda'],
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 1000,  # 決定木の数は十分多くする
        'learning_rate': 0.1,  # チューニングでは学習率大きくする
        'verbose': -1,
        'num_iteration': 100
    }
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=lgb_eval,
        num_boost_round=1000,  # 学習の反復数 (epoch 数)
        verbose_eval=-1,
        early_stopping_rounds=50
    )
    y_valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    y_valid_pred = [1 if val > 0.5 else 0 for val in y_valid_pred]
    score = f1_score(y_valid, y_valid_pred)
    
    return -1*score  # 最小化したいので -1 倍


for fold in cfg.trn_fold:
    train_df = df_train.loc[cfg.folds != fold]
    valid_df = df_train.loc[cfg.folds == fold]
    # カテゴリ変数の変換
    valid_idx = list(valid_df.index)
    train_idx = list(train_df.index)
    X_train, X_valid = train_df.drop([cfg.target], axis=1), valid_df.drop([cfg.target], axis=1)
    y_train, y_valid = train_df[cfg.target], valid_df[cfg.target]
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, y_valid, free_raw_data=False)
    
    # hyperopt でパラメタチューニング
    params_hyperopt = {
        'num_leaves': hp.quniform('num_leaves', 5, 40, 2),
        'max_depth': hp.quniform('max_depth', 5, 20, 2),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(1))
    }

    best = fmin(
        objective,
        space=params_hyperopt,
        algo=tpe.suggest,
        max_evals=200,
        verbose=-1,
        rstate=np.random.default_rng(cfg.seed)  # 乱数固定
    )
    
    best = {
        'num_leaves': int(best['num_leaves']),
        'max_depth': int(best['max_depth']),
        'reg_lambda': best['reg_lambda'],
    }
    print(best)
    break  # fold1 のみで tuning