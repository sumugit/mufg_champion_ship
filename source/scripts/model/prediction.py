import os
import gc  # メモリ開放
import sys
sys.path.append('../')
import config.setup as setup
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast  # GPU 計算の効率化, 高速化
from bert_base import BERTDataset, BERTModel


# テストデータでの予測 (Seudo Labeling)
def pseudo_inferring(cfg, fold_train, test):
    # dataset, dataloaderの作成
    test_dataset = BERTDataset(
        cfg,
        test['html_content'].to_numpy()
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True
    )
    model = BERTModel(cfg)
    model.load_state_dict(torch.load(cfg.model_fold_weight))
    model = model.to(cfg.device)
    # モデルを評価モードに移行
    model.eval()
    test_pred = []
    with torch.no_grad():
        # 予測
        for encoding in tqdm(test_loader, total=len(test_loader), disable=cfg.disable):
            encoding, max_len = setup.collatte(encoding)
            for k, v in encoding.items():
                encoding[k] = v.to(cfg.device)
            with autocast():
                output = model(encoding)
            output = output.softmax(axis=1).detach().cpu().numpy()
            test_pred.append(output)
    # バッチ毎 → 個別のレコード毎
    test_pred = np.concatenate(test_pred)
    # Pseudo Labeliing
    exceeded = np.array([*map(lambda x: max(x), test_pred)]) > cfg.threshold  # 閾値を超えたかどうかの True/False リスト
    percentage = sum(exceeded) / len(test_pred)
    print(f"The percentage of predictions over threshold is {percentage * 100} %")   
    test[cfg.target] = np.argmax(test_pred, axis=1)
    pl_test = test[exceeded]
    augemented_train = pd.concat([pl_test, fold_train])
    # augemented_train=augemented_train[augemented_train.columns[::-1]]
    return shuffle(augemented_train), percentage


# テストデータでの最終予測 (アンサンブル)
def inferring(cfg, test):
    # 各Foldで学習させた最良モデルの重み path 表示
    print('\n'.join(cfg.model_weights))
    sub_pred = np.zeros((len(test), 4), dtype=np.float32)
    # 各Foldで学習したモデルの重みを読込
    for fold, model_weight in enumerate(cfg.model_weights):
        # dataset, dataloaderの作成
        test_dataset = BERTDataset(
            cfg,
            test[cfg.target].to_numpy()
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=True
        )
        model = BERTModel(cfg)
        model.load_state_dict(torch.load(model_weight))
        model = model.to(cfg.device)
        # モデルを評価モードに移行
        model.eval()
        fold_pred = []
        with torch.no_grad():
            # 予測
            for encoding in tqdm(test_loader, total=len(test_loader), disable=cfg.disable):
                encoding, max_len = setup.collatte(encoding)
                for k, v in encoding.items():
                    encoding[k] = v.to(cfg.device)
                with autocast():
                    output = model(encoding)
                output = output.softmax(axis=1).detach().cpu().numpy()
                fold_pred.append(output)
        fold_pred = np.concatenate(fold_pred)
        
        np.save(os.path.join(cfg.EXP_PREDS,
                f'sub_pred_fold{fold}.npy'), fold_pred)
        # 各 Fold の予測値をアンサンブル
        sub_pred += fold_pred / len(cfg.model_weights)
        del model
        gc.collect()
    np.save(os.path.join(cfg.EXP_PREDS, f'sub_pred.npy'), sub_pred)
    return sub_pred