import os
import gc #メモリ開放
import sys
sys.path.append('../')
import config.setup as setup
import torch
import model.prediction as prediction
import numpy as np
import torch.nn as nn
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from model.fgm import FGM
from model.awp import AWP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler # GPU 計算の効率化, 高速化
from bert_base import BERTDataset, BERTModel
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from model.data_augmentation import data_li_exchange_augment, data_mask_augment

# LLDR (Layer-wise Learning Rate Decay)
def get_optimizer_grouped_parameters(
    cfg,
    model
):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": cfg.lr,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    # print(model.backbone) # モデルの layer
    layers = [model.backbone.embeddings] + \
        list(model.backbone.encoder.layer)
    layers.reverse()
    lr = cfg.lr
    for layer in layers:
        lr *= cfg.lldr  # top layers から bottom layers にかけて学習率を下げる
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters

# -- 学習 --
def training(cfg, train, test):
    setup.set_seed(cfg.seed)
    # 全 epoch 学習後の最良モデルにおけるクラスの予測値格納
    oof_pred = np.zeros((len(train), cfg.num_class), dtype=np.float32)

    # 損失関数: クロスエントロピー誤差
    # criterion = nn.CrossEntropyLoss(label_smoothing=cfg.epsilon)  # label smoothing あり
    criterion = nn.CrossEntropyLoss()  # label smoothing なし

    # StratifiedKFold
    for fold in cfg.trn_fold:
        cnt = 0  # Peudo Labeling のカウンタ
        percentage = 0  # test label の確信度 (> confidence)
        while cnt < 2:
            # データの分割, Dataset作成
            if cnt == 0:
                train_df = train.loc[cfg.folds != fold]
                valid_df = train.loc[cfg.folds == fold]
                # Data Augmentation
                # train_df, valid_df = data_li_exchange_augment(train_df, valid_df)
                # train_df, valid_df = data_mask_augment(train_df, valid_df)
                valid_idx = list(valid_df.index)
            train_idx = list(train_df.index)
            # 学習データ
            train_dataset = BERTDataset(
                cfg,
                train_df['html_content'].to_numpy(),
                train_df[cfg.target].to_numpy(),
            )
            # 検証データ
            valid_dataset = BERTDataset(
                cfg,
                valid_df['html_content'].to_numpy(),
                valid_df[cfg.target].to_numpy()
            )
            # Dataloader作成
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=False
            )

            # 検証での最良値
            best_val_preds = None
            best_val_score = -1

            # modelの読込
            model = BERTModel(cfg, criterion)
            model = model.to(cfg.device)                # GPU設定
            # model.init_weights(model)                 # Layer Re Initialization

            # Optimizerの設定
            # named_parameters: パラメータ名, パラメータの値 (多次元) を返す
            # reference: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # weight decay を適用しない項 (バイアス, レイヤ正規化)
            optimizer_grouped_parameters = []
            # weight decay させる parameter
            optimizer_grouped_parameters.append({
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': cfg.weight_decay
            })
            # weight decay させない parameter
            optimizer_grouped_parameters.append({
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            })
            ### 追記 ###
            grouped_optimizer_params = get_optimizer_grouped_parameters(
                cfg, model
            )
            optimizer = AdamW(
                grouped_optimizer_params,
                lr=cfg.lr,
                betas=cfg.beta,
                weight_decay=cfg.weight_decay,
            )
            ### 終了 ###
            """
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=cfg.lr,
                betas=cfg.beta,
                weight_decay=cfg.weight_decay,
            )
            """
            # Schedulerの設定
            # The total number of training steps. (train のサイズ × epoch 数)
            num_train_optimization_steps = int(
                len(train_loader) * cfg.n_epochs // cfg.gradient_accumulation_steps
            )
            # The number of steps for the warmup phase. (lr を更新するタイミング)
            num_warmup_steps = int(
                num_train_optimization_steps * cfg.num_warmup_steps_rate)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_optimization_steps
            )
            num_eval_step = len(train_loader) // cfg.num_eval + cfg.num_eval

            # 学習開始
            for epoch in range(cfg.n_epochs):
                print(f"# ============ start epoch:{epoch} ============== #")
                # train モードに切り替え
                model.train()
                # val_losses_batch = []
                scaler = GradScaler()
                fgm = FGM(model)
                ### AWP ###
                """
                awp = AWP(
                    model,
                    optimizer,
                    adv_lr=cfg.adv_lr,
                    adv_eps=cfg.adv_eps,
                    start_epoch=cfg.start_epoch,
                    scaler=scaler
                )
                """
                ### AWP ###
                # progress bar は pbar と表記することが多いみたい...
                with tqdm(train_loader, total=len(train_loader), disable=cfg.disable) as pbar:
                    # batch 毎の処理 (batch_size は DataLoader で定義済み)
                    for step, (encoding, labels) in enumerate(pbar):
                        # encoding の数合わせ
                        encoding, max_len = setup.collatte(encoding)
                        # GPU 設定
                        for k, v in encoding.items():
                            encoding[k] = v.to(cfg.device)
                        labels = labels.to(cfg.device)
                        # 全ての tensor の勾配を 0 に初期化
                        # (自動微分の Autograd で支えられている)
                        optimizer.zero_grad()
                        with autocast():
                            # 出力値 (各クラスの確率) と loss の取得
                            output, loss = model(encoding, labels)
                        # tqdm の pbar 右の loss, lr を動的に更新
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'lr': scheduler.get_lr()[0]
                        })
                        # Gradient Accumulation
                        if cfg.gradient_accumulation_steps > 1:
                            loss = loss / cfg.gradient_accumulation_steps
                        # 損失関数の勾配を計算 (誤差逆伝播法)
                        scaler.scale(loss).backward()
                        # Adversarial training
                        # embedding layer に敵対的な摂動を加える
                        fgm.attack()
                        ### AWP ###
                        """
                        awp.attack_backward(encoding, labels, epoch)
                        # 敵対的な摂動を加えられた状態での損失を計算
                        
                        with autocast():
                            adv_output, adv_loss = model(encoding, labels)
                        pbar.set_postfix({
                            'adv_loss': adv_loss.item(),
                            'lr': scheduler.get_lr()[0]
                        })
                        if cfg.gradient_accumulation_steps > 1:
                            adv_loss = adv_loss / cfg.gradient_accumulation_steps
                        # 損失関数の勾配を計算 (誤差逆伝播法)
                        scaler.scale(adv_loss).backward()
                        """
                        fgm.restore()
                        # awp.restore()
                        ### AWP ###
                        # Clipping gradients
                        if cfg.clip_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                cfg.clip_grad_norm
                            )
                        if (step+1) % cfg.gradient_accumulation_steps == 0:
                            # 学習率の更新 (実行しないと更新されない)
                            scaler.step(optimizer)
                            scaler.update()
                            scheduler.step()

                # evaluating
                val_preds = []
                val_losses = []
                val_nums = []
                model.eval()
                with torch.no_grad():
                    with tqdm(valid_loader, total=len(valid_loader), disable=cfg.disable) as pbar:
                        for (encoding, labels) in pbar:
                            encoding, max_len = setup.collatte(encoding)
                            for k, v in encoding.items():
                                encoding[k] = v.to(cfg.device)
                            labels = labels.to(cfg.device)
                            with autocast():
                                output, loss = model(encoding, labels)
                            output = output.sigmoid().detach().cpu().numpy()
                            val_preds.append(output)
                            val_losses.append(loss.item() * len(labels))
                            val_nums.append(len(labels))
                            pbar.set_postfix({
                                'val_loss': loss.item()
                            })

                val_preds = np.concatenate(val_preds)  # 各レコードの 2 クラス予測値を格納
                val_loss = sum(val_losses) / sum(val_nums)
                # validation 評価
                score = f1_score(np.argmax(val_preds, axis=1),
                                valid_df[cfg.target])
                val_log = {
                    'val_loss': val_loss,
                    'score': score,
                }
                print(val_log)
                # 最良モデルの更新
                if best_val_score < score:
                    print("save model weight")
                    best_val_preds = val_preds
                    best_val_score = score
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.EXP_MODEL, f"fold{fold}.pth")
                    )
            # BERTの推論
            cfg.model_fold_weight = os.path.join(cfg.EXP_MODEL, f"fold{fold}.pth")
            train_df, percentage = prediction.pseudo_inferring(cfg, train_df, test)
            cnt += 1
        # end while
        # validation の index の目的変数を最良モデルで予測
        oof_pred[valid_idx] = best_val_preds.astype(np.float32)
        # 最良モデル (kFold) の保存
        np.save(os.path.join(cfg.EXP_PREDS,
                f'oof_pred_fold{fold}.npy'), best_val_preds)
        del model
        gc.collect()

    # 全ての KFold から得られた予測値を保存
    np.save(os.path.join(cfg.EXP_PREDS, 'oof_pred.npy'), oof_pred)
    score = f1_score(np.argmax(oof_pred, axis=1),
                    train[cfg.target])
    print('CV:', round(score, 5))
    return score