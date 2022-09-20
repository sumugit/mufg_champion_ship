class Config:
    """ path or parameter configration """
    # 学習するモデルの読み込み
    MODEL_PATH = 'microsoft/deberta-v3-large'
    MODEL_PATH2 = '/home/sumiya/signate/mufg/source/pretrain/deberta_v3_large_5fold_2022/mufg-deberta-v3-large/'
    # ベースとなるディレクトリパスの指定
    PATH = '/home/sumiya/signate/mufg/source'
    CURRENT_PATH = PATH
    # パラメータ設定
    target = 'state'                # 目的変数
    num_class = 2                   # クラス数
    seed = 2022                     # seed 値
    num_fold = 5                    # CV 分割数
    trn_fold = range(5)             # Fold
    batch_size = 8                  # batct_size の設定
    n_epochs = 5                    # epoch 数の設定
    max_len = 256                   # token 数の最大の長さの設定
    lr = 2e-5                       # 学習率の設定
    weight_decay = 2e-5             # Optimizer の設定
    beta = (0.9, 0.98)              # AdamW のハイパラ
    num_warmup_steps_rate = 0.01    # lr を更新するタイミング
    clip_grad_norm = None           # 勾配が大きくならないように, 閾値を超えていたら修正する
    gradient_accumulation_steps = 1 # 勾配を更新するのに必要な step 数
    num_eval = 1                    # evaluation の step 数計算に利用 (今回は不要)
    gamma = 2.0                     # FocalLoss のハイパラ
    epsilon = 0.1                   # Label Smoothing のパラメータ
    threshold = 0.90                # Pseudo Labeling の閾値
    confidence = 0.92               # test label における予測の確信度
    num_sample_tokens = 1           # Masked Augmentation で 1 文書あたりに水増しする文書数
    mask_rate = 0.20                # Masked Augmentation で Mask する単語の割合
    disable = False                 # tqdm プログレスバー表示設定
    lldr = 0.9                      # LLRD での各層の learning rate 縮小比率
    start_epoch = 0                 # AWP を導入する epoch のタイミング
    adv_lr = 1.0                    # AWP のハイパラ
    adv_eps = 0.20                  # AWP の Adversarial attack の探索範囲
