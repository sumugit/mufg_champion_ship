import torch
import torch.nn as nn
from torch.utils.data import Dataset 
from transformers import AutoConfig, AutoModel

""" データの符号化, tensor化 """
class BERTDataset(Dataset):
    def __init__(self, cfg, texts, labels=None):
        self.cfg = cfg
        self.texts = texts
        self.labels = labels

    # -- 組み込み関数 len() を呼び出した時に実行されるメソッド --
    # tests のサイズ
    def __len__(self):
        return len(self.texts)

    # -- 要素を参照するときに呼び出されるメソッド --
    # texts の index 番目の文書
    def __getitem__(self, index):
        # 参照文書を tokenizing
        inputs = self.prepare_encoding(self.cfg, self.texts[index])
        # 訓練, 検証データ
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.int64)
            return inputs, label
        # テストデータ
        else:
            return inputs

    @staticmethod
    # 文書の符号化
    def prepare_encoding(cfg, text):
        inputs = cfg.tokenizer(
            text,
            add_special_tokens=True,
            max_length=cfg.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=False,
        )
        # inputs のキー:
        # input_ids, token_type_ids, attention_mask
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs


""" データの符号化, tensor化 """
class BERTModel(nn.Module):
    def __init__(self, cfg, criterion=None):
        super().__init__()
        self.cfg = cfg              # configuration
        self.criterion = criterion  # 損失 (callback 関数)
        # 事前学習済みモデルに設定されたパラメータ構成の読込
        self.config = AutoConfig.from_pretrained(
            cfg.MODEL_PATH,
            output_hidden_states=True
        )
        # 事前学習モデルの読込
        self.backbone = AutoModel.from_pretrained(
            cfg.MODEL_PATH2,
            config=self.config
        )
        # 全結合: 出力が 4クラス (次元) となるように線形変換
        # ここあとで変更してみる
        self.fc = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, cfg.num_class),
        )
        # 全結合層 (Linear) の重み初期化
        # nn.init.normal_(self.fc[1].weight, std=0.02)
        # nn.init.zeros_(self.fc[1].bias)


    def init_weights(self, module):
        """ Layer Re Initialization """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        


    # 出力と損失 (FocalLoss) を計算
    def forward(self, inputs, labels=None, fl=None):
        outputs = self.backbone(**inputs)["last_hidden_state"]
        outputs = outputs[:, 0, :]
        if labels is not None:
            logits = self.fc(outputs)
            loss = self.criterion(logits, labels)
            if fl is not None:
                pt = torch.exp(-loss)
                focal_loss = ((1-pt) ** self.cfg.gamma * loss).mean()
                return logits, focal_loss
            else:
                return logits, loss
        else:
            logits = self.fc(outputs)
            return logits