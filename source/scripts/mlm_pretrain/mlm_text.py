import sys
sys.path.append('../')
from config.config import Config
import os
import config.setup as setup
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# 構成のセットアップ
cfg = setup.setup(Config)

# データの読込
train_raw = pd.read_csv(os.path.join(cfg.INPUT, 'raw/train.csv'))
test_raw = pd.read_csv(os.path.join(cfg.INPUT, 'raw/test.csv'))
df = pd.concat([train_raw, test_raw])

def to_text(df):
    # 1レコードずつ改行付けて保存.
    df['html_content'] = df['html_content'].astype(str)
    cat1 = df['category1'].tolist()
    cat2 = df['category2'].tolist()
    replaced_texts = [text.replace('\n', '') for text in df['html_content'].tolist()]
    converted_texts = []
    for cat1_, cat2_, text in zip(cat1, cat2, replaced_texts):
        found = text.replace('<div class="contents">', '')[:-6]
        converted_texts.append(cat1_ + ' ' + cat2_ + ' ' + found)
    text = '\n'.join(converted_texts)
    with open(os.path.join(cfg.INPUT, 'html_content.txt'), 'w') as f:
        f.write(str(text))


to_text(df)