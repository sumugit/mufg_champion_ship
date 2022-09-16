from sklearn.preprocessing import LabelEncoder
import pandas as pd
import sys
sys.path.append('../')
import warnings
import os
import numpy as np
import config.setup as setup
from config.config import Config
warnings.filterwarnings('ignore')

# 構成のセットアップ
cfg = setup.setup(Config)

# データの読込
train_raw = pd.read_csv(os.path.join(cfg.INPUT, "raw/train.csv"))
test_raw = pd.read_csv(os.path.join(cfg.INPUT, "raw/test.csv"))
train_raw['train_flag'] = True
test_raw['train_flag'] = False
df = pd.concat([train_raw, test_raw])

df_goal = df['goal'].replace('100000+', '100000-100000').str.split('-', expand=True)
df_goal.rename(columns={0: 'goal_inf', 1: 'goal_sup'}, inplace=True)
df['mid_goal'] = ((df_goal['goal_inf'].astype(int) + df_goal['goal_sup'].astype(int))/2).astype(int)
df['word_count'] = df['html_content'].str.split().str.len()
df['inner_link'] = df['html_content'].str.count('href=')
df['num_lines'] = df['html_content'].str.count('\n') + 1
df_goal = df['html_content'].replace('\n', '')
# labelEncoding
le = LabelEncoder()
le.fit(df['country'])
label_encoded_column = le.fit_transform(df['country'])
df['country'] = pd.Series(label_encoded_column).astype('category')

df['html_content'] = df['html_content'].astype(str)
cat1 = df['category1'].tolist()
cat2 = df['category2'].tolist()
replaced_texts = [text.replace('\n', '')
                for text in df['html_content'].tolist()]
converted_texts = []
for cat1_, cat2_, text in zip(cat1, cat2, replaced_texts):
    found = text.replace('<div class="contents">', '')[:-6]
    converted_texts.append(cat1_ + ' ' + cat2_ + ' ' + found)
df['html_content'] = converted_texts

df_train = df[df['train_flag']==True]
df_train = df_train.drop(['train_flag'], axis=1)
df_train[cfg.target] = df_train[cfg.target].astype('int')
df_test = df[df['train_flag']==False]
df_test = df_test.drop(['train_flag'], axis=1)

df_train.to_csv(os.path.join(cfg.INPUT, "processed/processed_train.csv"), index=False)
df_test.to_csv(os.path.join(cfg.INPUT, "processed/processed_test.csv"), index=False)
