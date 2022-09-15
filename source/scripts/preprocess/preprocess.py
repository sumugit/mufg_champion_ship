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
df_train = df[df['train_flag']==True]
df_train = df_train.drop(['train_flag'], axis=1)
df_test = df[df['train_flag']==False]
df_test = df_test.drop(['train_flag'], axis=1)

df_train.to_csv(os.path.join(cfg.INPUT, "processed/processed_train.csv"))
df_test.to_csv(os.path.join(cfg.INPUT, "processed/processed_test.csv"))
