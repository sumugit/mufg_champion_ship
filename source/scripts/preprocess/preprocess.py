from html.parser import HTMLParser
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import sys
sys.path.append('../')
import warnings
import os
import numpy as np
import config.setup as setup
from config.config import Config
warnings.filterwarnings('ignore')


class MyHTMLParser(HTMLParser):
    def __init__(self):
        self.count = defaultdict(int)
        super().__init__()

    def handle_starttag(self, tag, attrs):
        self.count[tag] += 1

    def handle_startendtag(self, tag, attrs):
        self.count[tag] += 1

def count_tags(html):
    parser = MyHTMLParser()
    parser.feed(html)
    return parser.count

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
df['len_html'] = df['html_content'].str.len()
df['word_count'] = df['html_content'].str.split().str.len()
df['inner_link'] = df['html_content'].str.count('href=')
df['num_lines'] = df['html_content'].str.count('\n') + 1
df_goal = df['html_content'].replace('\n', '')

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

# タグの頻度から TF-IDF 値算出
whole_tag_df = pd.DataFrame(map(count_tags, df['html_content'])).fillna(0)
tf_trans = TfidfTransformer()
tag_columns = whole_tag_df.columns
whole_tag_df = pd.DataFrame(tf_trans.fit_transform(whole_tag_df).todense(), columns=tag_columns)
df[tag_columns] = whole_tag_df
df['num_tag'] = whole_tag_df.sum(axis=1)

# goal の分割
df[['goal1', 'goal2']] = df['goal'].str.split('-', expand=True)
df['goal1'] = df['goal1'].str.rstrip('+').fillna(-100).astype(int)
df['goal2'] = df['goal2'].str.rstrip('+').fillna(-100).astype(int)
df['goal_diff'] = df['goal2'] - df['goal1']

# labelEncoding
category_list = ['country', 'category1', 'category2']
for cat in category_list:
    le = LabelEncoder()
    le.fit(df[cat])
    label_encoded_column = le.fit_transform(df[cat])
    df[cat] = pd.Series(label_encoded_column).astype('category')

df_train = df[df['train_flag']==True]
df_train = df_train.drop(['train_flag'], axis=1)
df_train[cfg.target] = df_train[cfg.target].astype('int')
df_test = df[df['train_flag']==False]
df_test = df_test.drop(['train_flag'], axis=1)

print(df.columns.tolist())
df_train.to_csv(os.path.join(cfg.INPUT, "processed/processed_train.csv"), index=False)
df_test.to_csv(os.path.join(cfg.INPUT, "processed/processed_test.csv"), index=False)
