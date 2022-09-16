import re
import nltk
import random
import setup
import pandas as pd
import warnings
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')
import unicodedata
from nltk.corpus import wordnet
from configuration import Config
cfg = setup.setup(Config)


# li タグをランダムに入れ替え
def data_li_exchange_augment(train_df, valid_df):
    contents = []
    for idx, row in train_df.iterrows():
        soup = BeautifulSoup(row["description"], "html.parser")
        txt = soup.find_all("li")
        new_des = ''.join(map(str, random.sample(txt, len(txt))))
        contents.append((row["id"], new_des, row["jobflag"]))
    df = pd.DataFrame(contents, columns=["id", "description", "jobflag"])
    df = pd.concat([train_df, df])
    df = df.drop_duplicates()  # 重複行を除く
    # preprocess
    df["TrainFlag"] = True
    valid_df["TrainFlag"] = False
    df = df.append(valid_df)
    df["description"] = df["description"].map(preprocessing)
    # train
    df_train = df[df["TrainFlag"] == True]
    df_train = df_train.drop(["TrainFlag"], axis=1)
    df_train["jobflag"] = df_train["jobflag"].astype(int) - 1  # ラベルを 0 ~ 3 にする
    # valid
    df_valid = df[df["TrainFlag"] == False]
    df_valid = df_valid.drop(["TrainFlag"], axis=1)
    df_valid["jobflag"] = df_valid["jobflag"].astype(int) - 1  # ラベルを 0 ~ 3 にする
    
    return df_train, df_valid

# mask augmentation
def data_mask_augment(train_df, valid_df):
    train_df["TrainFlag"] = True
    valid_df["TrainFlag"] = False
    df = train_df.append(valid_df)
    df["description"] = df["description"].map(preprocessing)
    # train
    df_train = df[df["TrainFlag"] == True]
    df_train = df_train.drop(["TrainFlag"], axis=1)
    df_train["jobflag"] = df_train["jobflag"].astype(int) - 1  # ラベルを 0 ~ 3 にする
    # valid
    df_valid = df[df["TrainFlag"] == False]
    df_valid = df_valid.drop(["TrainFlag"], axis=1)
    df_valid["jobflag"] = df_valid["jobflag"].astype(int) - 1  # ラベルを 0 ~ 3 にする
    # df_train の一部を mask
    cnts = []
    for idx, row in df_train.iterrows():
        text = row["description"]
        idx_remove = []
        words_list = text.split()
        num_replace_token = int(len(words_list) * cfg.mask_rate)  # [MASK] する token 数
        for _ in range(num_replace_token):
            replace_idx = -1
            # 1 文字または重複 index は除く
            while (replace_idx < 0) or (len(words_list[replace_idx]) == 1) or (replace_idx in idx_remove):
                # 置き換える単語を抽出
                replace_idx = random.randrange(len(words_list))
            # mask token に置き換え
            text = text.replace(words_list[replace_idx], "[MASK]", 1)
            idx_remove.append(replace_idx)
        cnts.append((row["id"], text, row["jobflag"]))
    augmented_df = pd.DataFrame(cnts, columns=["id", "description", "jobflag"])
    df_train = pd.concat([df_train, augmented_df])
    df_train = df_train.drop_duplicates()  # 重複行を除く
    
    return df_train, df_valid

# 正規化
def normalize_text(text):
    text = re.sub('\r', '', text)     # 改行の除去
    text = re.sub('\n', '', text)     # 改行の除去
    text = re.sub('　', '', text)     # 全角空白の除去
    text = re.sub(r'\d+', '0', text)  # 数字文字の一律「0」化
    text = text.replace('e.g.', 'eg')  # 特殊表記の変換
    text = text.replace('eg.', 'eg')  # 特殊表記の変換
    text = text.replace('ie.', 'ie')  # 特殊表記の変換
    text = text.replace('cf.', 'cf')  # 特殊表記の変換
    text = re.sub('\bex.', 'ex', text)  # 特殊表記の変換
    text = text.replace('.', ' . ')   # ピリオドの前後に半角空白
    text = text.replace(',', ' , ')   # カンマの前後に半角空白
    text = text.replace('-', ' ')     # カンマの前後に半角空白
    # 記号の除去
    code_regex = re.compile(
        '[!"#$%&\\\\()*’+–/:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
    text = code_regex.sub('', text)
    return text

# Unicode 正規化
def normalize_unicode(text, form='NFKC'):
    normalized_text = unicodedata.normalize(form, str(text))
    return normalized_text

# html タグの除去
def clean_html_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

# js コードの削除
def clean_html_and_js_tags(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    [x.extract() for x in soup.findAll(['script', 'style'])]
    cleaned_text = soup.get_text()
    cleaned_text = ''.join(cleaned_text.splitlines())
    return cleaned_text

# url の削除
def clean_url(html_text):
    cleaned_text = re.sub(r'http\S+', '', html_text)
    return cleaned_text

# 全て小文字統一
def lower_text(text):
    return text.lower()

# Lemmatizing
def lemmatize_term(term, pos=None):
    if term == "has" or term == "as":
        return term
    elif pos is None:
        synsets = wordnet.synsets(term)
        if not synsets:
            return term
        pos = synsets[0].pos()
        if pos == wordnet.ADJ_SAT:
            pos = wordnet.ADJ
    return nltk.WordNetLemmatizer().lemmatize(term, pos=pos)

# 前処理まとめ
def preprocessing(text):
    text = normalize_unicode(text)
    text = clean_html_tags(text)
    text = clean_html_and_js_tags(text)
    text = clean_url(text)
    text = normalize_text(text)
    text = lower_text(text)
    text = ' '.join(lemmatize_term(e) for e in text.split())
    return str(text.strip())