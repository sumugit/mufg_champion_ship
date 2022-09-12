import sys
sys.path.append('../')
import warnings
import os
from config.setup import setup
from config.config import Config
import pandas as pd
warnings.filterwarnings('ignore')


# 構成のセットアップ
cfg = setup.setup(Config)

# データの読込
train_raw = pd.read_csv(os.path.join(cfg.INPUT, "raw/train.csv"))
test_raw = pd.read_csv(os.path.join(cfg.INPUT, "raw/test.csv"))
df = pd.concat([train_raw, test_raw])