import os
import sys
sys.path.append('../')
from config.config import Config
import config.setup as setup
import warnings
warnings.filterwarnings('ignore')
from transformers import (AutoModel, AutoModelForMaskedLM,
                        AutoTokenizer, LineByLineTextDataset,
                        DataCollatorForLanguageModeling,
                        Trainer, TrainingArguments)

# 構成のセットアップ
cfg = setup.setup(Config)

model = AutoModelForMaskedLM.from_pretrained(cfg.MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=os.path.join(cfg.INPUT, "html_content.txt"),
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=os.path.join(cfg.INPUT, "html_content.txt"),
    block_size=256)

# MLM, mask_prob=0.15
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=cfg.EXP_PRETRAIN,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=500,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
    )


trainer.train()
trainer.save_model(os.path.join(cfg.EXP_PRETRAIN, "mufg-deberta-v3-large"))