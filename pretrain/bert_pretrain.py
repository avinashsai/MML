import os
import random
import numpy as np
import torch
import torch.utils.data
from transformers import BertConfig, BertForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizerFast
from transformers import BertForMaskedLM
from transformers import LineByLineTextDataset
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
mainpath = 'pretrain/weights/'
datapath = 'data/'
logs_dir = ''

# Task (can be snli or amazon)
task = 'yt180m'

# Hyper parameters
if(task == 'amazon'):
    count = '500k'
elif(task == 'cc3m_webvid2m'):
    paths = datapath + task + '.txt'
elif(task == 'cc12m_coco_sbu_vg'):
    paths = datapath + task + '.txt'
elif(task == 'yt180m'):
    paths = datapath + task + '.txt'

batchsize = 8
numepochs = 5
numhiddenlayers = 12
lr = 2e-5
weightdecay = 1e-3
modelname = 'bert-base-uncased'
if(numhiddenlayers == 12):
    save_path = mainpath + 'bert_' + task
else:
    save_path = mainpath + 'bert_' + task + '_' + str(numhiddenlayers)

# Path to save the pre-trained model
if(os.path.exists(save_path) == False):
    os.mkdir(save_path)

config = BertConfig(
    num_hidden_layers=numhiddenlayers,
    type_vocab_size=2,
)

tokenizer = BertTokenizerFast.from_pretrained(modelname, cache_dir=logs_dir)
model = BertForMaskedLM(config=config).from_pretrained(modelname, cache_dir=logs_dir)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=paths,
    block_size=512,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=save_path,
    overwrite_output_dir=True,
    num_train_epochs=numepochs,
    learning_rate=lr,
    weight_decay=weightdecay,
    per_device_train_batch_size=batchsize,
    save_steps=10000000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model(save_path)
