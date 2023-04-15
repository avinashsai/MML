import os
import random
import numpy as np
import torch
import torch.utils.data
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
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
datapath = 'ling/data/'
logs_dir = ''

# Task (can be snli or amazon)
task = 'cc3m_sbu_vg'

# Hyper parameters
if(task == 'amazon'):
    count = '500k'
elif(task == 'cc3m_webvid2m'):
    paths = datapath + task + '.txt'
elif(task == 'cc3m_sbu_vg'):
    paths = datapath + task + '.txt'
elif(task == 'cc12m_coco_sbu_vg'):
    paths = datapath + task + '.txt'

batchsize = 64
numepochs = 10
numhiddenlayers = 6
if(numhiddenlayers == 12):
    save_path = mainpath + 'roberta_' + task
else:
    save_path = mainpath + 'roberta_' + task + '_' + str(numhiddenlayers)

# Path to save the pre-trained model
if(os.path.exists(save_path) == False):
    os.mkdir(save_path)

config = RobertaConfig(
    num_hidden_layers=numhiddenlayers,
    type_vocab_size=2,
)

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', cache_dir=logs_dir)
model = RobertaForMaskedLM(config=config).from_pretrained('roberta-base', cache_dir=logs_dir)

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
