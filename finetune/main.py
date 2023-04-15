import os
import argparse
import random
import logging
logging.disable(logging.INFO) 
logging.disable(logging.WARNING)
os.environ['TRANSFORMERS_CACHE'] = '/playpen-storage/avinashm/hg/'
import numpy as np
import pandas as pd
import torch

from load_glue import *
from load_cs import *
from load_superglue import *

from datautils import tokenize_sentences, TaskDataset, merge_sentences
from train import train_model_using_trainer

from transformers import (
    AutoTokenizer,
    CLIPTokenizerFast
)

print(torch.cuda.is_available())
main_data_path = '/playpen-storage/avinashm/Experiments/ling/data/'

logs_dir = '/playpen-storage/avinashm/Experiments/ling/finetune/logs'
resultspath = '/playpen-storage/avinashm/Experiments/ling/finetune/results'

if(os.path.exists(logs_dir) == False):
    os.mkdir(logs_dir)

if(os.path.exists(resultspath) == False):
    os.mkdir(resultspath)

max_sen_len_glue = 200
max_sen_len_cs = 300
max_sen_len_superglue = 250
n_gpu = 1

task_to_files = {'sst2': {'folder':'SST-2', 'trainfile': 'train.tsv', 'testfile': 'dev.tsv', 'numclasses': 2, 
                          'senlen': max_sen_len_glue},
                'qqp': {'folder':'QQP', 'trainfile': 'train.tsv', 'testfile': 'dev.tsv', 'numclasses': 2, 
                        'senlen': max_sen_len_glue},
                'rte': {'folder':'RTE', 'trainfile': 'train.tsv', 'testfile': 'dev.tsv', 'numclasses': 2, 
                        'senlen': max_sen_len_glue},
                'cola': {'folder':'CoLA', 'trainfile': 'train.tsv', 'testfile': 'dev.tsv', 'numclasses': 2, 
                         'senlen': max_sen_len_glue},
                'mnli': {'folder':'MNLI', 'trainfile': 'train.tsv', 'testfile': 'dev_matched.tsv', 'numclasses': 3, 
                        'senlen': max_sen_len_glue},
                'mnli_mis': {'folder':'MNLI', 'trainfile': 'train.tsv', 'testfile': 'dev_mismatched.tsv', 'numclasses': 3, 
                            'senlen': max_sen_len_glue},
                'qnli': {'folder':'QNLI', 'trainfile': 'train.tsv', 'testfile': 'dev.tsv', 'numclasses': 2, 
                         'senlen': max_sen_len_glue},
                'mrpc': {'folder': 'MRPC', 'trainfile': 'msr_paraphrase_train.txt', 'testfile': 'msr_paraphrase_test.txt',
                        'numclasses': 2, 'senlen': max_sen_len_glue},
                'wnli': {'folder':'WNLI', 'trainfile': 'train.tsv', 'testfile': 'dev.tsv', 'numclasses': 2, 
                         'senlen': max_sen_len_glue},
                'socialiqa': {'folder':'socialiqa', 'trainfile': ['train.jsonl','train-labels.txt'], 
                            'testfile': ['dev.jsonl','dev-labels.txt'], 'numclasses': 3, 'senlen': max_sen_len_cs},
                'cosmosqa': {'folder':'cosmosqa', 'trainfile': ['train.jsonl','train-labels.txt'], 
                            'testfile': ['valid.jsonl','valid-labels.txt'], 'numclasses': 4, 'senlen': max_sen_len_cs},
                'winogrande': {'folder':'winogrande', 'trainfile': ['train_xl.jsonl','train_xl-labels.lst'], 
                            'testfile': ['dev.jsonl','dev-labels.lst'], 'numclasses': 2, 'senlen': max_sen_len_cs},
                'codah': {'folder':'codah', 'trainfile': ['train.tsv', 'dev.tsv'], 
                            'testfile': 'test.tsv', 'numclasses': 4, 'senlen': max_sen_len_cs},
                'hellaswag': {'folder':'hellaswag', 'trainfile': 'hellaswag_train.jsonl', 
                            'testfile': 'hellaswag_val.jsonl', 'numclasses': 4, 'senlen': max_sen_len_cs},
                'paws': {'folder':'paws', 'trainfile': ['train.tsv', 'dev.tsv'], 
                            'testfile': 'test.tsv', 'numclasses': 2, 'senlen': max_sen_len_cs},
                'boolq': {'folder':'BoolQ', 'trainfile': 'train.jsonl', 
                            'testfile': 'val.jsonl', 'numclasses': 2, 'senlen': max_sen_len_superglue},
                'wic': {'folder':'WiC', 'trainfile': 'train.jsonl', 
                            'testfile': 'val.jsonl', 'numclasses': 2, 'senlen': max_sen_len_superglue},
                'cb': {'folder':'CB', 'trainfile': 'train.jsonl', 
                            'testfile': 'val.jsonl', 'numclasses': 3, 'senlen': max_sen_len_superglue},
                'copa': {'folder':'COPA', 'trainfile': 'train.jsonl', 
                            'testfile': 'val.jsonl', 'numclasses': 2, 'senlen': max_sen_len_superglue},
                'wsc': {'folder':'WSC', 'trainfile': 'train.jsonl', 
                            'testfile': 'val.jsonl', 'numclasses': 2, 'senlen': max_sen_len_superglue}}

original_dict = {'alpro': ['bert-base-uncased', 'alpro.pt'],
                'albef': ['bert-base-uncased', 'albef.pth'],
                'blip': ['bert-base-uncased', 'blip.pth'],
                'fit': ['distilbert-base-uncased', 'fit.pth'],
                'meter': ['roberta-base', 'meter.ckpt'],
                'violet': ['bert-base-uncased', 'violet.pt'],
                'clip': ['clip-vit-base-patch32', 'clip-vit-base-patch32']}

pretrained_dict = {'pretrain_alpro_bert_cc3m_webvid2m_6': 'bert-base-uncased',
                'pretrain_albef_bert_cc12m_coco_sbu_vg': 'bert-base-uncased',
                'pretrain_blip_bert_cc12m_coco_sbu_vg': 'bert-base-uncased',
                'pretrain_fit_distillbert_cc3m_webvid2m': 'distilbert-base-uncased',
                'pretrain_meter_roberta_cc3m_sbu_vg_6': 'roberta-base',
                'pretrain_violet_bert_yt10m_cc3m_webvid2m': 'bert-base-uncased'}


def main():
    parser = argparse.ArgumentParser(description='MML')
    parser.add_argument('--modelname', type=str, default='bert-small',
                        help='Model to run')
    parser.add_argument('--typ', type=str,  help='Type of tasks (glue/superglue/cs)')

    args = parser.parse_args()
    modelname = args.modelname
    typ = args.typ

    data_path = os.path.join(main_data_path, typ)

    save_res_dir = os.path.join(resultspath, modelname)
    if(os.path.exists(save_res_dir) == False):
        os.mkdir(save_res_dir)

    if(modelname.startswith('pretrain')):
        submodelpath = '/playpen-storage/avinashm/Experiments/ling/pretrain/weights'
        x = modelname.split("_")[2:]
        submodelname = x[0]
        for x1 in x[1:]:
            submodelname += "_"
            submodelname += x1
        modelpath = os.path.join(submodelpath, submodelname)
    else:
        submodelpath = '/playpen-storage/avinashm/Experiments/ling/premodels'
        submodelname = original_dict[modelname][1]
        modelpath = os.path.join(submodelpath, submodelname)

    print("#############################################################################################")
    print("Model Path: {} ".format(modelpath))

    if(typ == 'glue'):
        tasks = ['mnli', 'mnli_mis', 'qqp', 'sst2', 'mrpc', 'cola', 'rte', 'wnli']
        num_train_epochs = 5
    elif(typ == 'cs'):
        tasks = ['socialiqa', 'cosmosqa', 'winogrande', 'codah', 'hellaswag']
        num_train_epochs = 1
    elif(typ == 'superglue'):
        tasks = ['boolq', 'wic', 'cb', 'copa', 'wsc']
        num_train_epochs = 25

    tasks_mean_acc = []
    tasks_std_acc = []
    tasks_mean_f1 = []
    tasks_std_f1 = []
    tasks_mean_mat = []
    tasks_std_mat = []
    for task in tasks:
        print("#############################################################################################")
        print("Task: {} ".format(task))
        print("#############################################################################################")
        folderpath = os.path.join(data_path, task_to_files[task]['folder'])
        
        #seeds = [6035, 6792, 9061, 4599, 3496]
        seeds = [6035]
        print("Maximum Sentence Length: {} and Number of Epochs: {} ".format(task_to_files[task]['senlen'], num_train_epochs))
        print("Number of runs: {} ".format(len(seeds)))
        each_task_acc = []
        each_task_f1 = []
        each_task_mat = []
        for SEED in seeds:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if(n_gpu > 1):
                torch.cuda.manual_seed_all(SEED)

            if(task == 'sst2'):
                x = load_sst2(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 32
            elif(task == 'qqp'):
                x = load_qqp(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'rte'):
                x = load_rte(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 32
            elif(task == 'mnli'):
                x = load_mnli_matched(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'mnli_mis'):
                x = load_mnli_mismatched(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'cola'):
                x = load_cola(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 32
            elif(task == 'mrpc'):
                x = load_mrpc(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'wnli'):
                x = load_wnli(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 32
            elif(task == 'qnli'):
                x = load_qnli(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'socialiqa'):
                x = load_socialqa(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'cosmosqa'):
                x = load_cosmosqa(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'winogrande'):
                x = load_winogrande(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'codah'):
                x = load_codah(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'hellaswag'):
                x = load_hellaswag(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 64
            elif(task == 'boolq'):
                x = load_boolq(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 32
            elif(task == 'wic'):
                x = load_wic(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 32
            elif(task == 'cb'):
                x = load_cb(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 16
            elif(task == 'copa'):
                x = load_copa(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 8
            elif(task == 'wsc'):
                x = load_wsc(folderpath, task_to_files[task]['trainfile'], task_to_files[task]['testfile'])
                per_device_train_batch_size = per_device_val_batch_size = 8

            train_sentences, train_labels = x[0], x[1]
            val_sentences, val_labels = x[2], x[3]
            test_sentences, test_labels = x[4], x[5]

            print("#############################################################################################")
            print("Training data size: {} ".format(len(train_sentences)))
            print("Validation data size: {} ".format(len(val_sentences)))
            print("Testing data size: {} ".format(len(test_sentences)))

            if('fit' in modelname or 'clip' in modelname):
                train_sentences = merge_sentences(train_sentences)
                val_sentences = merge_sentences(val_sentences)
                test_sentences = merge_sentences(test_sentences)
            

            if(modelname.startswith('pretrain')):
                tokenizer = AutoTokenizer.from_pretrained(pretrained_dict[modelname],
                            do_lower_case=True,
                            cache_dir=logs_dir)
            else:
                if('clip' in modelname):
                    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                        do_lower_case=True,
                        cache_dir=logs_dir)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(original_dict[modelname][0],
                        do_lower_case=True,
                        cache_dir=logs_dir)

            numclasses = task_to_files[task]['numclasses']
            max_sen_len = task_to_files[task]['senlen']

            if('clip' in modelname):
                max_sen_len = 76

            train_inputids, train_attnmasks, train_tok_typ_ids = tokenize_sentences(tokenizer, train_sentences, max_sen_len)
            val_inputids, val_attnmasks, val_tok_typ_ids = tokenize_sentences(tokenizer, val_sentences, max_sen_len)
            test_inputids, test_attnmasks, test_tok_typ_ids = tokenize_sentences(tokenizer, test_sentences, max_sen_len)

            train_dataset = TaskDataset(train_inputids, train_attnmasks, train_tok_typ_ids, train_labels)
            val_dataset = TaskDataset(val_inputids, val_attnmasks, val_tok_typ_ids, val_labels)
            test_dataset = TaskDataset(test_inputids, test_attnmasks, test_tok_typ_ids, test_labels)

            testscores = train_model_using_trainer(
                task,
                modelpath, modelname, logs_dir, numclasses, train_dataset, 
                val_dataset, 
                test_dataset, 
                num_train_epochs,
                per_device_train_batch_size,
                per_device_val_batch_size
                )
            
            each_task_acc.append(testscores['eval_accuracy'])
            each_task_f1.append(testscores['eval_f1'])
            each_task_mat.append(testscores['eval_matcoef'])

        tasks_mean_acc.append(np.mean(each_task_acc))
        tasks_mean_f1.append(np.mean(each_task_f1))
        tasks_mean_mat.append(np.mean(each_task_mat))

        tasks_std_acc.append(np.std(each_task_acc))
        tasks_std_f1.append(np.std(each_task_f1))
        tasks_std_mat.append(np.std(each_task_mat))

    df = pd.DataFrame({'task': tasks, 'accuracy_mean': tasks_mean_acc, 'accuracy_std': tasks_std_acc,
                        'f1_mean': tasks_mean_f1, 'f1_std': tasks_std_f1, 'mat_mean': tasks_mean_mat,
                        'mat_std': tasks_std_mat})

    df.to_csv(os.path.join(save_res_dir, typ + '_' + modelname + '.csv'), header=True, index=False)

if __name__ == '__main__':
    main()