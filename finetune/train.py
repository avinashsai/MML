import math
import copy
from tqdm.auto import tqdm

import torch
from transformers import (
    get_scheduler,
    TrainingArguments,
    Trainer
)
from datautils import compute_metrics
from getmodel import load_model

def train_model_using_trainer(
    task,
    modelpath,
    modelname,
    logs_dir,
    numclasses, 
    train_dataloader, 
    val_dataloader, 
    test_dataloader, 
    num_train_epochs,
    per_device_train_batch_size,
    per_device_val_batch_size):

    model = load_model(modelpath, modelname, numclasses)

    if(task == 'cola'):
        best_metric = 'eval_matcoef'
    else:
        best_metric = 'eval_accuracy'

    training_args = TrainingArguments(
        output_dir=logs_dir,
        num_train_epochs=num_train_epochs,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        save_strategy='epoch',
        metric_for_best_model=best_metric,
        greater_is_better=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_val_batch_size,
        logging_dir=logs_dir,
        logging_steps=10000000,
        save_steps=1000000
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        compute_metrics=compute_metrics
        )
    trainer.train()
    testscores = trainer.evaluate(test_dataloader)
    return testscores
