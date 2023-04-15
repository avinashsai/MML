import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, labels, portionsplit=0.15):
    traindata, testdata, trainlabels, testlabels = train_test_split(data, labels, test_size=portionsplit, random_state=42)
    return traindata, testdata, trainlabels, testlabels

def load_boolq(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)

    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    with open(train_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        train_sentences.append((result['passage'], result['question']))
        if(result['label'] == False):
            train_labels.append(0)
        else:
            train_labels.append(1)

    with open(test_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        test_sentences.append((result['passage'], result['question']))
        if(result['label'] == False):
            test_labels.append(0)
        else:
            test_labels.append(1)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_wic(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)

    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    with open(train_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        train_sentences.append((result['sentence1'], [result['sentence2'], result['word']]))
        if(result['label'] == False):
            train_labels.append(0)
        else:
            train_labels.append(1)

    with open(test_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        test_sentences.append((result['sentence1'], [result['sentence2'], result['word']]))
        if(result['label'] == False):
            test_labels.append(0)
        else:
            test_labels.append(1)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_cb(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)

    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    with open(train_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        train_sentences.append((result['premise'], result['hypothesis']))
        if(result['label'] == "contradiction"):
            train_labels.append(0)
        elif(result['label'] == "entailment"):
            train_labels.append(1)
        else:
            train_labels.append(2)

    with open(test_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        test_sentences.append((result['premise'], result['hypothesis']))
        if(result['label'] == "contradiction"):
            test_labels.append(0)
        elif(result['label'] == "entailment"):
            test_labels.append(1)
        else:
            test_labels.append(2)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_copa(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)

    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    with open(train_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        train_sentences.append((result['premise'], [result['choice1'], result['choice2']]))
        train_labels.append(int(result['label']))

    with open(test_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        test_sentences.append((result['premise'], [result['choice1'], result['choice2']]))
        test_labels.append(int(result['label']))

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_wsc(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)

    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []
    with open(train_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        train_sentences.append((result['text'], [result['target']['span1_text'], result['target']['span2_text']]))
        if(result['label'] == False):
            train_labels.append(0)
        else:
            train_labels.append(1)

    with open(test_filepath, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        test_sentences.append((result['text'], [result['target']['span1_text'], result['target']['span2_text']]))
        if(result['label'] == False):
            test_labels.append(0)
        else:
            test_labels.append(1)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]
