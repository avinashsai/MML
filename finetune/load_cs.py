import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, labels, portionsplit=0.15):
    traindata, testdata, trainlabels, testlabels = train_test_split(data, labels, test_size=portionsplit, random_state=42)
    return traindata, testdata, trainlabels, testlabels

def load_socialqa(datapath, trainfiles, testfiles):
    trainqa_filepath = os.path.join(datapath, trainfiles[0])
    trainlab_filepath = os.path.join(datapath, trainfiles[1])
    testqa_filepath = os.path.join(datapath, testfiles[0])
    testlab_filepath = os.path.join(datapath, testfiles[1])

    with open(trainqa_filepath, 'r') as json_file:
        train_json_list = list(json_file)

    train_sentences = []
    for d in train_json_list:
        result = json.loads(d)
        con = result['context']
        ques = result['question']
        train_sentences.append((str(con), [str(ques), str(result['answerA']), str(result['answerB']), str(result['answerC'])]))

    train_labels = []
    with open(trainlab_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_labels.append(int(line[:-1]) - 1)

    with open(testqa_filepath, 'r') as json_file:
        test_json_list = list(json_file)

    test_sentences = []
    for d in test_json_list:
        result = json.loads(d)
        con = result['context']
        ques = result['question']
        test_sentences.append((str(con), [str(ques), str(result['answerA']), str(result['answerB']), str(result['answerC'])]))

    test_labels = []
    with open(testlab_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_labels.append(int(line[:-1]) - 1)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_hellaswag(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)

    with open(train_filepath, 'r') as json_file:
        train_json_list = list(json_file)

    train_sentences = []
    train_labels = []
    for d in train_json_list:
        result = json.loads(d)
        con = result['ctx']
        x = [str(p) for p in result["endings"]]
        train_sentences.append((str(con), x))
        train_labels.append(int(result['label']))

    with open(test_filepath, 'r') as json_file:
        test_json_list = list(json_file)

    test_sentences = []
    test_labels = []
    for d in test_json_list:
        result = json.loads(d)
        con = result['ctx']
        x = [str(p) for p in result["endings"]]
        test_sentences.append((str(con), x))
        test_labels.append(int(result['label']))

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_cosmosqa(datapath, trainfiles, testfiles):
    trainqa_filepath = os.path.join(datapath, trainfiles[0])
    trainlab_filepath = os.path.join(datapath, trainfiles[1])
    testqa_filepath = os.path.join(datapath, testfiles[0])
    testlab_filepath = os.path.join(datapath, testfiles[1])

    with open(trainqa_filepath, 'r') as json_file:
        train_json_list = list(json_file)

    train_sentences = []
    for d in train_json_list:
        result = json.loads(d)
        con = result['context']
        ques = result['question']
        x = [str(ques), str(result['answer0']), str(result['answer1']), str(result['answer2']), str(result['answer3'])]
        train_sentences.append((str(con), x))

    train_labels = []
    with open(trainlab_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_labels.append(int(line[:-1]))

    with open(testqa_filepath, 'r') as json_file:
        test_json_list = list(json_file)

    test_sentences = []
    for d in test_json_list:
        result = json.loads(d)
        con = result['context']
        ques = result['question']
        x = [str(ques), str(result['answer0']), str(result['answer1']), str(result['answer2']), str(result['answer3'])]
        test_sentences.append((str(con), x))

    test_labels = []
    with open(testlab_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_labels.append(int(line[:-1]))

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_winogrande(datapath, trainfiles, testfiles):
    trainqa_filepath = os.path.join(datapath, trainfiles[0])
    trainlab_filepath = os.path.join(datapath, trainfiles[1])
    testqa_filepath = os.path.join(datapath, testfiles[0])
    testlab_filepath = os.path.join(datapath, testfiles[1])

    with open(trainqa_filepath, 'r') as json_file:
        train_json_list = list(json_file)

    train_sentences = []
    for d in train_json_list:
        result = json.loads(d)
        sentence = result['sentence']
        sentence1 = sentence.replace('_',  result['option1'])
        sentence2 = sentence.replace('_', result['option2'])
        train_sentences.append((sentence1, sentence2))
        

    train_labels = []
    with open(trainlab_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_labels.append(int(line[:-1]) - 1)

    with open(testqa_filepath, 'r') as json_file:
        test_json_list = list(json_file)

    test_sentences = []
    for d in test_json_list:
        result = json.loads(d)
        sentence = result['sentence']
        sentence1 = sentence.replace('_',  result['option1'])
        sentence2 = sentence.replace('_', result['option2'])
        test_sentences.append((sentence1, sentence2))

    test_labels = []
    with open(testlab_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            test_labels.append(int(line[:-1]) - 1)

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_codah(datapath, trainfiles, testfile):
    train_filepath = os.path.join(datapath, trainfiles[0])
    dev_filepath = os.path.join(datapath, trainfiles[1])
    test_filepath = os.path.join(datapath, testfile)

    train_sentences = []
    train_labels = []
    with open(train_filepath, 'r') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            train_sentences.append((str(x[1]), [str(x[2]), str(x[3]), str(x[4]), str(x[5])]))
            train_labels.append(int(x[-1]))

    val_sentences = []
    val_labels = []
    with open(dev_filepath, 'r') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            val_sentences.append((str(x[1]), [str(x[2]), str(x[3]), str(x[4]), str(x[5])]))
            val_labels.append(int(x[-1]))

    test_sentences = []
    test_labels = []
    with open(test_filepath, 'r') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            test_sentences.append((str(x[1]), [str(x[2]), str(x[3]), str(x[4]), str(x[5])]))
            test_labels.append(int(x[-1]))

    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]
