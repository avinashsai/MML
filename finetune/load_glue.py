import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, labels, portionsplit=0.15):
    traindata, testdata, trainlabels, testlabels = train_test_split(data, labels, test_size=portionsplit, random_state=42)
    return traindata, testdata, trainlabels, testlabels

def load_sst2(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_df = pd.read_csv(train_filepath, sep='	')
    test_df = pd.read_csv(test_filepath, sep='	')

    train_sentences = train_df.sentence.tolist()
    train_labels = train_df.label.tolist()
    test_sentences = test_df.sentence.tolist()
    test_labels = test_df.label.tolist()

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences, 
        train_labels,
        test_size=0.15, random_state=42)
    
    return [train_sentences, train_labels, val_sentences, val_labels, test_sentences, test_labels]

def load_qqp(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_df = pd.read_csv(train_filepath, sep='	')
    test_df = pd.read_csv(test_filepath, sep='	')

    train_sentences1 = train_df.question1.tolist()
    train_sentences2 = train_df.question2.tolist()
    train_labels = train_df.is_duplicate.tolist()

    test_sentences1 = test_df.question1.tolist()
    test_sentences2 = test_df.question2.tolist()
    test_labels = test_df.is_duplicate.tolist()

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_labels, val_labels = train_test_split(
        train_sentences1,
        train_sentences2, 
        train_labels,
        test_size=0.15, random_state=42)

    train_sentences = []
    for s1, s2 in zip(train_sentences1, train_sentences2):
        train_sentences.append((s1, s2))
    
    val_sentences = []
    for s1, s2 in zip(val_sentences1, val_sentences2):
        val_sentences.append((s1, s2))
    
    test_sentences = []
    for s1, s2 in zip(test_sentences1, test_sentences2):
        test_sentences.append((s1, s2))
    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x

def load_rte(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_df = pd.read_csv(train_filepath, sep='	')
    test_df = pd.read_csv(test_filepath, sep='	')

    train_df = train_df.astype(str)
    test_df = test_df.astype(str)

    train_sentences1 = train_df.sentence1.tolist()
    train_sentences2 = train_df.sentence2.tolist()
    train_labels = []
    for i in range(len(train_df)):
        if(train_df['label'][i].startswith('entailment')):
            train_labels.append(1)
        else:
            train_labels.append(0)

    test_sentences1 = test_df.sentence1.tolist()
    test_sentences2 = test_df.sentence2.tolist()
    test_labels = []
    for i in range(len(test_df)):
        if(test_df['label'][i].startswith('entailment')):
            test_labels.append(1)
        else:
            test_labels.append(0)

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_labels, val_labels = train_test_split(
        train_sentences1,
        train_sentences2, 
        train_labels,
        test_size=0.15, random_state=42)

    train_sentences = []
    for s1, s2 in zip(train_sentences1, train_sentences2):
        train_sentences.append((s1, s2))
    
    val_sentences = []
    for s1, s2 in zip(val_sentences1, val_sentences2):
        val_sentences.append((s1, s2))
    
    test_sentences = []
    for s1, s2 in zip(test_sentences1, test_sentences2):
        test_sentences.append((s1, s2))
    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x

def load_cola(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []  
    with open(train_filepath, 'r', encoding='latin-1') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            train_sentences.append(x[-1])
            train_labels.append(int(x[1]))
    
    with open(test_filepath, 'r', encoding='latin-1') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            test_sentences.append(x[-1])
            test_labels.append(int(x[1]))

    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        train_sentences,
        train_labels,
        test_size=0.15, random_state=42)

    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x


def load_mnli_matched(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_sentences1 = []
    train_sentences2 = []
    train_labels = []
    test_sentences1 = []
    test_sentences2 = []
    test_labels = []  
    with open(train_filepath, 'r', encoding='latin-1') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            train_sentences1.append(x[8])
            train_sentences2.append(x[9])
            if(x[-1].startswith('entailment')):
                train_labels.append(1)
            elif(x[-1].startswith('contradiction')):
                train_labels.append(0)
            else:
                train_labels.append(2)
    
    with open(test_filepath, 'r', encoding='latin-1') as f:
        for line in f.readlines():
            x = line.split('	')
            test_sentences1.append(x[8])
            test_sentences2.append(x[9])
            if(x[-1].startswith('entailment')):
                test_labels.append(1)
            elif(x[-1].startswith('contradiction')):
                test_labels.append(0)
            else:
                test_labels.append(2)

    train_sentences1 = train_sentences1[1:]
    train_sentences2 = train_sentences2[1:]
    test_sentences1 = test_sentences1[1:]
    test_sentences2 = test_sentences2[1:]

    train_labels = train_labels[1:]
    test_labels = test_labels[1:]

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_labels, val_labels = train_test_split(
        train_sentences1,
        train_sentences2, 
        train_labels,
        test_size=0.15, random_state=42)

    train_sentences = []
    for s1, s2 in zip(train_sentences1, train_sentences2):
        train_sentences.append((s1, s2))
    
    val_sentences = []
    for s1, s2 in zip(val_sentences1, val_sentences2):
        val_sentences.append((s1, s2))
    
    test_sentences = []
    for s1, s2 in zip(test_sentences1, test_sentences2):
        test_sentences.append((s1, s2))
    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x

def load_mnli_mismatched(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_sentences1 = []
    train_sentences2 = []
    train_labels = []
    test_sentences1 = []
    test_sentences2 = []
    test_labels = []  
    with open(train_filepath, 'r', encoding='latin-1') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            train_sentences1.append(x[8])
            train_sentences2.append(x[9])
            if(x[-1].startswith('entailment')):
                train_labels.append(1)
            elif(x[-1].startswith('contradiction')):
                train_labels.append(0)
            else:
                train_labels.append(2)
    
    with open(test_filepath, 'r', encoding='latin-1') as f:
        for line in f.readlines():
            x = line.split('	')
            test_sentences1.append(x[8])
            test_sentences2.append(x[9])
            if(x[-1].startswith('entailment')):
                test_labels.append(1)
            elif(x[-1].startswith('contradiction')):
                test_labels.append(0)
            else:
                test_labels.append(2)

    train_sentences1 = train_sentences1[1:]
    train_sentences2 = train_sentences2[1:]
    test_sentences1 = test_sentences1[1:]
    test_sentences2 = test_sentences2[1:]

    train_labels = train_labels[1:]
    test_labels = test_labels[1:]

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_labels, val_labels = train_test_split(
        train_sentences1,
        train_sentences2, 
        train_labels,
        test_size=0.15, random_state=42)

    train_sentences = []
    for s1, s2 in zip(train_sentences1, train_sentences2):
        train_sentences.append((s1, s2))
    
    val_sentences = []
    for s1, s2 in zip(val_sentences1, val_sentences2):
        val_sentences.append((s1, s2))
    
    test_sentences = []
    for s1, s2 in zip(test_sentences1, test_sentences2):
        test_sentences.append((s1, s2))
    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x

def load_mrpc(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)

    train_sentences1 = []
    train_sentences2 = []
    train_labels = []
    with open(train_filepath, 'r') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            if(x[0] != "0" and x[0] != "1"):
                continue
            train_labels.append(int(x[0]))
            train_sentences1.append(x[-2])
            train_sentences2.append(x[-1])

    test_sentences1 = []
    test_sentences2 = []
    test_labels = []
    with open(test_filepath, 'r') as f:
        for line in f.readlines():
            x = line[:-1].split('	')
            if(x[0] != "0" and x[0] != "1"):
                continue
            test_labels.append(int(x[0]))
            test_sentences1.append(x[-2])
            test_sentences2.append(x[-1])

    print(train_sentences1[0:5])
    print(train_sentences2[0:5])
    print(train_labels[0:5])

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_labels, val_labels = train_test_split(
        train_sentences1,
        train_sentences2, 
        train_labels,
        test_size=0.15, random_state=42)

    train_sentences = []
    for s1, s2 in zip(train_sentences1, train_sentences2):
        train_sentences.append((s1, s2))
    
    val_sentences = []
    for s1, s2 in zip(val_sentences1, val_sentences2):
        val_sentences.append((s1, s2))
    
    test_sentences = []
    for s1, s2 in zip(test_sentences1, test_sentences2):
        test_sentences.append((s1, s2))
    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x

def load_wnli(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_df = pd.read_csv(train_filepath, sep='	')
    test_df = pd.read_csv(test_filepath, sep='	')

    train_sentences1 = train_df.sentence1.tolist()
    train_sentences2 = train_df.sentence2.tolist()
    train_labels = train_df.label.tolist()

    test_sentences1 = test_df.sentence1.tolist()
    test_sentences2 = test_df.sentence2.tolist()
    test_labels = test_df.label.tolist()

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_labels, val_labels = train_test_split(
        train_sentences1,
        train_sentences2, 
        train_labels,
        test_size=0.15, random_state=42)

    train_sentences = []
    for s1, s2 in zip(train_sentences1, train_sentences2):
        train_sentences.append((s1, s2))
    
    val_sentences = []
    for s1, s2 in zip(val_sentences1, val_sentences2):
        val_sentences.append((s1, s2))
    
    test_sentences = []
    for s1, s2 in zip(test_sentences1, test_sentences2):
        test_sentences.append((s1, s2))
    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x

def load_qnli(datapath, trainfile, testfile):
    train_filepath = os.path.join(datapath, trainfile)
    test_filepath = os.path.join(datapath, testfile)
    train_df = pd.read_csv(train_filepath, sep='\t')
    test_df = pd.read_csv(test_filepath, sep='\t')

    train_sentences1 = train_df.question.tolist()
    train_sentences2 = train_df.sentence.tolist()
    train_labels = []
    for i in range(len(train_df)):
        if(train_df['label'][i].startswith('entailment')):
            train_labels.append(1)
        else:
            train_labels.append(0)

    test_sentences1 = test_df.question.tolist()
    test_sentences2 = test_df.sentence.tolist()
    test_labels = []
    for i in range(len(test_df)):
        if(test_df['label'][i].startswith('entailment')):
            test_labels.append(1)
        else:
            test_labels.append(0)

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_labels, val_labels = train_test_split(
        train_sentences1,
        train_sentences2, 
        train_labels,
        test_size=0.15, random_state=42)

    train_sentences = []
    for s1, s2 in zip(train_sentences1, train_sentences2):
        train_sentences.append((s1, s2))
    
    val_sentences = []
    for s1, s2 in zip(val_sentences1, val_sentences2):
        val_sentences.append((s1, s2))
    
    test_sentences = []
    for s1, s2 in zip(test_sentences1, test_sentences2):
        test_sentences.append((s1, s2))
    x = []
    x += [train_sentences, train_labels]
    x += [val_sentences, val_labels]
    x += [test_sentences, test_labels]
    return x