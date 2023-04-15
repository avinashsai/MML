from tqdm import tqdm
import torch
from sklearn.metrics import *

def merge_sentences(sentences):
    x = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        new_sentence = sentence[0]
        if(type(sentence) == tuple):
            if(type(sentence[1]) == list):
                for s in sentence[1]:
                    new_sentence += " " + s
            else:
                new_sentence += " " + sentence[1]
        
        x.append(new_sentence)
    
    return x

class TaskDataset(torch.utils.data.Dataset):
  def __init__(self, inputids, attnmasks, tokentypeids, labels):
    self.inputids = inputids
    self.attnmasks = attnmasks
    self.tokentypeids = tokentypeids
    self.labels = labels

  def __getitem__(self, idx):
    item = {}
    item['input_ids'] = self.inputids[idx]
    item['attention_mask'] = self.attnmasks[idx]
    item['token_type_ids'] = self.tokentypeids[idx]
    item['labels'] = self.labels[idx]
    return item

  def __len__(self):
    return len(self.labels)

def tokenize_sentences(tokenizer, sentences, maxlen):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    issplitwords = False
    for i in tqdm(range(len(sentences))):
        sentence = sentences[i]
        sentence1 = None
        sentence2 = None
        if(type(sentence) == tuple):
            sentence1, sentence2 = sentence
            if(type(sentence[1]) == list):
                issplitwords = True
            else:
                sentence2 = str(sentence2)
        else:
            sentence1 = sentence
        
        sentence1 = str(sentence1)
        encoded = tokenizer.encode_plus(
            text=sentence1,
            text_pair=sentence2,
            add_special_tokens=True,
            padding='max_length',
            max_length=maxlen,
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
            )

        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        token_type_ids.append(encoded['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0, out=None)
    attention_masks = torch.cat(attention_masks, dim=0, out=None)
    token_type_ids = torch.cat(token_type_ids, dim=0, out=None)

    return input_ids, attention_masks, token_type_ids

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    avg = 'macro'
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=avg)
    acc = accuracy_score(labels, preds) * 100
    mat_coef = matthews_corrcoef(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1 * 100,
        'precision': precision,
        'recall': recall,
        'matcoef': mat_coef
    }
