import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModel,
                          BertForSequenceClassification,
                          ElectraForSequenceClassification)
from transformers.optimization import get_cosine_schedule_with_warmup
from operator import itemgetter



def str2bool(str):
    if str == 'true':
        return True
    elif str == 'false':
        return False
    else:
        raise 'Invalid arguments, boolean value expected.'
    

def get_model_and_tokenizer(model_path, num_labels):
    if 'bert' in model_path.lower():
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    elif 'electra' in model_path.lower():
        #model = ElectraForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
        model = AutoModel.from_pretrained(model_path)
    else:
        raise 'Invalid model_path, not even related to BERT or ELECTRA.'

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def get_optimizer_and_scheduler(model, lr, warmup_steps, t_total):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler


def get_train_valid_test_indices(data_length):
    test_indices  = [0 + i*10 for i in range(data_length//10)]
    valid_indices = [1 + i*10 for i in range(data_length//10)]
    train_indices = list(set(range(data_length)) - set(valid_indices) - set(test_indices))
    return train_indices, valid_indices, test_indices


def get_train_valid_test_data(data, train_indices, valid_indices, test_indices):
    train_data = list(itemgetter(*train_indices)(data))
    valid_data = list(itemgetter(*valid_indices)(data))
    test_data  = list(itemgetter(*test_indices)(data)) 
    return train_data, valid_data, test_data


def calculate_accuracy(x, y):
    assert x.shape == y.shape
    assert x.dim() == y.dim() == 2

    issame = (x == y).tolist()
    issame = sum(issame, [])
    return issame.count(True) / len(issame)


def compact_name(model_path):
    if model_path == 'beomi/KcELECTRA-base-v2022':
        return 'KcELECTRAv2022'