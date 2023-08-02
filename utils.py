import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModel,
                          BertForSequenceClassification,
                          ElectraForSequenceClassification)
from transformers.optimization import get_cosine_schedule_with_warmup
import json
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


def get_tokenizer_pad_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(['[PAD]'])


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


def load_json_data(json_path):
    with open(json_path, 'r', encoding='UTF8') as f:
        data = json.load(f)
    return data


def slice_2d_tensors(tensors, tokens_per_word,
                     token_pad_emb, word_pad_emb,
                     max_tokens_per_word, max_words_per_sent):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tensors: [max_length, hidden_size], [150, 768]
    # tokens_per_word: [max_words_per_sent], [75]
    start_row = 0
    sliced_lst = []

    for tokens in tokens_per_word:
        if int(tokens) == 0: break
        end_row = start_row + int(tokens)
        sliced = tensors[start_row:end_row, :]

        # Pad & Truncate in token level
        while sliced.size(0) < max_tokens_per_word:
            sliced = sliced.to(device)
            token_pad_emb = token_pad_emb.to(device)
            sliced = torch.cat((sliced, token_pad_emb), dim=0)
        if sliced.size(0) > max_tokens_per_word:
            sliced = sliced[:max_tokens_per_word]

        start_row = end_row
        sliced_lst.append(sliced)
    sliced_lst = torch.stack(sliced_lst)
    
    # Pad & Truncate in word level
    while sliced_lst.size(0) < max_words_per_sent:
        sliced_lst = sliced_lst.to(device)
        word_pad_emb = word_pad_emb.to(device)
        sliced_lst = torch.cat((sliced_lst, word_pad_emb), dim=0) 
    if sliced_lst.size(0) > max_words_per_sent:
        sliced_lst = sliced_lst[:max_words_per_sent]
    return sliced_lst
    # sliced_lst: [max_words_per_sent, max_tokens_per_word, hidden_size], [75, 10, 768]


def slice_3d_tensors(tensors, tokens_per_word,
                     token_pad_emb, word_pad_emb,
                     max_tokens_per_word, max_words_per_sent):
    # tensors: [batch_size, max_length, hidden_size], [256, 150, 768]
    # tokens_per_word: [batch_size, max_words_per_sent], [256, 75]
    batch_size = tensors.size(0)
    all_sliced_lst = []

    for i in range(batch_size):
        tensors_ = tensors[i]
        tokens_per_word_ = tokens_per_word[i]

        sliced_lst = slice_2d_tensors(tensors_, tokens_per_word_,
                                      token_pad_emb, word_pad_emb,
                                      max_tokens_per_word, max_words_per_sent)
        all_sliced_lst.append(sliced_lst)
    return torch.stack(all_sliced_lst)
    # all_sliced_lst: [batch_size, max_words_per_sent, max_tokens_per_word, hidden_size], [256, 75, 10, 768]


def calculate_accuracy(x, y):
    assert x.shape == y.shape
    assert x.dim() == y.dim() == 2

    issame = (x == y).tolist()
    issame = sum(issame, [])
    return issame.count(True) / len(issame)


def compact_name(model_path):
    if model_path == 'beomi/KcELECTRA-base-v2022':
        return 'KcELECTRAv2022'