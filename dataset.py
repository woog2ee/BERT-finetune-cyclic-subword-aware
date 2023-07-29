import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np



class CustomDataset(Dataset):
    def __init__(self, tokenizer, max_length, subword_aware, data):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subword_aware = subword_aware
        self.dataset = list(zip([d[1] for d in data], [d[0] for d in data]))[:500]


    def tokenize(self, sent, padding=True):
        if padding:
            tokenized = self.tokenizer(sent,
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_length)
        else:
            tokenized = self.tokenizer.tokenize(sent)
        return tokenized


    def get_word_level_binary_labels(self, sent1, sent2):
        words1, words2 = sent1.split(' '), sent2.split(' ')  
        labels = [1 if w1 != w2 else 0 for w1, w2 in zip(words1, words2)]
        return labels


    def get_token_level_binary_labels(self, sent1, sent2):
        word_level_binary_labels = self.get_word_level_binary_labels(sent1, sent2)
        word_token_lengths = [len(self.tokenize(word, False)) for word in sent1.split(' ')]
        
        labels = [[word_level_binary_labels[i]] * word_token_lengths[i]
                  for i in range(len(word_level_binary_labels))]
        labels = sum(labels, [])

        labels += [0] * (self.max_length - len(labels))
        assert len(labels) == self.max_length
        return labels


    def __len__(self):
        return len(self.dataset)

    
    def __getitem__(self, idx):
        augmented, origin = self.dataset[idx][0], self.dataset[idx][1]
        
        tokenized = self.tokenize(augmented)
        input, attn_mask = tokenized['input_ids'], tokenized['attention_mask']
        
        if self.subword_aware:
            label = self.get_word_level_binary_labels(augmented, origin)
        else:
            label = self.get_token_level_binary_labels(augmented, origin)

        output = {'input': input, 'attn_mask': attn_mask, 'label': label}
        return {k: torch.tensor(v) for k, v in output.items()}