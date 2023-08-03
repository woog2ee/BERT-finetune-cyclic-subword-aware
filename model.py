import torch
import torch.nn as nn
import os
from utils import (slice_3d_tensors, compact_name)



class CustomClassifier(nn.Module):
    def __init__(self, device, model, model_path, hidden_size, num_labels,
                 subword_aware, pad_id, max_tokens_per_word, max_words_per_sent, dropout=0.3):
        super().__init__()
        if 'bert' in model_path.lower():
            self.model_type = 'bert'
        elif 'electra' in model_path.lower():
            self.model_type = 'electra'

        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.subword_aware = subword_aware
        self.token_pad_emb = self.get_trained_token_embedding(model, pad_id).to(device)
        self.word_pad_emb = self.token_pad_emb.repeat(max_tokens_per_word, 1).unsqueeze(0).to(device)
        
        self.max_tokens_per_word = max_tokens_per_word
        self.max_words_per_sent = max_words_per_sent

        if self.subword_aware:
            self.embedder = SubwordAwareEmbedder(hidden_size,
                                                 max_tokens_per_word, max_words_per_sent)
        # token_pad_emb: [1, hidden_size]
        # word_pad_emb = [1, max_tokens_per_word, hidden_size]


    def forward(self, token_ids, tokens_per_word):
        # token_ids: [batch_size, max_length]
        # tokens_per_word: [batch_size, words_per_sent]
      
        if self.model_type == 'electra':
            out = self.model(token_ids).last_hidden_state
            # last_hidden_state: [batch_size, max_length, hidden_size]
        
        if self.subword_aware:
            sliced = slice_3d_tensors(out, tokens_per_word,
                                      self.token_pad_emb, self.word_pad_emb,
                                      self.max_tokens_per_word, self.max_words_per_sent)
            # sliced: [batch_size, max_words_per_sent, max_tokens_per_word, hidden_size]
            out = self.embedder(sliced)
            # last_hidden_state: [batch_size, max_words_per_sent, hidden_size]

        out = self.dropout(out)
        return torch.sigmoid(self.classifier(out))
        # classified: [batch_size, max_length, num_labels]
        # classified: [batch_size, max_words_per_sent, num_labels]


    def get_trained_token_embedding(self, model, idx):
        return model.embeddings.word_embeddings.weight[idx]
        # token_embedding: [1, hidden_size]



class SubwordAwareEmbedder(nn.Module):
    def __init__(self, hidden_size,
                 max_tokens_per_word, max_words_per_sent, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_tokens_per_word = max_tokens_per_word
        self.max_words_per_sent = max_words_per_sent

        self.model = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=2,
                             bias=True,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=True)
        self.weights = nn.Parameter(torch.empty(hidden_size*2, hidden_size))
        self.register_parameter('weights', self.weights)

        self.init_weight(self.model)
        self.init_weight(self.weights)


    def forward(self, input):
        # input: [batch_size, max_words_per_sent, max_tokens_per_word, hidden_size]
        batch_size = input.size(0)

        reshaped = input.view(-1, self.max_tokens_per_word, self.hidden_size)
        # reshaped: [-1, max_tokens_per_word, hidden_size]

        out, h_n = self.model(reshaped)

        forward_out, backward_out = torch.split(out, self.hidden_size, dim=2)
        # each_out: [-1, max_tokens_per_word, hidden_size]

        forward_last_hidden = forward_out[:, -1, :]
        backward_last_hidden = backward_out[:, 0, :]
        # each_last_hidden: [-1, hidden_size]

        concat = torch.cat((forward_last_hidden, backward_last_hidden), dim=1)
        combined = concat @ self.weights
        # combined: [-1, hidden_size]

        return combined.view(batch_size, self.max_words_per_sent, -1)
        # out: [batch_size, max_words_per_sent, hidden_size]



    def init_weight(self, layer):
        if 'Parameter' in str(type(layer)):
            return nn.init.uniform_(layer, -1.0, 1.0)

        elif 'LSTM' in str(type(layer)):
            for m in layer.modules():
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)



class CustomEarlyStopping():
    def __init__(self, patience, save_path, save_name):
        self.cnt = 0
        self.best_loss = 10000000
        self.best_epoch = 0
        self.stop = False
        self.patience = patience
        self.save_path = save_path+save_name
        self.best_model_path = None


    def __call__(self, model, epoch, valid_loss):
        if valid_loss <= self.best_loss:
            try: os.remove(self.best_model_path)
            except: pass
        
            self.cnt        = 0
            self.best_loss  = valid_loss
            self.best_epoch = epoch

            torch.save(model, self.save_path+f'_{epoch+1}.pt')
            self.best_model_path = self.save_path+f'_{epoch+1}.pt'
            self.stop = False

        else:
            self.cnt += 1
            if self.cnt == self.patience:
                self.stop = True
        return self.stop, self.best_model_path