import torch
import torch.nn as nn
import os
from utils import (slice_3d_tensors, compact_name, check_tensors_on_cpu)



class CustomClassifier(nn.Module):
    def __init__(self, device, model, model_path, hidden_size, num_labels,
                 subword_aware, pad_id, max_tokens_per_word, max_words_per_sent, dropout=0.3, printt=False):
        super().__init__()
        if 'bert' in model_path.lower():
            self.model_type = 'bert'
        elif 'electra' in model_path.lower():
            self.model_type = 'electra'

        self.device = device
        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels).to(device)

        self.subword_aware = subword_aware


        self.token_pad_emb = self.get_trained_token_embedding(model, pad_id).to(device)
        self.word_pad_emb = self.token_pad_emb.repeat(max_tokens_per_word, 1).unsqueeze(0).to(device)
        # <changed>
        if hasattr(self, 'token_pad_emb'): setattr(self, 'token_pad_emb', self.token_pad_emb)
        else: self.register_buffer('token_pad_emb', self.token_pad_emb)
        if hasattr(self, 'word_pad_emb'): setattr(self, 'word_pad_emb', self.word_pad_emb)
        else: self.register_buffer('word_pad_emb', self.word_pad_emb)
        # self.register_buffer('token_pad_emb', self.token_pad_emb)
        # self.register_buffer('word_pad_emb', self.word_pad_emb)
        
        #printt=True
        if printt:
            print(f'#### token_pad_emb: {self.token_pad_emb.shape} {self.token_pad_emb.is_cuda}')
            print(f'#### word_pad_emb: {self.word_pad_emb.shape} {self.word_pad_emb.is_cuda}')

        self.max_tokens_per_word = max_tokens_per_word
        self.max_words_per_sent = max_words_per_sent

        if self.subword_aware: # <changed>
            self.embedder = SubwordAwareEmbedder(hidden_size,
                                                 max_tokens_per_word, max_words_per_sent).to(device)
            #printt=True
            if printt:
                print('check tensors on: self.embedder')
                check_tensors_on_cpu(self.embedder)
        # token_pad_emb: [1, hidden_size]
        # word_pad_emb = [1, max_tokens_per_word, hidden_size]


    def forward(self, token_ids, tokens_per_word, printt=False):
        # token_ids: [batch_size, max_length]
        # tokens_per_word: [batch_size, words_per_sent]
        #printt=True
        if self.model_type == 'electra':
            out = self.model(token_ids).last_hidden_state
            if printt: print(f'#### out: {out.shape} {out.is_cuda} {type(out)} {out.dtype}')
            # last_hidden_state: [batch_size, max_length, hidden_size]
        
        if self.subword_aware:
            sliced = slice_3d_tensors(out, tokens_per_word,
                                      self.token_pad_emb, self.word_pad_emb,
                                      self.max_tokens_per_word, self.max_words_per_sent)
            if printt: print(f'#### sliced {sliced.shape} {sliced.is_cuda} {type(sliced)} {sliced.dtype}')
            # sliced: [batch_size, max_words_per_sent, max_tokens_per_word, hidden_size]
            out = self.embedder(sliced)
            if printt: print(f'### out {out.shape} {out.is_cuda} {type(out)} {out.dtype}')
            # last_hidden_state: [batch_size, max_words_per_sent, hidden_size]

        out = self.dropout(out)

        returned = torch.sigmoid(self.classifier(out))
        if printt: print(f'### returned {returned.shape} {returned.is_cuda} {type(returned)} {returned.dtype}')
        return returned
        #return torch.sigmoid(self.classifier(out))
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


    def forward(self, input, printt=False):
        #printt=True
        # input: [batch_size, max_words_per_sent, max_tokens_per_word, hidden_size]
        batch_size = input.size(0)

        reshaped = input.view(-1, self.max_tokens_per_word, self.hidden_size)
        if printt: print(f'#### reshaped : {reshaped.shape} {reshaped.is_cuda} {type(reshaped)} {reshaped.dtype}')
        # reshaped: [-1, max_tokens_per_word, hidden_size]

        out, h_n = self.model(reshaped)
        if printt: print(f'#### out : {out.shape} {out.is_cuda} {type(out)} {out.dtype}')

        forward_out, backward_out = torch.split(out, self.hidden_size, dim=2)
        if printt: print(f'#### forward_out: {forward_out.shape} {forward_out.is_cuda} {type(forward_out)} {forward_out.dtype}')
        if printt: print(f'#### backward_out: {backward_out.shape} {backward_out.is_cuda} {type(backward_out)} {backward_out.dtype}')
        # each_out: [-1, max_tokens_per_word, hidden_size]

        forward_last_hidden = forward_out[:, -1, :]
        backward_last_hidden = backward_out[:, 0, :]
        if printt: print(f'#### forward_last_hidden : {forward_last_hidden.shape} {forward_last_hidden.is_cuda} {type(forward_last_hidden)} {forward_last_hidden.dtype}')
        if printt: print(f'#### backward_last_hidden : {backward_last_hidden.shape} {backward_last_hidden.is_cuda} {type(backward_last_hidden)} {backward_last_hidden.dtype}')
        # each_last_hidden: [-1, hidden_size]

        concat = torch.cat((forward_last_hidden, backward_last_hidden), dim=1)
        if printt: print(f'#### concat : {concat.shape} {concat.is_cuda} {type(concat)} {concat.dtype}')
        combined = concat @ self.weights
        if printt: print(f'#### self.weights : {self.weights.shape} {self.weights.is_cuda} {type(self.weights)} {self.weights.dtype}')
        if printt: print(f'#### combined : {combined.shape} {combined.is_cuda} {type(combined)} {combined.dtype}')
        # combined: [-1, hidden_size]

        returned = combined.view(batch_size, self.max_words_per_sent, -1)
        if printt: print(f'#### returned: {returned.shape} {returned.is_cuda} {type(returned)} {returned.dtype}')
        return returned
        #return combined.view(batch_size, self.max_words_per_sent, -1)
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