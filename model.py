import torch
import torch.nn as nn



class CustomClassifier(nn.Module):
    def __init__(self, model, model_path, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        if 'bert' in model_path.lower():
            self.model_type = 'bert'
        elif 'electra' in model_path.lower():
            self.model_type = 'electra'

        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)


    def forward(self, token_ids):
        # token_ids: [batch_size, max_length]
      
        if self.model_type == 'electra':
            out = self.model(token_ids).last_hidden_state
            # last_hidden_state: [batch_size, max_length, hidden_size]

        out = self.dropout(out)
        return self.classifier(out)
        # classfied: [batch_size, max_length, num_classes]