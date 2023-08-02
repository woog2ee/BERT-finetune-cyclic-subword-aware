import torch
import torch.nn as nn
from utils import (calculate_accuracy, compact_name)
import os
from tqdm import tqdm



class CustomTrainer:
    def __init__(self, model, device, output_device, optimizer, scheduler):
        self.model = model.to(device)
        self.model = nn.DataParallel(self.model, output_device=output_device)
        self.device = device

        self.criterion = nn.BCELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler


    def iteration(self, epochs, train_loader, valid_loader, early_stopping):
        all_train_acc, all_valid_acc = [], []
        all_train_hist, all_valid_hist = [], []

        for epoch in range(epochs):

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0.0, 0.0
            train_hist, valid_hist = [], []

            # Train
            train_iter = tqdm(enumerate(train_loader),
                              desc='EP_%s:%d' % ('train', epoch+1),
                              total=len(train_loader),
                              bar_format='{l_bar}{r_bar}')
            self.model.train()
            for idx, batch in train_iter:
                self.optimizer.zero_grad()

                batch = {k: v.to(self.device) for k, v in batch.items()}
                token_ids, label = batch['input'], batch['label']
                tokens_per_word = batch['tokens_per_word']
                label = torch.tensor(label, dtype=torch.float16)
         
                out = self.model(token_ids, tokens_per_word)
                out = torch.sigmoid(out)
                out = torch.tensor(torch.argmax(out, dim=2), dtype=torch.float16)
        
                loss = self.criterion(out, label)
                loss.requires_grad_(True)
                loss.backward()
                train_loss += loss.item()
           
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 5)
                self.optimizer.step()
                self.scheduler.step()
               
                acc = calculate_accuracy(out, label)
                train_acc += acc

                post_fix = {'epoch': epoch+1,
                            'iter': idx+1, 
                            'train_loss': train_loss / (idx+1), 
                            'train_acc': train_acc / (idx+1)}
                if (idx+1) % 200 == 0:
                    train_iter.write(str(post_fix))
                    train_hist.append([train_loss / (idx+1), train_acc / (idx+1)])

            train_acc = train_acc / (idx+1)
            train_loss = train_loss / (idx+1)
            print(f'===== Epoch {epoch+1} Train Accuracy: {train_acc}')


            # Valid
            valid_iter = tqdm(enumerate(valid_loader),
                             desc='EP_%s:%d' % ('valid', epoch+1),
                             total=len(valid_loader),
                             bar_format='{l_bar}{r_bar}')
            self.model.eval()
            for idx, batch in valid_iter:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                token_ids, label = batch['input'], batch['label']
                tokens_per_word = batch['tokens_per_word']
                label = torch.tensor(label, dtype=torch.float16)

                out = self.model(token_ids, tokens_per_word)
                out = torch.sigmoid(out)
                out = torch.tensor(torch.argmax(out, dim=2), dtype=torch.float16)

                loss = self.criterion(out, label)
                valid_loss += loss.item()

                acc = calculate_accuracy(out, label)
                valid_acc += acc

                post_fix = {'epoch': epoch+1,
                            'iter': idx+1, 
                            'valid_loss': valid_loss / (idx+1), 
                            'valid_acc': valid_acc / (idx+1)}
                if (idx+1) % 200 == 0:
                    valid_iter.write(str(post_fix))
                    valid_hist.append([valid_loss / (idx+1), valid_acc / (idx+1)])

            valid_acc = valid_acc / (idx+1)
            valid_loss = valid_loss / (idx+1)
            print(f'===== Epoch {epoch+1} Valid Accuracy: {valid_acc}')

            all_train_acc.append(train_acc)
            all_valid_acc.append(valid_acc)
            all_train_hist.append(train_hist)
            all_valid_hist.append(valid_hist)
        
            stop, best_model_path = early_stopping(self.model, epoch, valid_loss)
            if stop:
                print(f'\n======== EarlyStopping at Epoch {epoch+1}\n')
                print(best_model_path)
                break
        
        return best_model_path, all_train_acc, all_valid_acc, all_train_hist, all_valid_hist



    def predict(self, test_loader, best_model_path):
        # Test
        test_acc = 0.0
        test_iter = tqdm(enumerate(test_loader),
                         desc='EP_%s' % ('test'),
                         total=len(test_loader),
                         bar_format='{l_bar}{r_bar}')

        best_model = torch.load(best_model_path)
        best_model.eval()
        for idx, batch in test_iter:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            token_ids, label = batch['input'], batch['label']
            tokens_per_word = batch['tokens_per_word']
            label = torch.tensor(label, dtype=torch.float16)

            out = self.model(token_ids, tokens_per_word)
            out = torch.sigmoid(out)
            out = torch.tensor(torch.argmax(out, dim=2), dtype=torch.float16)

            acc = calculate_accuracy(out, label)
            test_acc += acc
        test_acc = test_acc / (idx+1)
        return test_acc



    def save_model(self, epoch, save_name, save_path):
        try: os.remove(self.best_model_path)
        except: pass

        save_path += f'{save_name}_{epoch+1}.pt'
        torch.save(self.model, save_path)
        print(f'===== Epoch {epoch+1} Model Saved at {save_path}\n')
        return save_path