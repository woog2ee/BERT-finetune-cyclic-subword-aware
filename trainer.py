import torch
import torch.nn as nn
from utils import (get_train_valid_test_indices,
                   get_train_valid_test_data,
                   calculate_accuracy, compact_name)
from dataset import CustomDataset
from torch.utils.data import DataLoader
import os
import json
import time
from tqdm import tqdm



class CustomTrainer:
    def __init__(self, model, device, output_device,
                 cyclic_learning, valid_idx,
                 data_path, tokenizer, max_length, subword_aware,
                 batch_size, num_workers, data_length,
                 optimizer, scheduler):
        self.model = model.to(device)
        self.model = nn.DataParallel(self.model, output_device=output_device)
        self.device = device

        # For cyclic learning
        self.cyclic_learning = cyclic_learning
        self.valid_idx = valid_idx

        # To make train & valid & test dataset
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.subword_aware = subword_aware
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.json_indices = [i for i in range(9, 39, 1)]
        self.train_indices, self.valid_indices, self.test_indices = get_train_valid_test_indices(data_length)
        self.test_loader = None

        # To finetune
        self.criterion = nn.BCELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_model_path = None



    def get_train_valid_test_loader(self, epoch, cyclic=True):
        # Set which json file to train depending on cyclic learning
        if cyclic:
            valid_idx = self.json_indices[epoch]
        else:
            valid_idx = self.valid_idx


        # Load json file
        s_time = time.time()
        with open(self.data_path+f'total_cut_{valid_idx}.json', 'r') as f:
            data = json.load(f)
        
        if cyclic:
            print(f'===== Epoch {epoch+1} File "total_cut_{valid_idx}" Loaded {time.time() - s_time}s')
        else:
            print(f'========== File "total_cut_{valid_idx}" Loaded {time.time() - s_time}s')


        # Make CustomDataset
        train_data, valid_data, test_data = get_train_valid_test_data(data,
                                        self.train_indices, self.valid_indices, self.test_indices)
        train_dataset = CustomDataset(self.tokenizer, self.max_length, self.subword_aware, train_data)
        valid_dataset = CustomDataset(self.tokenizer, self.max_length, self.subword_aware, valid_data)
        test_dataset  = CustomDataset(self.tokenizer, self.max_length, self.subword_aware, test_data)
        
        if cyclic:
            print(f'===== Epoch {epoch+1} Dataset Created')
        else:
            print(f'========== Dataset Created')


        # Make DataLoader
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers)
        if cyclic:
            print(f'===== Epoch {epoch+1} DataLoader Created\n')
        else:
            print('========== DataLoader Created\n')
        
        return train_loader, valid_loader



    def iteration(self, epochs, model_path, save_name, save_path):
        all_train_acc, all_valid_acc = [], []

        # Train with only one type of DataLoader when doing no cyclic learning
        if not self.cyclic_learning:
            train_loader, valid_loader = self.get_train_valid_test_loader(None, False)


        best_valid_acc = 0.0
        for epoch in range(epochs):

            # Train with various type of DataLoader when doing cyclic learning
            if self.cyclic_learning:
                train_loader, valid_loader, test_loader = self.get_train_valid_test_loader(epoch)

            train_loss, valid_loss = 0.0, 0.0
            train_acc, valid_acc = 0.0, 0.0


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
                label = torch.tensor(label, dtype=torch.float16)
         
                out = self.model(token_ids)
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
                if (idx+1) % 500 == 0: train_iter.write(str(post_fix))
            train_acc = train_acc / (idx+1)
            print(f'===== Epoch {epoch+1} Train Accuracy: {train_acc}')


            # Valid
            valid_iter = tqdm(enumerate(valid_loader),
                             desc='EP_%s:%d' % ('test', epoch+1),
                             total=len(valid_loader),
                             bar_format='{l_bar}{r_bar}')
            self.model.eval()
            for idx, batch in valid_iter:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                token_ids, label = batch['input'], batch['label']
                label = torch.tensor(label, dtype=torch.float16)

                out = self.model(token_ids)
                out = torch.sigmoid(out)
                out = torch.tensor(torch.argmax(out, dim=2), dtype=torch.float16)

                acc = calculate_accuracy(out, label)
                valid_acc += acc
            valid_acc = valid_acc / (idx+1)
            print(f'===== Epoch {epoch+1} Valid Accuracy: {valid_acc}')

            if best_valid_acc <= valid_acc:
                self.best_model_path = self.save_model(epoch, model_path, save_name, save_path)
                best_valid_acc = valid_acc
            
            all_train_acc.append(train_acc)
            all_valid_acc.append(valid_acc)
        
        return all_train_acc, all_valid_acc



    def predict(self):
        # Test
        test_acc = 0.0

        test_loader = self.test_loader
        test_iter = tqdm(enumerate(test_loader),
                         desc='EP_%s' % ('test'),
                         total=len(test_loader),
                         bar_format='{l_bar}{r_bar}')

        best_model = torch.load(self.best_model_path)
        best_model.eval()
        for idx, batch in test_iter:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            token_ids, label = batch['input'], batch['label']
            label = torch.tensor(label, dtype=torch.float16)

            out = self.model(token_ids)
            out = torch.sigmoid(out)
            out = torch.tensor(torch.argmax(out, dim=2), dtype=torch.float16)

            acc = calculate_accuracy(out, label)
            test_acc += acc
        test_acc = test_acc / (idx+1)
        return test_acc



    def save_model(self, epoch, model_path, save_name, save_path):
        try:
            os.remove(self.best_model_path)
        except: pass

        compact_model_path = compact_name(model_path)
        save_path += f'{save_name}_{epoch+1}.pt'
        torch.save(self.model, save_path)
        print(f'===== Epoch {epoch+1} Model Saved at {save_path}\n')
        return save_path