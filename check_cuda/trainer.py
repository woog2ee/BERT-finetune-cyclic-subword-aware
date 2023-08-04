import torch
import torch.nn as nn
from utils import (calculate_accuracy, compact_name,
                  check_tensors_on_cpu)
import os
from tqdm import tqdm



def iteration(device, model, output_device,
              criterion, optimizer, scheduler,
              epochs, train_loader, valid_loader, save_path, printt=False):
    #model = model.to(device)
    model = nn.DataParallel(model, output_device=output_device)
    nextiscuda = next(model.parameters()).is_cuda
    print(f'### nextiscuda {nextiscuda}')
    #check_tensors_on_cpu(model)
    
    train_epoch_acc_lst, valid_epoch_acc_lst = [], []
    train_epoch_loss_lst, valid_epoch_loss_lst = [], []
    train_batch_acc_lst, valid_batch_acc_lst = [], []
    train_batch_loss_lst, valid_batch_loss_lst = [], []
    #printt=True
    for epoch in range(epochs):

        train_epoch_acc, valid_epoch_acc = 0.0, 0.0
        train_epoch_loss, valid_epoch_loss = 0.0, 0.0
        
        train_batch_acc, valid_batch_acc = [], []
        train_batch_loss, valid_batch_loss = [], []

        train_acc, valid_acc = 0.0, 0.0
        train_loss, valid_loss = 0.0, 0.0

        # Train
        train_iter = tqdm(enumerate(train_loader),
                          desc='EP_%s:%d' % ('train', epoch+1),
                          total=len(train_loader),
                          bar_format='{l_bar}{r_bar}')
        model.train()
        for idx, batch in train_iter:
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            token_ids, label = batch['input'], batch['label']
            tokens_per_word = batch['tokens_per_word']          
            if printt:
                print(f'#### token_ids : {token_ids.shape} {token_ids.is_cuda}')
                print(f'#### label : {label.shape} {label.is_cuda}')
                print(f'#### tokens_per_word : {tokens_per_word.shape} {tokens_per_word.is_cuda}')
            out = model(token_ids, tokens_per_word)
            if printt: print(f'#### out : {out.shape} {out.is_cuda}')
            #out = out.squeeze(-1)  # BCE
          
            out, label = out.view(-1, 2), label.view(-1)  # CE 
            if printt:
                print(f'#### out view : {out.shape} {out.is_cuda}')
                print(f'#### label view : {label.shape} {label.is_cuda}')
            #print(f'### {out.is_cuda} {label.is_cuda}')
            loss = criterion(out, label)
            #train_loss += loss.item()
            #loss.requires_grad_(True)
            loss.backward()
            train_loss += loss.item()
    
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            
            #out = torch.round(out)  # BCE
            out = torch.argmax(out, dim=1)  # CE
            acc = calculate_accuracy(out, label)
            train_acc += acc

            post_fix = {'epoch': epoch+1,
                        'iter': idx+1, 
                        'train_loss': train_loss / (idx+1), 
                        'train_acc': train_acc / (idx+1)}
            if (idx+1) % 200 == 0:
                train_iter.write(str(post_fix))
                train_batch_acc.append(train_acc / (idx+1))
                train_batch_loss.append(train_loss / (idx+1))

        train_epoch_acc = train_acc / (idx+1)
        train_epoch_loss = train_loss / (idx+1)
        print(f'===== Epoch {epoch+1} Train Accuracy: {train_epoch_acc}')


        # Valid
        valid_iter = tqdm(enumerate(valid_loader),
                          desc='EP_%s:%d' % ('valid', epoch+1),
                          total=len(valid_loader),
                          bar_format='{l_bar}{r_bar}')
        model.eval()
        for idx, batch in valid_iter:
            batch = {k: v.to(device) for k, v in batch.items()}
            token_ids, label = batch['input'], batch['label']
            tokens_per_word = batch['tokens_per_word']

            out = model(token_ids, tokens_per_word)
            
            out, label = out.view(-1, 2), label.view(-1)
            loss = criterion(out, label)
            valid_loss += loss.item()

            out = torch.argmax(out, dim=1)
            acc = calculate_accuracy(out, label)
            valid_acc += acc

            post_fix = {'epoch': epoch+1,
                        'iter': idx+1, 
                        'valid_loss': valid_loss / (idx+1), 
                        'valid_acc': valid_acc / (idx+1)}
            if (idx+1) % 200 == 0:
                valid_iter.write(str(post_fix))
                valid_batch_acc.append(valid_acc / (idx+1))
                valid_batch_loss.append(valid_loss / (idx+1))

        valid_epoch_acc = valid_acc / (idx+1)
        valid_epoch_loss = valid_loss / (idx+1)
        print(f'===== Epoch {epoch+1} Valid Accuracy: {valid_epoch_acc}')

        train_epoch_acc_lst.append(train_epoch_acc)
        train_epoch_loss_lst.append(train_epoch_loss)
        valid_epoch_acc_lst.append(valid_epoch_acc)
        valid_epoch_loss_lst.append(valid_epoch_loss)

        train_batch_acc_lst.append(train_batch_acc)
        train_batch_loss_lst.append(train_batch_loss)
        valid_batch_acc_lst.append(valid_batch_acc)
        valid_batch_loss_lst.append(valid_batch_loss)
    
        save_model(model, epoch, save_path)
        # stop, best_model_path = early_stopping(self.model, epoch, valid_loss)
        # if stop:
        #     print(f'\n======== EarlyStopping at Epoch {epoch+1}\n')
        #     print(best_model_path)
        #     break
    
    return train_epoch_acc_lst, train_epoch_loss_lst, valid_epoch_acc_lst, valid_epoch_loss_lst,\
        train_batch_acc_lst, train_batch_loss_lst, valid_batch_acc_lst, valid_batch_loss_lst


def save_model(model, epoch, save_path):
    # try: os.remove(self.best_model_path)
    # except: pass
    torch.save(model, save_path+f'_{epoch+1}.pt')
    print(f'===== Epoch {epoch+1} Model Saved at {save_path}_{epoch+1}.pt\n')


def predict(device, epoch, test_loader, save_path):
    # Test
    test_acc = 0.0
    test_iter = tqdm(enumerate(test_loader),
                     desc='EP_%s' % ('test'),
                     total=len(test_loader),
                     bar_format='{l_bar}{r_bar}')

    model = torch.load(save_path+f'_{epoch+1}.pt')
    print(f'\n========== For Testing, {save_path}_{epoch+1}.pt Loaded')
    model.eval()
    for idx, batch in test_iter:
        batch = {k: v.to(device) for k, v in batch.items()}
        token_ids, label = batch['input'], batch['label']
        tokens_per_word = batch['tokens_per_word']

        out = model(token_ids, tokens_per_word)

        out, label = out.view(-1, 2), label.view(-1)
            
        out = torch.argmax(out, dim=1)
        acc = calculate_accuracy(out, label)
        test_acc += acc
    test_acc = test_acc / (idx+1)
    return test_acc