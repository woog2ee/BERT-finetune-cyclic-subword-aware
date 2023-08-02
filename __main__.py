import torch
import torch.nn as nn
from utils import (str2bool,
                   load_json_data,
                   get_model_and_tokenizer,
                   get_tokenizer_pad_id,
                   get_optimizer_and_scheduler)
from dataset import CustomDataset
from torch.utils.data import DataLoader, RandomSampler
from trainer import CustomTrainer
from model import CustomClassifier, CustomEarlyStopping
import os
import json
import math
import time
import random
import argparse
import numpy as np



if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data_path', type=str, default='/HDD/seunguk/KGEC/')
    parser.add_argument('--data_name', type=str, default='total_resorted_29_')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--save_name', type=str)

    parser.add_argument('--curriculum', type=str2bool)
    parser.add_argument('--subword_aware', type=str2bool,
                        help='Train with additional LSTM layers above the BERT or not')
    parser.add_argument('--output_device', type=list)

    parser.add_argument('--max_tokens_per_word', type=int, default=6)
    parser.add_argument('--max_words_per_sent', type=int, default=16)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=4000)

    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()


    print(f'========== Using Device with {args.device}')
    print(f'========== Setting Seeds with {args.seed}\n')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    print(f'========== Loading Model & Tokenizer with {args.model_path}\n')
    num_labels = 2
    model, tokenizer = get_model_and_tokenizer(args.model_path, num_labels)
    pad_id = get_tokenizer_pad_id(tokenizer)
    classifier = CustomClassifier(args.device, model, args.model_path, args.hidden_size, num_labels,
                                  args.subword_aware, pad_id, args.max_tokens_per_word, args.max_words_per_sent)
    early_stopping = CustomEarlyStopping(args.patience, args.save_path, args.save_name)


    print(f'========== Loading Dataset & DataLoader\n')
    train_data = load_json_data(args.data_path+args.data_name+'train.json')
    valid_data = load_json_data(args.data_path+args.data_name+'valid.json')
    test_data = load_json_data(args.data_path+args.data_name+'test.json')

    train_dataset = CustomDataset(tokenizer, args.max_length,
                                  args.subword_aware, args.max_words_per_sent, train_data)
    valid_dataset = CustomDataset(tokenizer, args.max_length,
                                  args.subword_aware, args.max_words_per_sent, valid_data)
    test_dataset = CustomDataset(tokenizer, args.max_length,
                                 args.subword_aware, args.max_words_per_sent, test_data)

    if args.curriculum:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=RandomSampler(train_dataset),
                                  num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  sampler=RandomSampler(valid_dataset),
                                  num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 sampler=RandomSampler(test_dataset),
                                 num_workers=args.num_workers)


    print('========== Setting Optimizer & Scheduler\n')
    t_total = len(train_loader) * args.epochs
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.lr, args.warmup_steps, t_total)


    print('========== Training & Testing Start\n')
    trainer = CustomTrainer(classifier, args.device, args.output_device, optimizer, scheduler)

    best_model_path, all_train_acc, all_valid_acc, all_train_hist, all_valid_hist = trainer.iteration(args.epochs, train_loader, valid_loader, early_stopping)
    test_acc = trainer.predict(test_loader, best_model_path)    


    f = open(args.save_path+args.save_name+'.txt', 'w')
    f.write('========== Training Accuracy: ')
    f.write(str(all_train_acc))
    f.write('\n\n')

    f.write('========== Valid Accuracy: ')
    f.write(str(all_valid_acc))
    f.write('\n\n')

    f.write('========== Training History: ')
    f.write(str(all_train_hist))
    f.write('\n\n')

    f.write('========== Valid History: ')
    f.write(str(all_valid_hist))
    f.write('\n\n')

    f.write(f'========== Test Accuracy: {test_acc}')
    f.close()

    print(f'========== Test Accuracy: {test_acc}')