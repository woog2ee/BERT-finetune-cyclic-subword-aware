import torch
import torch.nn as nn
from utils import (str2bool,
                   get_model_and_tokenizer,
                   get_optimizer_and_scheduler)
from trainer import CustomTrainer
from model import CustomClassifier
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
    parser.add_argument('--data_length', type=int, default=6036434)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--save_path', type=str, default='/HDD/seunguk/ckp/')

    parser.add_argument('--cyclic_learning', type=str2bool,
                        help='Train with cyclic curriculum learning or not')
    parser.add_argument('--valid_idx', type=int,
                        help='Set which json file to train when doing no cyclic learning')

    parser.add_argument('--subword_aware', type=str2bool,
                        help='Train with additional LSTM layers above the BERT or not')
    parser.add_argument('--output_device', type=list)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=4000)

    print('========== Loading All Parse Arguments\n')
    args = parser.parse_args()


    print(f'========== With Cyclic Curriculum Learning: {args.cyclic_learning}')
    print(f'========== Using Device with {args.device}')
    print(f'========== Setting Seeds with {args.seed}\n')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    print(f'========== Loading Model & Tokenizer with {args.model_path}\n')
    model, tokenizer = get_model_and_tokenizer(args.model_path, 2)
    classifier = CustomClassifier(model, args.model_path, args.hidden_size, 2)


    print('========== Setting Optimizer & Scheduler')
    train_loader_length = math.ceil(args.data_length * 0.9 / args.batch_size)
    t_total = train_loader_length * args.epochs
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.lr,
                                                       args.warmup_steps, t_total)


    print('========== Training & Testing Start\n')
    trainer = CustomTrainer(classifier, args.device, args.output_device,
                            args.cyclic_learning, args.valid_idx,
                            args.data_path, tokenizer, args.max_length, args.subword_aware,
                            args.batch_size, args.num_workers, args.data_length,
                            optimizer, scheduler)

    all_train_acc, all_valid_acc = trainer.iteration(args.epochs, args.model_path, args.save_name, args.save_path)
    test_acc = trainer.predict()    

    print('========== Training Accuracy History')
    print(all_train_acc, '\n')

    print('========== Valid Accuracy History')
    print(all_valid_acc, '\n')

    print(f'========== Test Accuracy: {test_acc}')