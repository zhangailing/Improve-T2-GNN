import numpy as np
import torch
import time
import os
from train import Train
from args import args
from utils import setup_seed

if __name__ == '__main__':
    if not os.path.exists('.checkpoints'):
        os.makedirs('.checkpoints')
    setup_seed(args.seed, torch.cuda.is_available())

    acc_fea = []
    acc_str = []
    acc_stu = []
    repeats = args.repeat

    for repeat in range(repeats):
        print('******************** Repeat {} Done ********************\n'.format(repeat + 1))

        train = Train(args, acc_fea, acc_str, acc_stu)
        t_total = time.time()

        # Track actual epochs for each model
        actual_epochs_fea = 0
        actual_epochs_str = 0
        actual_epochs_stu = 0

        # Pre-train Feature teacher model
        early_stop_fea = False
        for epoch in range(args.epoch_fea):
            actual_epochs_fea += 1
            if train.pre_train_teacher_fea(epoch):
                early_stop_fea = True
                break
        if early_stop_fea:
            train.save_checkpoint(ts='teacher_fea')

        # Pre-train Structure teacher model
        early_stop_str = False
        for epoch in range(args.epoch_str):
            actual_epochs_str += 1
            if train.pre_train_teacher_str(epoch):
                early_stop_str = True
                break
        if early_stop_str:
            train.save_checkpoint(ts='teacher_str')

        # Train student model GCN
        early_stop_stu = False
        for epoch in range(args.epoch_stu):
            actual_epochs_stu += 1
            if train.train_student(epoch):
                early_stop_stu = True
                break
        if early_stop_stu:
            train.save_checkpoint(ts='student')

        # Load best pre-trained teacher models
        if early_stop_fea:
            train.load_checkpoint(ts='teacher_fea')
        if early_stop_str:
            train.load_checkpoint(ts='teacher_str')
        # Load best student model
        if early_stop_stu:
            train.load_checkpoint(ts='student')

        print('\n')

        # Test models
        train.test('teacher_fea')
        train.test('teacher_str')
        train.test('student')

        print('******************** Repeat {} Done ********************\n'.format(repeat + 1))

    print('{} repeats Run on {} dataset'.format(repeats,args.dataset))
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    print('{} epoch Feature Teacher train, Test Acc Avg: {:.6f}'.format(actual_epochs_fea,sum(acc_fea) / repeats))
    print('{} epoch Structure Teacher train, Test Acc Avg: {:.6f}'.format(actual_epochs_str,sum(acc_str) / repeats))
    print('{} epoch Student train, Test Acc Avg: {:.6f}'.format(actual_epochs_stu,sum(acc_stu) / repeats))
