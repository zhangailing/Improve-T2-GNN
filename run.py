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
    # 控制训练重复的轮数
    for repeat in range(repeats):
        print('-------------------- Repeat {} Start -------------------'.format(repeat+1))

        train = Train(args,acc_fea,acc_str,acc_stu)
        t_total = time.time()

        # pre-train Feature teacher model
        for epoch in range(args.epoch_fea):
            train.pre_train_teacher_fea(epoch)
        train.save_checkpoint(ts='teacher_fea')

        # pre-train Structure teacher model
        for epoch in range(args.epoch_str):
            train.pre_train_teacher_str(epoch)
        train.save_checkpoint(ts='teacher_str')

        # train student model GCN
        for epoch in range(args.epoch_stu):
            train.train_student(epoch)
        train.save_checkpoint(ts='student')

        # load best pre-train teahcer models
        train.load_checkpoint(ts='teacher_fea')
        train.load_checkpoint(ts='teacher_str')
        # test student model GCN
        train.load_checkpoint(ts='student')

        ## test models
        train.test('teacher_fea')
        train.test('teacher_str')
        train.test('student')

        print('******************** Repeat {} Done ********************\n'.format(repeat+1))

    print('Run on {} dataset'.format(args.dataset))
    print('Run on {} txt'.format(train.repeat))
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    print('Total repeats: {}'.format(repeats))
    print('{} epoch Fea Test Acc: {} Test Avg: {:.6f}'.format(args.epoch_fea,acc_fea,sum(acc_fea) / repeats))
    print('{} epoch Str Test Acc: {} Test Avg: {:.6f}'.format(args.epoch_str,acc_str,sum(acc_str) / repeats))
    print('{} epoch Stu Test Acc: {} Test Avg: {:.6f}'.format(args.epoch_stu,acc_stu,sum(acc_stu) / repeats))



