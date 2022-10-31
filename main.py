import torch
import torch.optim as optim
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from timm.models import create_model

import numpy as np
import os
import argparse

import utils
import pruner
import models

parser = argparse.ArgumentParser()

# model config
parser.add_argument('--model')
parser.add_argument('--pretrain', action='store_true')

# training config
parser.add_argument('--dataset', choices=['cifar10', 'cifar100'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=150)

# pruning config
parser.add_argument('--pruning', action='store_true')
parser.add_argument('--sparsity', type=float)
parser.add_argument('--method', choices=['random', 'magnitude', 'snip', 'grasp', 'snip_magnitude'])
parser.add_argument('--alpha', type=float)

# else
parser.add_argument('--save_dir', default='./outputs')
parser.add_argument('--seed', type=int, default=428)

args = parser.parse_args()

def main():
    device = torch.cuda.device_count()
    print('num gpus:', device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    utils.mixup_args['num_classes'] = num_classes
    train_loader, valid_loader = utils.get_dataloader(args.dataset, args.batch_size, data_download=True)

    model = create_model(args.model, pretrained=args.pretrain)
    model.head = nn.Linear(model.head.in_features, num_classes)
    if args.pretrain:
        init_data = model.state_dict()
    
    # pruning
    if args.pruning:
        print('pruning method:', args.method)
        print('sparsity:', args.sparsity)

        if 'mixer' in args.model:
            rm_modules = models.mixer_rm_modules(model)
        elif 'vit' in args.model:
            rm_modules = models.vit_rm_modules(model)
        elif 'pool' in args.model:
            rm_modules = models.pool_rm_modules(model)

        if args.method == 'random':
            pruner.SCORE = pruner.random(rm_modules)
        elif args.method == 'magnitude':
            pruner.SCORE = pruner.magnitude(rm_modules)
        elif args.method == 'snip':
            pruner.SCORE = pruner.snip(model, rm_modules, train_loader, device)
        elif args.method == 'grasp':
            pruner.SCORE = pruner.grasp(model, rm_modules, train_loader, device)
        elif args.method == 'snip_magnitude':
            pruner.SCORE = pruner.snip_magnitude(model, rm_modules, train_loader, device, args.alpha)
        
        model = create_model(args.model, pretrained=args.pretrain)
        model.head = nn.Linear(model.head.in_features, num_classes)
        if args.pretrain:
           model.load_state_dict(init_data)

        if 'mixer' in args.model:
            rm_modules = models.mixer_rm_modules(model)
        elif 'vit' in args.model:
            rm_modules = models.vit_rm_modules(model)
        elif 'pool' in args.model:
            rm_modules = models.pool_rm_modules(model)

        pruner.prune.global_unstructured(
            rm_modules,
            pruning_method=pruner.Pruner,
            amount = args.sparsity
        )

    # saving config
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_path = args.save_dir + r'/' + args.model
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if args.pruning:
        save_path = save_path + r'/' + args.method
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    if args.method == 'snip_magnitude':
        file_name = 'alpha{}_sparsity{}.chkpt'.format(args.alpha, int(args.sparsity * 100))
    else:
        if not args.pruning:
            file_name = 'non_pruning.chkpt'
        else:
            file_name = 'sparsity{}.chkpt'.format(int(args.sparsity * 100))
            if not args.pretrain:
                file_name = 'fs' + file_name
        


    history = {
        'train_loss': [],
        'valid_top1_acc': [],
        'valid_top5_acc': [],
        'valid_loss': [],
        'best_epoch': 0,
        'best_model': None
    }

    optimizer = optim.Adam
    loss_func = nn.CrossEntropyLoss()
    scheduler = CosineLRScheduler(
        optim.Adam(nn.Linear(1, 1).parameters()), t_initial=args.epochs, 
        lr_min=1e-4, 
        warmup_t=10, 
        warmup_lr_init=5e-5, 
        warmup_prefix=True
        )

    # training
    best_loss = float('inf')


    for epoch in range(args.epochs):
        lr = scheduler.get_epoch_values(epoch)[0]
    
        train_loss = utils.train_multi_gpu(
            model, train_loader, loss_func, optimizer, device, epoch+1, lr=lr, mixup=True
        )
        valid_loss, valid_top1_acc, valid_top5_acc = utils.validation_multi_gpu(
            model, valid_loader, loss_func, device, epoch+1
        )
        model.zero_grad()
    
        history['train_loss'].append(train_loss)   
        history['valid_top1_acc'].append(valid_top1_acc)
        history['valid_top5_acc'].append(valid_top5_acc)
        history['valid_loss'].append(valid_loss)
    
        print('top1-acc:', valid_top1_acc)
        if best_loss > valid_loss:
            best_loss = valid_loss
            history['best_epoch'] = epoch
            history['best_model'] = model.state_dict()
            print('updated best loss!')
    
        print('best acc:', max(history['valid_top1_acc']))
        
        torch.save(history, os.path.join(save_path, file_name))

if __name__ == '__main__':
    main()