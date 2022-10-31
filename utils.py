import torch
import torch.nn as nn

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from timm.data.mixup import Mixup

from tqdm import tqdm
import os

def get_dataloader(name, batch_size, data_download=True):
    path = './datasets/'
    if not os.path.exists(path):
        os.mkdir(path)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])
    
    if name == 'cifar10':
        train_data = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
        valid_data = datasets.CIFAR10(root=path, train=False, download=True, transform=valid_transform)
    elif name == 'cifar100':
        train_data = datasets.CIFAR100(root=path, train=True, download=True, transform=train_transform)
        valid_data = datasets.CIFAR100(root=path, train=False, download=True, transform=valid_transform)
    
    return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4), torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=4)

mixup_args = {
    'mixup_alpha': 0.,
    'cutmix_alpha': 1.0,
    'cutmix_minmax': None,
    'prob': 1.0,
    'switch_prob': 0.,
    'mode': 'batch',
    'label_smoothing': 0.1,
    'num_classes': 10
    }

def train(model, train_loader, loss_func, optimizer, device, epoch, lr=0.001, mixup=False):
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)
    if mixup:
        mixup_fn = Mixup(**mixup_args)
    
    losses = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    
    model.train()
    for inputs, targets in tqdm(train_loader, desc='TRAIN: {}'.format(epoch)):
        if mixup:
            inputs, targets = mixup_fn(inputs, targets)
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
        
    return losses / train_loader.dataset.__len__()

def validation(model, valid_loader, loss_func, device, epoch):
    model = model.to(device)
    
    losses = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader, desc='VALID: {}'.format(epoch)):
            inputs, targets = inputs.to(device), targets.to(device)
        
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
        
            losses += loss.item()
            _, pred = outputs.topk(5, dim=1)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            top1_acc += correct[:, :1].sum().item()
            top5_acc += correct[:, :5].sum().item()
        
    model.zero_grad()
    return losses / valid_loader.dataset.__len__(), top1_acc / valid_loader.dataset.__len__(), top5_acc / valid_loader.dataset.__len__()


def train_multi_gpu(model, train_loader, loss_func, optimizer, device, epoch, lr=0.001, mixup=False):
    device = [i for i in range(device)]
    model = nn.DataParallel(model, device_ids=device).to('cuda:0')

    optimizer = optimizer(model.module.parameters(), lr=lr)
    if mixup:
        mixup_fn = Mixup(**mixup_args)
    
    losses = 0.0
    
    model.train()
    for inputs, targets in tqdm(train_loader, desc='TRAIN: {}'.format(epoch)):
        if mixup:
            inputs, targets = mixup_fn(inputs, targets)
        inputs, targets = inputs.cuda(0), targets.cuda(0)
        
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
        
    return losses / train_loader.dataset.__len__()

def validation_multi_gpu(model, valid_loader, loss_func, device, epoch):
    device = [i for i in range(device)]
    model = nn.DataParallel(model, device_ids=device).to('cuda:0')
    
    losses = 0.0
    top1_acc = 0.0
    top5_acc = 0.0
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader, desc='VALID: {}'.format(epoch)):
            inputs, targets = inputs.cuda(0), targets.cuda(0)
        
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
        
            losses += loss.item()
            _, pred = outputs.topk(5, dim=1)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            top1_acc += correct[:, :1].sum().item()
            top5_acc += correct[:, :5].sum().item()
        
    model.zero_grad()
    return losses / valid_loader.dataset.__len__(), top1_acc / valid_loader.dataset.__len__(), top5_acc / valid_loader.dataset.__len__()