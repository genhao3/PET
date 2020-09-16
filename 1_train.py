# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import random

import numpy as np
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import time, datetime
import config
import cv2
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一块gpu
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms

import torch.nn as nn

import torch.optim as optim

from torch.utils.data.dataset import Dataset
import logging
import os


def write_log(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = log_dir + '/log_dogenet-b7_' + args.v + '.log'
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
    )
    return logging


class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        start_time = time.time()
        img = Image.open(self.train_jpg[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        label = 0
        if 'CN' in self.train_jpg[index]:
            label = 0
        elif 'AD' in self.train_jpg[index]:
            label = 1
        elif 'MCI' in self.train_jpg[index]:
            label = 2
        return img, torch.from_numpy(np.array(label))

    def __len__(self):
        return len(self.train_jpg)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        logging.info(('\t'.join(entries)))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class DogeNet(nn.Module):
    def __init__(self):
        super(DogeNet, self).__init__()

        # model = EfficientNet.from_pretrained('efficientnet-b5', weights_path='./model/efficientnet-b5-b6417697.pth')
        model = EfficientNet.from_pretrained('efficientnet-b7', weights_path='./model/efficientnet-b7-dcc49843.pth')
        in_channel = model._fc.in_features
        model._fc = nn.Linear(in_channel, 3)
        self.efficientnet = model

    def forward(self, img):
        out = self.efficientnet(img)
        return out


class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()

        model = models.resnet50(pretrained=False)
        pre = torch.load("./pretrain/resnet50-19c8e357.pth")
        model.load_state_dict(pre)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Linear(512 * 4, 3)
        self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            # loss = criterion(output, target)
            loss = CrossEntropyLoss_label_smooth(output, target, num_classes=3)  # 加2
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.info((' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                      .format(top1=top1, top5=top5)))
        return top1


def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_loader):
                input = input.cuda()
                target = target.cuda()

                # compute output
                output = model(input)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        # loss = criterion(output, target)
        loss = CrossEntropyLoss_label_smooth(output, target, num_classes=3)  # 加2
        # '''warm up module'''
        # if epoch<warm_epoch:
        #     warm_up=min(1.0,warm_up+0.9/warm_iteration)
        #     loss*=warm_up

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)


def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=10, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes),
                                 fill_value=epsilon / (num_classes - 1))
    targets = targets.data.cpu()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1),
                             value=1 - epsilon)
    # outputs = outputs.data.cpu()
    log_prob = nn.functional.log_softmax(outputs, dim=1).cpu()
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


if __name__ == '__main__':
    args = config.args
    logging = write_log(args.save_dir)
    print('K={}\tepochs={}\tbatch_size={}\tresume={}\tlr={}'.format(args.k,
                                                                    args.epochs, args.batch_size, args.resume, args.lr))
    logging.info('K={}\tepochs={}\tbatch_size={}\tresume={}\tlr={}'.format(args.k,
                                                                           args.epochs, args.batch_size, args.resume,
                                                                           args.lr))
    train_jpg = np.array(glob.glob(args.dataset_train_path))
    skf = KFold(n_splits=args.k, random_state=233, shuffle=True)
    best_acc = 0
    for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg, train_jpg)):
        train_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[train_idx],
                      transforms.Compose([
                          # transforms.RandomGrayscale(),
                          transforms.RandomRotation(degrees=args.RandomRotation, expand=True),

                          transforms.Resize((args.Resize, args.Resize)),
                          transforms.RandomAffine(10),
                          transforms.ColorJitter(brightness=args.ColorJitter, contrast=args.ColorJitter,
                                                 saturation=args.ColorJitter),  # 加入1
                          # transforms.ColorJitter(hue=.05, saturation=.05),
                          # transforms.RandomCrop((450, 450)),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),

                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            QRDataset(train_jpg[val_idx],
                      transforms.Compose([
                          # transforms.RandomCrop(128),
                          transforms.RandomRotation(degrees=args.RandomRotation, expand=True),

                          transforms.Resize((args.Resize, args.Resize)),
                          transforms.ColorJitter(brightness=args.ColorJitter, contrast=args.ColorJitter,
                                                 saturation=args.ColorJitter),  # 加入1
                          # transforms.Resize((124, 124)),
                          # transforms.RandomCrop((450, 450)),
                          # transforms.RandomCrop((88, 88)),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomVerticalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
                      ), batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True
        )

        lr = args.lr
        # lr = lr * 0.1
        use_gpu = torch.cuda.is_available()
        print('use_gpu', use_gpu)
        args.start_epoch = 0
        model = DogeNet().cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        if args.resume:  # 万一训练中断，可以恢复训练
            checkpoint_path = args.save_dir + '/checkpoint' + '/best_new_dogenet_b7_' + args.v + '.pth.tar'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                start_flod_idx = checkpoint['flod_idx']
                if start_flod_idx > flod_idx:
                    continue
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                print('加载epoch={}成功!,best_acc:{}'.format(checkpoint['epoch'], best_acc))
                logging.info('加载epoch={}成功!,best_acc:{}'.format(checkpoint['epoch'], best_acc))
            else:
                print('无保存模型，重新训练')
                logging.info('无保存模型，重新训练')
        if flod_idx > 0 and not args.resume:  # 从第二折起，迭代前面最好的模型继续训练
            model.load_state_dict(torch.load(args.save_dir + '/best_acc_dogenet_b7_' + args.v + '.pth'))
            print('加载最好的模型')
            logging.info('加载最好的模型')
        # elif flod_idx > 0:
        #     model.load_state_dict(torch.load(log_dir + '/best_acc_dogenet_b7v3.pt'))
        # model.train(mode=True)
        # model = nn.DataParallel(model).cuda()

        for epoch in range(args.start_epoch, args.epochs):
            print('K/Epoch[{}/{} {}/{}]:'.format(flod_idx, args.k, epoch, args.epochs))
            logging.info('K/Epoch[{}/{} {}/{}]:'.format(flod_idx, args.k, epoch, args.epochs))
            scheduler.step()
            train(train_loader, model, criterion, optimizer, epoch)
            val_acc = validate(val_loader, model, criterion)
            args.resume = False
            if val_acc.avg.item() >= best_acc:
                best_acc = val_acc.avg.item()
                torch.save(model.state_dict(), args.save_dir + '/best_acc_dogenet_b7_' + args.v + '.pth')
            print('best_acc is :{}, lr:{}'.format(best_acc, scheduler.get_lr()))
            logging.info('best_acc is :{}, lr:{}'.format(best_acc, scheduler.get_lr()))
            # 保存最新模型
            checkpoint_path = args.save_dir + '/checkpoint'
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                     'best_acc': best_acc, 'flod_idx': flod_idx, 'scheduler': scheduler.state_dict()}
            torch.save(state, checkpoint_path + '/best_new_dogenet_b7_' + args.v + '.pth.tar')
        # break
