# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import time, datetime
import pdb, traceback

import cv2
# import imagehash
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b4')

import torch

import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一块gpu
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset


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


class DogeNet(nn.Module):
    def __init__(self):
        super(DogeNet, self).__init__()

        # model = EfficientNet.from_pretrained('efficientnet-b5', weights_path='./model/efficientnet-b5-b6417697.pth')
        # model = EfficientNet.from_pretrained('efficientnet-b7', weights_path='./model/efficientnet-b7-dcc49843.pth')
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
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(2048, 3)
        self.resnet = model

    #         model = EfficientNet.from_pretrained('efficientnet-b4')
    #         model._fc = nn.Linear(1792, 2)
    #         self.resnet = model

    def forward(self, img):
        out = self.resnet(img)
        return out


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


args = config.args
test_jpg = [args.dataset_test_path + '/{0}.jpg'.format(x) for x in range(1, 2001)]
test_jpg = np.array(test_jpg)

test_pred = None
for model_path in ['best_acc_dogenet_b7' + args.v + '.pth', 'resnet18_fold0.pt', 'resnet18_fold1.pt', 'resnet18_fold2.pt',
                   'resnet18_fold3.pt', 'resnet18_fold4.pt', 'resnet18_fold5.pt',
                   'resnet18_fold6.pt', 'resnet18_fold7.pt', 'resnet18_fold8.pt',
                   'resnet18_fold9.pt']:
    if model_path != 'best_acc_dogenet_b7' + args.v + '.pth':
        continue
    test_loader = torch.utils.data.DataLoader(
        QRDataset(test_jpg,
                  transforms.Compose([
                      # transforms.RandomCrop(128),
                      transforms.RandomRotation(degrees=args.RandomRotation, expand=True),  # 没旋转只有0.85,旋转有0.90
                      transforms.Resize((args.Resize, args.Resize)),
                      transforms.ColorJitter(brightness=args.ColorJitter, contrast=args.ColorJitter, saturation=args.ColorJitter),  # 加入1
                      # transforms.CenterCrop((450, 450)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])
                  ), batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True
    )

    # model = VisitNet().cuda()
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    model = DogeNet().cuda()
    model.load_state_dict(torch.load(args.save_dir + '/' + model_path))
    # model = nn.DataParallel(model).cuda()
    if test_pred is None:
        test_pred = predict(test_loader, model, 5)
    else:
        test_pred += predict(test_loader, model, 5)
    break

test_csv = pd.DataFrame()
test_csv['uuid'] = list(range(1, 2001))
test_csv['label'] = np.argmax(test_pred, 1)
test_csv['label'] = test_csv['label'].map({1: 'AD', 0: 'CN', 2: 'MCI'})
test_csv.to_csv(args.save_dir + '/best_acc_dogenet_b7_' + args.v + '.csv', index=None)
