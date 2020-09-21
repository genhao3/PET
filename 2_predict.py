# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

import time
from PIL import Image

import torch

import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定第一块gpu
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.transforms as transforms
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
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
        model = EfficientNet.from_pretrained('efficientnet-b8')
        in_channel = model._fc.in_features
        model._fc = nn.Linear(in_channel, 3)
        self.efficientnet = model

    def forward(self, img):
        out = self.efficientnet(img)
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
model_path = 'best_acc_dogenet_b8' + args.v + '.pth'  # 模型名称

test_loader = torch.utils.data.DataLoader(
    QRDataset(test_jpg,
              transforms.Compose([
                  # transforms.RandomCrop(128),
                  transforms.RandomRotation(degrees=args.RandomRotation, expand=True),  # 没旋转只有0.85,旋转有0.90
                  transforms.Resize((args.Resize, args.Resize)),
                  transforms.ColorJitter(brightness=args.ColorJitter, contrast=args.ColorJitter,
                                         saturation=args.ColorJitter),  # 加入1
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
model.load_state_dict(torch.load(args.save_dir + '/' + model_path))  # 模型文件路径，默认放在args.save_dir下
# model = nn.DataParallel(model).cuda()
if test_pred is None:
    test_pred = predict(test_loader, model, 5)
else:
    test_pred += predict(test_loader, model, 5)

test_csv = pd.DataFrame()
test_csv['uuid'] = list(range(1, 2001))
test_csv['label'] = np.argmax(test_pred, 1)
test_csv['label'] = test_csv['label'].map({1: 'AD', 0: 'CN', 2: 'MCI'})
test_csv.to_csv(args.save_dir + '/best_acc_dogenet_b8' + args.v + '.csv', index=False)
