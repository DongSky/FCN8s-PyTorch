import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from dataset import *
from myutils import *
from FCN8s_pytorch import *

import argparse
import os
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--phase",type=str,default="train")
parser.add_argument("--param",type=str,default=None)
parser.add_argument("--data",type=str,default="./train")
parser.add_argument("--out",type=str,default="./sample")
opt=parser.parse_args()

iter_num=30
learning_rate=1e-4
color_trans=Colorize()
dataRoot = opt.data
if not os.path.exists(opt.out):
    os.mkdir(opt.out)
if opt.phase == 'train':
    checkRoot = opt.out
    loader = torch.utils.data.DataLoader(
        SBDClassSeg(dataRoot, split='train', transform=True),
        batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
else:
    outputRoot = opt.out
    loader = torch.utils.data.DataLoader(
        VOC2012ClassSeg(dataRoot, transform=True),
        batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

"""nets"""
model = FCN()
if opt.param is None:
    vgg16 = torchvision.models.vgg16(pretrained=True)
    model.init_vgg16(vgg16, copy_fc8=False, init_upscore=True)
else:
    model.load_state_dict(torch.load(opt.param))

criterion = CrossEntropyLoss2d()
optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.5, 0.999))

model = model.cuda()
cnt=0
if opt.phase=="train":
    for i in range(iter_num):
        for ib,data in enumerate(loader):
            inputs = Variable(data[0]).cuda()
            targets = Variable(data[1]).cuda()
            #print(targets)
            model.zero_grad()
            outputs = model(inputs)
            #print(outputs)
            loss = criterion(outputs, targets)
            #print(loss.data[0])
            loss.backward()
            optimizer.step()
            if ib%2==0:
                print("loss: %.4f (epoch: %5d, step: %5d)"%(loss.data[0],it,ib))
            cnt+=1
            if cnt%1000==0:
                image=color_transform(outputs[0].cpu().max(0)[1].data)
                imsave('%s/FCN-epoch-%d-step-%d.png'%(checkRoot, it, ib),image)
        filename=('%s/FCN-epoch-%d-step-%d.pth'%(checkRoot, it, ib))
        torch.save(model.state_dict(), filename)
        print('save: (epoch: %d, step: %d)' % (it, ib))
else:
    for ib, data in enumerate(loader):
        print('testing batch %d' % ib)
        inputs = Variable(data[0]).cuda()
        outputs = model(inputs)
        hhh = color_transform(outputs[0].cpu().max(0)[1].data)
        imsave(os.path.join(outputRoot, data[1][0] + '.png'), hhh)
