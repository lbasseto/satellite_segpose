import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import torch.nn.functional as F

from tensorboardX import SummaryWriter

import PnP

import numpy as np
import math
import random

class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.mp1 = nn.MaxPool1d(8)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        rawx = x

        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(batchsize, 2, -1)
        return rawx + x

class PnPDataset(Dataset):
    def __init__(self):
        self.Pi = np.load('Pi.npy')
        self.gPi = np.load('gPi.npy')
        self.gRt = np.load('gRt.npy')
        assert (len(self.Pi) == len(self.gPi))
        assert (len(self.Pi) == len(self.gRt))

    def __len__(self):
        return len(self.Pi)

    def __getitem__(self, idx):
        pi = self.Pi[idx, :].reshape((-1, 2)).transpose()
        gpi = self.gPi[idx, :].reshape((-1,2)).transpose()
        grt = self.gRt[idx, :]
        return pi, gpi, grt

class KernelMd(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(KernelMd, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, D_out)

        self.act1 = torch.nn.LeakyReLU(0.1, inplace=True)
        self.act2 = torch.nn.LeakyReLU(0.1, inplace=True)
        self.act3 = torch.nn.LeakyReLU(0.1, inplace=True)
        self.act4 = torch.nn.LeakyReLU(0.1, inplace=True)
        self.act5 = torch.nn.LeakyReLU(0.1, inplace=True)
        # self.act1 = torch.nn.ReLU()
        # self.act2 = torch.nn.ReLU()
        # self.act1 = torch.nn.Tanh()
        # self.act2 = torch.nn.Tanh()
        # self.act3 = torch.nn.Tanh()
        # self.act4 = torch.nn.Tanh()
        # self.act5 = torch.nn.Tanh()

        self.bn1 = torch.nn.BatchNorm1d(H)
        self.bn2 = torch.nn.BatchNorm1d(H)
        self.bn3 = torch.nn.BatchNorm1d(H)
        self.bn4 = torch.nn.BatchNorm1d(H)
        self.bn5 = torch.nn.BatchNorm1d(H)

        # init models
        torch.nn.init.normal(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear1.bias, 0)
        torch.nn.init.normal(self.linear2.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear2.bias, 0)
        torch.nn.init.normal(self.linear3.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear3.bias, 0)
        torch.nn.init.normal(self.linear4.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear4.bias, 0)
        torch.nn.init.normal(self.linear5.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear5.bias, 0)
        torch.nn.init.normal(self.linear6.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear6.bias, 0)

    def forward(self, x):
        batchsize = x.size()[0]

        out1 = self.act1(self.bn1(self.linear1(x.view(batchsize, -1))))
        out2 = self.act2(self.bn2(self.linear2(out1)))
        out3 = self.act3(self.bn3(self.linear3(out2)))
        out4 = self.act4(self.bn4(self.linear4(out3)))
        out5 = self.act5(self.bn5(self.linear5(out4)))
        out6 = self.linear6(out5)

        return out6.view(batchsize, 2, -1) + x
        # return x

Pw = Variable(torch.from_numpy(np.load('Pw.npy')))
K = Variable(torch.from_numpy(np.load('K.npy')))

dataset = PnPDataset()
# trainSize = int(len(dataset)*0.8)
# trainSet, valSet = random_split(dataset, [trainSize, len(dataset)-trainSize])
dataloaderTrain = DataLoader(dataset, batch_size=100, shuffle=True)
# dataloaderVal = DataLoader(valSet, batch_size=1)

# nnModel = STN3d(8).double()
nnModel = KernelMd(2*len(Pw), 256, 2*len(Pw)).double()

learning_rate = 1e-3
optimizer = torch.optim.Adam(nnModel.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(nnModel.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)

for epoch in range(500):
    for i, sample in enumerate(dataloaderTrain):
        Pi, gPi, gRt = sample

        Pi = Variable(Pi)
        gPi = Variable(gPi)
        gRt = Variable(gRt)

        outnn = nnModel(Pi)

        trainLoss = torch.DoubleTensor([0])
        repLoss = torch.DoubleTensor([0])
        for n in range(len(outnn)):
            R, T = PnP.EPPnP(K, Pw, outnn[n, :].t())
            # R, T = PnP.EPPnP(K, Pw, Pi[n])
            rgt = gRt[n, :].view(-1, 3)
            rg = rgt[:3, :]
            tg = rgt[3, :]

            th, tr = PnP.rtLoss(K, Pw, R, T, rg, tg, 0)
            trainLoss = trainLoss + (th+tr)

            repLoss = repLoss + PnP.rtLoss(K, Pw, R, T, rg, tg, 1)

        trainLoss /= len(outnn)
        repLoss /= len(outnn)
        print(epoch, i, trainLoss.data.item(), repLoss.data.item())

        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

    scheduler.step()