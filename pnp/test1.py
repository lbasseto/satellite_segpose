import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from tensorboardX import SummaryWriter

import PnP

import numpy as np
import math
import random

from tensorboardX import SummaryWriter

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
        pi = self.Pi[idx, :]
        gpi = self.gPi[idx, :]
        grt = self.gRt[idx, :]
        return pi, gpi, grt

Pw = Variable(torch.from_numpy(np.load('Pw.npy')))
K = Variable(torch.from_numpy(np.load('K.npy')))

dataset = PnPDataset()
# trainSize = int(len(dataset)*0.8)
# trainSet, valSet = random_split(dataset, [trainSize, len(dataset)-trainSize])
dataloaderTrain = DataLoader(dataset, batch_size=1, shuffle=False)
# dataloaderVal = DataLoader(valSet, batch_size=1)

writer = SummaryWriter('runs/test1_0')

for i, sample in enumerate(dataloaderTrain):
    Pi, gPi, gRt = sample

    Pi = Variable(Pi, requires_grad=True)
    gPi = Variable(gPi)
    gRt = Variable(gRt)

    lr = 1e-2
    loss = 100
    lastR = None
    lastT = None
    step = 0

    while loss > 1:
        quatLoss = Variable(torch.from_numpy(np.array(0.0)))
        reprojLoss = Variable(torch.from_numpy(np.array(0.0)))
        for n in range(len(Pi)):
            R, T = PnP.EPPnP(K, Pw, Pi[n].view(-1, 2))

            rgt = gRt[n, :].view(-1, 3)
            rg = rgt[:3, :]
            tg = rgt[3, :]
            theta, lt = PnP.rtLoss(K, Pw, R, T, rg, tg, 0)
            reproj = PnP.rtLoss(K, Pw, R, T, rg, tg, 1)
            quatLoss = quatLoss + (theta + lt)
            reprojLoss = reprojLoss + reproj

            if lastR is not None and step % 100 == 0:
                diffTh, diffTrans = PnP.rtLoss(K, Pw, R, T, lastR, lastT, 0)
                diffRep = PnP.rtLoss(K, Pw, R, T, lastR, lastT, 1)
                writer.add_scalar(str(i)+'/delta_Theta', diffTh.data.item(), step)
                writer.add_scalar(str(i)+'/delta_Translation', diffTrans.data.item(), step)
                writer.add_scalar(str(i)+'/delta_Reprojection', diffRep.data.item(), step)

                writer.add_scalar(str(i) + '/error_Theta', theta.data.item(), step)
                writer.add_scalar(str(i) + '/error_Translation', lt.data.item(), step)
                writer.add_scalar(str(i) + '/error_Reprojection', reproj.data.item(), step)

            lastR = R
            lastT = T
            step = step + 1

        quatLoss = quatLoss / len(Pi)
        reprojLoss = reprojLoss / len(Pi)

        print(i, quatLoss.data.item(), reprojLoss.data.item())

        reprojLoss.backward()
        loss = reprojLoss.data.item()

        Pi.data -= lr * Pi.grad.data
        Pi.grad.data.zero_()
