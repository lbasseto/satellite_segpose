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
        self.gCc = np.load('gCc.npy')
        assert (len(self.Pi) == len(self.gPi))
        assert (len(self.Pi) == len(self.gRt))
        assert (len(self.Pi) == len(self.gCc))

    def __len__(self):
        return len(self.Pi)

    def __getitem__(self, idx):
        pi = self.Pi[idx, :]
        gpi = self.gPi[idx, :]
        grt = self.gRt[idx, :]
        gcc = self.gCc[idx, :]
        return pi, gpi, grt, gcc

Pw = Variable(torch.from_numpy(np.load('Pw.npy')))
K = Variable(torch.from_numpy(np.load('K.npy')))

dataset = PnPDataset()
# trainSize = int(len(dataset)*0.8)
# trainSet, valSet = random_split(dataset, [trainSize, len(dataset)-trainSize])
dataloaderTrain = DataLoader(dataset, batch_size=1, shuffle=False)
# dataloaderVal = DataLoader(valSet, batch_size=1)

writer = SummaryWriter('runs/test1_0')

for i, sample in enumerate(dataloaderTrain):
    Pi, gPi, gRt, gCc = sample

    Pi = Variable(Pi, requires_grad=True)
    gPi = Variable(gPi)
    gRt = Variable(gRt)
    gCc = Variable(gCc)

    lr = 1e4
    loss = 100
    lastR = None
    lastT = None
    step = 0

    while loss > 1e-7:
        quatLoss = Variable(torch.from_numpy(np.array(0.0)))
        reprojLoss = Variable(torch.from_numpy(np.array(0.0)))
        efLoss = Variable(torch.from_numpy(np.array(0.0)))
        for n in range(len(Pi)):
            uv = Pi[n].view(-1, 2)
            cc = gCc[n].view(-1, 1)

            # normalize the uv
            uv1 = torch.cat((uv, torch.ones(len(uv), 1.0).double()), 1)
            uv = torch.inverse(K).mm(uv1.t()).t()
            uv = uv[:, :2]
            Cw, Alpha, M = PnP.PrepareData(Pw, uv)
            mtm = M.t().mm(M)
            eigenFreeLoss = cc.t().mm(mtm).mm(cc)
            efLoss = efLoss + eigenFreeLoss

            # get RT from M
            Km = PnP.GetKernel(M)
            vK = Km[:, 3]  # take the last column: the one with smallest eigenvalues
            vK = vK.view(4, 3)
            m = vK[:, 2].mean()
            if (m <= 0).data.numpy():
                vK = -vK
            R, T, r = PnP.Procrustes(Cw, vK)

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

                writer.add_scalar(str(i) + '/error_eigenfree', eigenFreeLoss.data.item(), step)

            lastR = R
            lastT = T
            step = step + 1

        quatLoss = quatLoss / len(Pi)
        reprojLoss = reprojLoss / len(Pi)
        efLoss = efLoss / len(Pi)

        print(i, quatLoss.data.item(), reprojLoss.data.item(), efLoss.data.item())

        efLoss.backward()
        loss = efLoss.data.item()

        Pi.data -= lr * Pi.grad.data
        Pi.grad.data.zero_()
