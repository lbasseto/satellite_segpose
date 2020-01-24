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

from torchviz import make_dot

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

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2

def construct_pred_MtM(nB, vpoints, x, y):
    nV = len(vpoints)

    rx = x.view(1, -1)
    ry = y.view(1, -1)

    # normalize 2d points
    rz = torch.ones(rx.shape)
    rxyz = torch.cat((rx, ry, rz), 0)
    uv = torch.inverse(K).mm(rxyz)[:2].t()

    # left M
    lm = torch.cat((torch.eye(2,2).repeat(nB*nV, 1), (-uv).view(-1,1)), 1)
    lm = lm.view(nB*nV, 2, 3)

    # right M
    # virtual control points, in world frame
    # the default 4 control points: Cw = Variable(torch.from_numpy(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])))
    # get the representation of 3D world points in
    # linear combination of the control points
    co = 1.0 - (vpoints[:, 0] + vpoints[:, 1] + vpoints[:, 2])
    Alphas = torch.cat((vpoints, co.view(-1, 1)), 1)
    rm = kronecker_product(Alphas, torch.eye(3, 3)).repeat(nB, 1)
    rm = rm.view(nB*nV, 3, 12)

    M = torch.bmm(lm, rm).view(nB, 2*nV, 12)
    mtm = torch.bmm(M.transpose(1, 2), M)

    return mtm.view(nB, 12 * 12)

Pw = Variable(torch.from_numpy(np.load('Pw.npy'))).float()
K = Variable(torch.from_numpy(np.load('K.npy'))).float()

bSize = 16
dataset = PnPDataset()
# trainSize = int(len(dataset)*0.8)
# trainSet, valSet = random_split(dataset, [trainSize, len(dataset)-trainSize])
dataloaderTrain = DataLoader(dataset, batch_size=bSize, shuffle=False)
# dataloaderVal = DataLoader(valSet, batch_size=1)

writer = SummaryWriter('runs/test1_0')

for i, sample in enumerate(dataloaderTrain):
    Pi, gPi, gRt, gCc = sample

    Pi = Variable(Pi.float(), requires_grad=True)
    gPi = Variable(gPi).float()
    gRt = Variable(gRt).float()
    gCc = Variable(gCc).float()

    lr = 1e-3
    loss = 100
    lastR = None
    lastT = None
    step = 0

    while loss > 1e-7:
        # quatLoss = Variable(torch.from_numpy(np.array(0.0)))
        # reprojLoss = Variable(torch.from_numpy(np.array(0.0)))
        # efLoss = Variable(torch.from_numpy(np.array(0.0)))

        xy = Pi.view(bSize, -1, 2)
        x = xy[:, :, 0]
        y = xy[:, :, 1]
        mtm = construct_pred_MtM(bSize, Pw, x, y)
        xtmtmx = torch.bmm(torch.bmm(gCc.view(-1, 1, 12), mtm.view(-1, 12, 12)), gCc.view(-1, 12, 1)).view(-1)

        efLoss = 1e7 * xtmtmx.sum()

        if False:
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

        # quatLoss = quatLoss / len(Pi)
        # reprojLoss = reprojLoss / len(Pi)
        # efLoss = efLoss / len(Pi)

        # print(i, quatLoss.data.item(), reprojLoss.data.item(), efLoss.data.item())
        print(i, efLoss.data.item())

        efLoss.backward()
        loss = efLoss.data.item()

        # make_dot(efLoss).view()

        Pi.data -= lr * Pi.grad.data
        Pi.grad.data.zero_()
