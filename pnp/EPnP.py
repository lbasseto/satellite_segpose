import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

from tensorboardX import SummaryWriter

from PnP import EPPnP, rtLoss

import numpy as np
import math
import random

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

def CheckDataBatch(Pc, gPc, gRt):

    N = len(Pc)
    assert(N == len(gPc) and N == len(gRt))

    avg1 = 0
    avg2 = 0
    for i in range(N):
        pc = Pc[i, :].view(-1, 2)
        gpc = gPc[i, :].view(-1, 2)
        grt = gRt[i, :].view(-1, 3)
        R = grt[:3, :]
        T = grt[3, :]

        guv = Pw.mm(R) + T
        guv[:, 0] = guv[:, 0] / guv[:, 2]
        guv[:, 1] = guv[:, 1] / guv[:, 2]
        guv = guv[:, :2]

        err1 = torch.norm(guv-gpc)
        err2 = torch.norm(guv-pc)

        avg1 += err1.data[0]
        avg2 += err2.data[0]

    avg1 /= N
    avg2 /= N

    print(avg1, avg2)

Pw = Variable(torch.from_numpy(np.load('Pw.npy')))
K = Variable(torch.from_numpy(np.load('K.npy')))

dataset = PnPDataset()
trainSize = int(len(dataset)*0.8)
trainSet, valSet = random_split(dataset, [trainSize, len(dataset)-trainSize])

dataloaderTrain = DataLoader(trainSet, batch_size=128, shuffle=True)
dataloaderVal = DataLoader(valSet, batch_size=128)

class KernelMd(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(KernelMd, self).__init__()

        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

        # self.act1 = torch.nn.LeakyReLU(0.1, inplace=True)
        # self.act2 = torch.nn.LeakyReLU(0.1, inplace=True)
        # self.act1 = torch.nn.ReLU()
        # self.act2 = torch.nn.ReLU()
        self.act1 = torch.nn.Tanh()
        self.act2 = torch.nn.Tanh()

        self.bn1 = torch.nn.BatchNorm1d(H)
        self.bn2 = torch.nn.BatchNorm1d(H)

        # init models
        torch.nn.init.normal(self.linear1.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear1.bias, 0)
        torch.nn.init.normal(self.linear2.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear2.bias, 0)
        torch.nn.init.normal(self.linear3.weight, mean=0, std=0.01)
        torch.nn.init.constant(self.linear3.bias, 0)

    def forward(self, x):
        out1 = self.act1(self.bn1(self.linear1(x)))
        out2 = self.act2(self.bn2(self.linear2(out1)))
        # out1 = self.act1(self.linear1(x))
        # out2 = self.act2(self.linear2(out1))
        out3 = self.linear3(out2)

        return out3 + x

# nnModel = KernelMd(2*len(Pw), 256, 7).double()
nnModel = KernelMd(2*len(Pw), 256, 2*len(Pw)).double()

learning_rate = 1e-3
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(nnModel.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(nnModel.parameters(), lr=learning_rate) # so poor ?!
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)

writer = SummaryWriter('runs/pnp4')

# for epoch in range(500):
#     for i, sample in enumerate(dataloaderTrain):
#         Pi, gPi, gRt = sample
#         Pi = Variable(Pi)
#         gPi = Variable(gPi)
#         gRt = Variable(gRt)
#
#         outnn = nnModel(Pi)
#         loss = loss_fn(outnn, gPi)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if i % 10 == 0:
#             writer.add_scalar('loss', loss.data.item(), epoch * len(dataloaderTrain) + i)
#             print(epoch, i, loss.data.item())
#
#             # R, T = EPPnP(Pw, outnn[0, :].view(-1, 2))
#             # print(torch.cat((R,T.view(1,-1)), 0))
#             # print(gRt[0,:].view(-1, 3))
#
#     # validation
#     validLoss = 0
#     N = 0
#     for k, valSample in enumerate(dataloaderVal):
#         Pi, gPi, gRt = valSample
#         Pi = Variable(Pi)
#         gPi = Variable(gPi)
#         gRt = Variable(gRt)
#         outnn = nnModel(Pi)
#
#         for n in range(len(outnn)):
#             # use PNP
#             R, T = EPPnP(K, Pw, outnn[n, :].view(-1, 2))
#
#             if n == 0 and k == 0:
#                 print(torch.cat((R, T.view(1, -1)), 0))
#                 print(gRt[0, :].view(-1, 3))
#
#             rgt = gRt[n, :].view(-1, 3)
#             rg = rgt[:3, :]
#             tg = rgt[3, :]
#             loss = rtLoss(K, Pw, R, T, rg, tg, 1)
#
#             validLoss += loss.data.item()
#             N += 1
#
#     validLoss /= N
#
#     print(epoch, validLoss)
#     writer.add_scalar('val_loss0', validLoss, epoch * len(dataloaderTrain) + i)
#
#     scheduler.step()
# exit(0)

for epoch in range(500):
    for i, sample in enumerate(dataloaderTrain):
        # print(i)
        # print(sample)

        Pi, gPi, gRt = sample

        Pi = Variable(Pi)
        gPi = Variable(gPi)
        gRt = Variable(gRt)

        # CheckDataBatch(Pc, gPc, gRt)

        outnn = nnModel(Pi)

        # use PNP

        R, T = EPPnP(K, Pw, outnn[0, :].view(-1, 2), 1)
        # q = outnn[0, :4]
        # R = quater2mat(q)
        # T = outnn[0, 4:]

        # if i == 0:
        #     # print(outnn[0, :].view(-1, 3))
        #     print(torch.cat((R,T.view(1,-1)), 0))
        #     print(gRt[0,:].view(-1, 3))
        # # loss = loss_fn(outnn, gRt)

        rgt = gRt[0,:].view(-1,3)
        rg = rgt[:3,:]
        tg = rgt[3,:]
        trainLoss = rtLoss(K, Pw, R, T, rg, tg, 1)
        for n in range(len(outnn)):
            if n == 0:
                continue

            R, T = EPPnP(K, Pw, outnn[n, :].view(-1, 2))
            # q = outnn[n, :4]
            # R = quater2mat(q)
            # T = outnn[n, 4:]

            rgt = gRt[n, :].view(-1, 3)
            rg = rgt[:3, :]
            tg = rgt[3, :]

            trainLoss = trainLoss + rtLoss(K, Pw, R, T, rg, tg, 1)

        optimizer.zero_grad()

        trainLoss /= len(outnn)

        # print(loss.data[0])
        trainLoss.backward()

        optimizer.step()

        if i % 10 == 0:
            print(epoch, i, trainLoss.data.item())
            writer.add_histogram('input', Pi.data.numpy(), epoch * len(dataloaderTrain) + i)

            writer.add_histogram('w1', nnModel.linear1.weight.data.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('w2', nnModel.linear2.weight.data.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('w3', nnModel.linear3.weight.data.numpy(), epoch * len(dataloaderTrain) + i)
            #
            writer.add_histogram('b1', nnModel.linear1.bias.data.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('b2', nnModel.linear2.bias.data.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('b3', nnModel.linear3.bias.data.numpy(), epoch * len(dataloaderTrain) + i)
            #
            writer.add_histogram('w1g', nnModel.linear1.weight.grad.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('w2g', nnModel.linear2.weight.grad.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('w3g', nnModel.linear3.weight.grad.numpy(), epoch * len(dataloaderTrain) + i)
            #
            writer.add_histogram('b1g', nnModel.linear1.bias.grad.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('b2g', nnModel.linear2.bias.grad.numpy(), epoch * len(dataloaderTrain) + i)
            # writer.add_histogram('b3g', nnModel.linear3.bias.grad.numpy(), epoch * len(dataloaderTrain) + i)

            writer.add_scalar('loss0', trainLoss.data.item(), epoch * len(dataloaderTrain) + i)

    # validation
    validLoss = 0
    for k, valSample in enumerate(dataloaderVal):
        Pi, gPi, gRt = valSample
        Pi = Variable(Pi)
        gPi = Variable(gPi)
        gRt = Variable(gRt)
        outnn = nnModel(Pi)

        for n in range(len(outnn)):
            # use PNP
            R, T = EPPnP(K, Pw, outnn[0, :].view(-1, 2))
            # q = outnn[n, :4]
            # R = quater2mat(q)
            # T = outnn[n, 4:]

            if n==0 and k == 0:
                print(torch.cat((R, T.view(1, -1)), 0))
                print(gRt[0, :].view(-1, 3))

            rgt = gRt[n, :].view(-1, 3)
            rg = rgt[:3, :]
            tg = rgt[3, :]
            loss = rtLoss(K, Pw, R, T, rg, tg, 1)

            validLoss += loss.data.item()

    validLoss /= len(valSet)

    writer.add_scalar('val_loss0', validLoss, epoch * len(dataloaderTrain) + i)

    scheduler.step()