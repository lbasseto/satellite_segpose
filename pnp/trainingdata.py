import torch
from torch.autograd import Variable

import numpy as np
import math
import random

import PnP

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

def Rand(min, max):
    return min + (max - min) * random.random()

def CheckRatation(R):
    # check the validation of ration matrix
    tR = np.matrix(R)
    ident = tR * tR.transpose()
    identErr = np.linalg.norm(ident - np.identity(3))
    assert (abs(identErr) <= 1e-10)

    det = np.linalg.det(tR)
    assert (abs(det - 1.0) <= 1e-10)

    return identErr, det

def RandomRotation():
    range = 1

    # use eular formulation, three different rotation angles on 3 axis
    phi = Rand(0, range * math.pi * 2)
    theta = Rand(0, range * math.pi)
    psi = Rand(0, range * math.pi * 2)

    R0 = []
    R0.append(math.cos(psi) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.sin(psi))
    R0.append(math.cos(psi) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.sin(psi))
    R0.append(math.sin(psi) * math.sin(theta))

    R1 = []
    R1.append(-math.sin(psi) * math.cos(phi) - math.cos(theta) * math.sin(phi) * math.cos(psi))
    R1.append(-math.sin(psi) * math.sin(phi) + math.cos(theta) * math.cos(phi) * math.cos(psi))
    R1.append(math.cos(psi) * math.sin(theta))

    R2 = []
    R2.append(math.sin(theta) * math.sin(phi))
    R2.append(-math.sin(theta) * math.cos(phi))
    R2.append(math.cos(theta))

    R = []
    R.append(R0)
    R.append(R1)
    R.append(R2)

    # print(R)
    CheckRatation(R)

    return np.array(R)

def RandomTranslation():
    tx = Rand(-2, 2)
    ty = Rand(-2, 2)
    tz = Rand(4, 5)
    # tz = Rand(4, 8)
    return np.array([[tx,ty,tz]])

def DrawCubeBorder(plt, xs, ys, param):
    # 12 borders
    tx = np.array([xs[0], xs[1], xs[3], xs[2], xs[0], xs[4], xs[5], xs[7], xs[6], xs[4],
                   xs[6], xs[2], xs[3], xs[7], xs[5], xs[1]])
    ty = np.array([ys[0], ys[1], ys[3], ys[2], ys[0], ys[4], ys[5], ys[7], ys[6], ys[4],
                   ys[6], ys[2], ys[3], ys[7], ys[5], ys[1]])
    plt.plot(tx, ty, param, linewidth=5)

def DrawPoseCube(plt, K, R, T, param):
    # 12 borders
    eightCorners = torch.DoubleTensor([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]])
    ty = eightCorners.mm(R) + T
    ty = ty.mm(K.t())
    xs = (ty[:, 0] / ty[:, 2]).detach().numpy()
    ys = (ty[:, 1] / ty[:, 2]).detach().numpy()
    x = np.array([xs[0], xs[1], xs[3], xs[2], xs[0], xs[4], xs[5], xs[7], xs[6], xs[4],
                   xs[6], xs[2], xs[3], xs[7], xs[5], xs[1]])
    y = np.array([ys[0], ys[1], ys[3], ys[2], ys[0], ys[4], ys[5], ys[7], ys[6], ys[4],
                   ys[6], ys[2], ys[3], ys[7], ys[5], ys[1]])
    plt.plot(x, y, param, linewidth=5)

def DrawAxis(plt, K, R, T, linewidth=2):
    if True:
        anchors = np.array([[0,0,0], [0,0,1], [0,1,0], [1,0,0]])
        anchors = torch.from_numpy(anchors).double()
        ty = anchors.mm(R) + T
        ty = ty.mm(K.t())
        tu = (ty[:, 0] / ty[:, 2]).detach().numpy()
        tv = (ty[:, 1] / ty[:, 2]).detach().numpy()
        plt.arrow(tu[0], tv[0], tu[1] - tu[0], tv[1] - tv[0], head_width=5, color='r', linewidth=linewidth)
        plt.arrow(tu[0], tv[0], tu[2] - tu[0], tv[2] - tv[0], head_width=5, color='g', linewidth=linewidth)
        plt.arrow(tu[0], tv[0], tu[3] - tu[0], tv[3] - tv[0], head_width=5, color='b', linewidth=linewidth)

    if False:
        Q = PnP.mat2quater(R)
        xyz = Q[1:].view(-1,3)
        xyz = torch.cat((torch.from_numpy(np.array([[0, 0, 0]])).double(), xyz))
        qy = xyz.mm(R) + T
        qy = qy.mm(K.t())
        qu = (qy[:, 0] / qy[:, 2]).numpy()
        qv = (qy[:, 1] / qy[:, 2]).numpy()
        plt.arrow(qu[0], qv[0], qu[1] - qu[0], qv[1] - qv[0], head_width=5, color='y', linewidth=linewidth)

def WorldPoints():
    if False:
        numberPoints = 1000

        ptList = []
        for i in range(numberPoints):
            pt = np.random.randn(1, 3)[0]
            pt /= np.linalg.norm(pt)

            ptList.append(pt.tolist())

        return np.array(ptList)

    else:
        supportedNumbers = [6,8,14,20,26]
        # assert(pointNumber in supportedNumbers)

        origin = [[0.0, 0.0, 0.0]]

        sixPlainCenters = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],
                           [-0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]]

        eightCorners = [[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
                        [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]]

        twelveLineCenters = [[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.5, 0.0], [-0.5, -0.5, 0.0],
                             [0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [-0.5, 0.0, 0.5], [-0.5, 0.0, -0.5],
                             [0.0, 0.5, 0.5], [0.0, 0.5, -0.5], [0.0, -0.5, 0.5], [0.0, -0.5, -0.5]]

        # basePoints = np.array(eightCorners + origin + sixPlainCenters + twelveLineCenters)
        # basePoints = np.array(eightCorners  + twelveLineCenters)
        # basePoints = np.array(sixPlainCenters)
        basePoints = np.array(eightCorners)

        # return basePoints
        return np.concatenate((basePoints, basePoints*0.5, basePoints*0.25), axis=0)

def GenerateTrainingSamples(Pw, sampleCnt, noise):
    # the world points are fixed positions

    f = 800.0 # assume the focus length
    width = 640
    height = 480
    u0 = 320.0
    v0 = 240.0
    # intrinsic Matrix
    K = np.array([[f, 0.0, u0], [0.0, f, v0], [0.0, 0.0, 1.0]])

    # virtual control points, in world frame
    Cw = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    Pi = []
    gPi = [] # GT projection on camera frame
    gRt = [] # GT rotation and translation
    gCc = [] # GT projection of control points
    for i in range(sampleCnt):
        rt = None
        Ximg = None
        cc = None
        # repeat generate samples until all points are in the image range
        while True:
            gtR = RandomRotation()
            gtT = RandomTranslation()
            rt = np.concatenate((gtR, gtT))

            Xcam = np.matmul(Pw, gtR) + gtT

            cc = (np.matmul(Cw, gtR) + gtT).reshape(1,-1)
            cc /= np.linalg.norm(cc)

            # projection
            Ximg = np.matmul(Xcam, K.transpose())
            Ximg[:, 0] = Ximg[:, 0] / Ximg[:, 2]
            Ximg[:, 1] = Ximg[:, 1] / Ximg[:, 2]
            Ximg = Ximg[:, :2]

            # until all points are in the image range
            minx = min(Ximg[:, 0])
            maxx = max(Ximg[:, 0])
            miny = min(Ximg[:, 1])
            maxy = max(Ximg[:, 1])
            if minx >= 0 and maxx < width and miny >= 0 and maxy < height:
                break

        gCc.append(cc.tolist()[0])
        gRt.append(rt.reshape((1, -1)).tolist()[0])
        gPi.append(Ximg.reshape((1, -1)).tolist()[0])

        # add the noise
        gn = np.random.randn(len(Pw), 2) * noise
        res = Ximg + gn

        # draw
        if False:
            plt.scatter(Ximg[:,0], Ximg[:,1])
            DrawCubeBorder(plt, Ximg[:,0], Ximg[:,1], 'b')

            plt.scatter(res[:,0], res[:,1])
            plt.xlim(0, 640)
            plt.ylim(0, 480)
            plt.show()

        Pi.append(res.reshape((1, -1)).tolist()[0])

    Pi = np.array(Pi)
    gPi = np.array(gPi)
    gRt = np.array(gRt)
    gCc = np.array(gCc)

    return K, Pi, gPi, gRt, Cw, gCc

def TestRawPnP(K, actXimg, gXimg, gRt, Cw, gCc, Pw):
    shape= list(actXimg.shape)
    N = shape[0]
    M = shape[1]

    K = Variable(torch.from_numpy(K))
    Pw = Variable(torch.from_numpy(Pw))
    Pi = Variable(torch.from_numpy(actXimg))
    gPi = Variable(torch.from_numpy(gXimg))
    gRt = Variable(torch.from_numpy(gRt))
    Cw = Variable(torch.from_numpy(Cw))
    gCc = Variable(torch.from_numpy(gCc))

    AvgReprojErr = 0
    AvgRotationErr = 0
    AvgTransErr = 0
    for i in range(N):
        pi1 = Pi[i, :].view(-1, 2)
        gpi1 = gPi[i, :].view(-1, 2)
        grt1 = gRt[i, :].view(-1, 3)
        gcc1 = gCc[i, :].view(-1, 3)

        gR = grt1[:3, :]
        gT = grt1[3, :]
        R, T = PnP.EPPnP(K, Pw, pi1)
        # R, T, r = PnP.Procrustes(Cw, gcc1)

        rpjErr = PnP.rtLoss(K, Pw, R, T, gR, gT, 1)
        rotErr, transErr = PnP.rtLoss(K, Pw, R, T, gR, gT, 0)

        # esy = Pw.mm(R) + T
        # esy[:, 0] = esy[:, 0] / esy[:, 2]
        # esy[:, 1] = esy[:, 1] / esy[:, 2]
        # esy = esy[:, :2]

        # gtLoss = (gpc1 - pc1).pow(2).sum()
        # loss = (gpc1 - esy).pow(2).sum()

        if i%1000 == 0:
            print('--------')
            print(grt1)
            print(torch.cat((R, T), 0))
            print('   reproj error: ', rpjErr.data.item())
            print('   rotation error: ', rotErr.data.item())
            print('   trans error: ', transErr.data.item())

            # draw
            if False:
                DrawAxis(plt, K, R, T, 1)
                DrawAxis(plt, K, gR, gT)
                plt.xlim(0, 640)
                plt.ylim(0, 480)
                # plt.show()


            if False:
                plt.scatter(pi1[:,0].numpy(), pi1[:,1].numpy())

                # plt.scatter(gpc1[:, 0].numpy(), gpc1[:, 1].numpy())
                DrawCubeBorder(plt, gpi1[:, 0].numpy(), gpi1[:, 1].numpy(), 'g')

                ty = Pw.mm(R) + T
                ty = ty.mm(K.t())
                tu = ty[:, 0] / ty[:, 2]
                tv = ty[:, 1] / ty[:, 2]
                # plt.scatter(tu.numpy(), tv.numpy())
                DrawCubeBorder(plt, tu.numpy(), tv.numpy(), 'r')

                plt.xlim(0, 640)
                plt.ylim(0, 480)
                plt.show()


        # avgGtErr += gtLoss.data[0]
        AvgReprojErr += rpjErr.data.item()
        AvgRotationErr += rotErr.data.item()
        AvgTransErr += transErr.data.item()

    print('\n--------AVG--------')
    print('   reproj error: ', AvgReprojErr / N)
    print('   rotation error: ', AvgRotationErr / N)
    print('   trans error: ', AvgTransErr / N)
    return

if __name__ == "__main__":
    TotalSampleCnt = 10*1000
    Pw = WorldPoints()
    K, Pi, gPi, gRt, Cw, gCc = GenerateTrainingSamples(Pw, TotalSampleCnt, 500)

    TestRawPnP(K, Pi, gPi, gRt, Cw, gCc, Pw)

    np.save('K', K)
    np.save('Pi', Pi)
    np.save('gPi', gPi)
    np.save('gRt', gRt)
    np.save('gCc', gCc)
    np.save('Pw', Pw)
    np.save('Cw', Cw)
