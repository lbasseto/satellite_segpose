import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import cv2
from utils import *

def Procrustes(X, Y):
    #
    # input: each line of X or Y is a point
    #
    # will return R, T, r
    # that's find these 3 best matrix to let R(X.t)+T = r(Y.t),
    # in which R is a rotation matrix
    #
    # here, use the Kabsch algorithm
    #

    N = len(X)  # number of points
    assert (len(Y) == N)

    # 1st step: translated first, so that their centroid coincides with the origin of the coordinate system.
    # This is done by subtracting from the point coordinates the coordinates of the respective centroid.
    ox = X.mean(0)  # mean X
    cx = X - ox  # center X
    nx_factor = cx.norm()  # to normalize
    nx = cx / nx_factor
    # print(nx)

    oy = Y.mean(0)  # mean Y
    cy = Y - oy  # center Y
    ny_factor = cy.norm()  # to normalize
    ny = cy / ny_factor
    # print(ny)

    # 2nd step: calculating a cross-covariance matrix A. or In matrix notation:
    A = nx.t().mm(ny)
    # print(A)

    # 3rd step
    U, S, V = torch.svd(A)
    # U,S,V = np.linalg.svd(A) # !!! V is already transposed
    # print(U)
    # print(S)
    # print(V)

    tmpR = V.mm(U.t())

    # compute determinant
    det = tmpR[0][0] * tmpR[1][1] * tmpR[2][2] + tmpR[0][1] * tmpR[1][2] * tmpR[2][0] + tmpR[0][2] * tmpR[1][0] * tmpR[2][1] \
          - tmpR[0][2] * tmpR[1][1] * tmpR[2][0] - tmpR[0][0] * tmpR[1][2] * tmpR[2][1] - tmpR[0][1] * tmpR[1][0] * tmpR[2][2]
    # det = np.linalg.det(R.data.numpy())
    # print(det)

    R = V.mm(torch.diag(torch.FloatTensor([1, 1, det]).type_as(A))).mm(U.t())

    # if det < 0:
        # use this term cause worse accuracy ?
        # R = V.mm(torch.diag(torch.FloatTensor([1,1,-1]).type_as(A))).mm(U.t())

        # use this term cause jitter?
        # R = -R

    scale = S.sum() * (nx_factor / ny_factor)
    t = R.mm(-ox.view(-1, 1)) + scale * oy.view(-1, 1)

    return R, t, scale

class PowerIteration(torch.autograd.Function):
    '''
    find the leading eigenvector by power iteration method
    refer to 'Deep Learning of Graph Matching, CVPR2018'
    '''
    @staticmethod
    def forward(ctx, M):
        Iter = 20  # fixed iteration number

        v0 = torch.ones(M.shape[1], 1).type_as(M)  # initial v
        v = [v0]
        for k in range(Iter):
            mvk = torch.mm(M, v[k])
            v.append(mvk / mvk.norm())  # this is v_k1

        ctx.intermediate_results = M, v  # save intermediate results for backward
        return v[Iter]

    @staticmethod
    def backward(ctx, grad_output):
        M, v = ctx.intermediate_results

        N = M.shape[1]
        Iter = len(v) - 1

        Lvk1 = grad_output
        eye = torch.eye(N).type_as(M)
        # calculate dL/dM iteratively
        Lm = 0
        for k in range(Iter - 1, -1, -1):
            mid = ((eye - torch.mm(v[k + 1], v[k + 1].t()))
                   / torch.mm(M, v[k]).norm()).mm(Lvk1)
            Lvk = M.mm(mid)
            Lm = Lm + mid.mm(v[k].t())
            Lvk1 = Lvk

        return Lm

def KroneckerProduct(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (t1.unsqueeze(2).unsqueeze(3).repeat(1, t2_height, t2_width, 1).view(out_height, out_width))
    return expanded_t1 * tiled_t2

def ComputeM(cpoints, k, uv, xyz):
    u = uv[:,0].view(-1, 1)
    v = uv[:,1].view(-1, 1)
    xyz = xyz.view(-1, 3)

    ptCnt = u.shape[0]
    assert(v.shape[0] == ptCnt and xyz.shape[0] == ptCnt)

    # normalize uv by k
    d = torch.ones(ptCnt).view(-1, 1).type_as(u)
    uvd = torch.cat((u, v, d), 1).t()
    uv = torch.inverse(k).mm(uvd)[:2].t()

    # left M
    lm = torch.cat((torch.eye(2, 2).repeat(ptCnt, 1).type_as(u), (-uv).view(-1, 1)), 1)
    lm = lm.view(ptCnt, 2, 3)

    # get the representation of 3D world points in
    # linear combination of the 4 control points
    pinv = torch.mm(cpoints.t(), cpoints).inverse().mm(cpoints.t()).t()
    Alphas = torch.mm(pinv, xyz.t()).t()
    Alphas = torch.cat(((1.0 - Alphas.sum(dim=1)).view(-1,1), Alphas[:, 1:]), 1)

    # right M
    rm = KroneckerProduct(Alphas, torch.eye(3, 3).type_as(u))
    rm = rm.view(ptCnt, 3, 12)

    M = torch.bmm(lm, rm).view(2*ptCnt, 12)
    return M

def ComputeMtM(cpoints, k, uv, xyz, weights):
    M = ComputeM(cpoints, k, uv, xyz)

    # weighted MTM
    if weights is None:
        mtm = torch.mm(M.t(), M)
    else:
        # # eat too much space
        # diagweights = torch.diag(weights.view(-1,1).repeat(1, 2).view(-1))
        # mtm = torch.mm(M.t(), diagweights).mm(M)
        mtm = torch.mm(M.t(), weights.view(-1, 1).repeat(1, 2).view(-1, 1).repeat(1, 12) * M)

    return mtm

def vertices_reprojection_tensor(vertices, r, t, k, width, height):
    p = torch.mm(k, torch.mm(r, vertices.t()) + t.view(-1, 1))
    x = (p[0] / (p[2] + 1e-5)) / width
    y = (p[1] / (p[2] + 1e-5)) / height
    return torch.cat((x.view(-1, 1), y.view(-1, 1)), 1)

def ComputePnP(controlpoints, mtm):
    q = 1e-4
    A = torch.inverse(mtm + q * torch.eye(mtm.shape[1]).type_as(mtm))

    evec = PowerIteration.apply(A)
    if evec.view(4, 3)[:, 2].mean() < 0:
        evec = -evec

    eval = evec.t().mm(A).mm(evec) / evec.t().mm(evec)  # get the corresponding eigenvalue
    eval = (1.0 / eval) - q

    # debug
    # if True:
    if False:
        U, S, V = torch.svd(mtm)
        print(S[8:], eval)
        print(V[:,11])
        print(evec)

    R, T, scale = Procrustes(controlpoints, evec.view(4, 3))

    return R, T, eval, evec, scale

if __name__ == "__main__":
    def Rand(min, max):
        return min + (max - min) * random.random()
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
        return np.array(R)
    def RandomTranslation():
        tx = Rand(-1, 1)
        ty = Rand(-1, 1)
        tz = Rand(6, 8)
        # tz = Rand(4, 8)
        return np.array([tx, ty, tz])

    def test_updateweights():
        control_points = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # virtual_points = torch.FloatTensor([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        #                                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
        virtual_points = 2 * (torch.rand(30, 3) - 0.5)

        K = torch.FloatTensor([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # intrinsic matrix
        width = 640
        height = 480

        # random.seed(2)
        # torch.manual_seed(0)

        gt_r = torch.from_numpy(RandomRotation()).float()
        gt_t = torch.from_numpy(RandomTranslation()).float()
        gt_v = (control_points.mm(gt_r.t()) + gt_t.view(1, -1)).view(-1, 1)
        gt_v = gt_v / gt_v.norm()

        rp3d = virtual_points
        # p3d = torch.rand((ptCnt, 3))
        # p3d.requires_grad = True

        # initialize the 2D points
        # p2d = torch.rand((ptCnt, 2))
        rp2d = vertices_reprojection_tensor(rp3d.data, gt_r, gt_t, K, width, height) + 0.0 * torch.rand(len(rp3d), 2)
        rp2d = rp2d[:10]
        # p2d.requires_grad = True

        # connect across them
        p2d = rp2d.repeat(1, len(rp3d)).view(-1,2)
        p3d = rp3d.repeat(len(rp2d), 1)
        assert(len(p2d) == len(p3d))

        # initialize the weights
        weights = torch.zeros(len(p2d), 1)
        weights.requires_grad = True

        IterCnt = 100000
        # learning_rate = 1e-5
        # learning_rate = 1e-4
        # learning_rate = 1e-3
        # learning_rate = 1e-2
        # learning_rate = 1e-1
        learning_rate = 1e3
        for iter in range(IterCnt):
            gt_rep = vertices_reprojection_tensor(p3d.data, gt_r, gt_t, K, width, height)
            gt_3dt = (virtual_points.data.mm(gt_r.t()) + gt_t.view(1, -1))

            t1 = time.time()

            # normalize weights according to rows
            wt = torch.sigmoid(weights)
            wt = wt.view(len(rp2d), len(rp3d))
            wt = wt / wt.sum(dim=1).view(-1, 1).repeat(1, len(rp3d))

            mtm = ComputeMtM(control_points, K, p2d, p3d, wt, width, height)
            R, T, ev = ComputePnP(control_points, mtm)

            pred_3dt = (virtual_points.data.mm(R.t()) + T.view(1, -1))

            losseig = ev
            # loss3dt = (pred_3dt-gt_3dt).pow(2).sum(dim=1).mean()
            loss3dt = (pred_3dt - gt_3dt).abs().sum(dim=1).mean()

            lossvmtmv = gt_v.t().mm(mtm).mm(gt_v)  # optimize xmtmx to 0
            # lossrep = (p2d+delta2d-gt_rep).pow(2).sum(dim=1).mean() # reprojection loss directly
            # losspnp = losseig + loss3dt
            # losspnp = losseig
            # losspnp = loss3dt

            loss = lossvmtmv
            # loss = loss3dt
            loss.backward()

            # update weights using gradient descent

            t2 = time.time()
            # print('%f' % (t2-t1))

            with torch.no_grad():
                # p2d.data.sub_(learning_rate * p2d.grad.data)
                # p2d.grad.data.zero_()
                # delta2d.data.sub_(learning_rate * delta2d.grad.data)
                # delta2d.grad.data.zero_()
                # p3d.data.sub_(learning_rate * p3d.grad.data)
                # p3d.grad.data.zero_()
                weights.data.sub_(learning_rate * weights.grad.data)
                weights.grad.data.zero_()
                if iter % 10 == 0:
                    print("At iter %d, the loss: %f, %f, %f" % (iter, losseig.data.item(), loss3dt.data.item(),
                                                                         lossvmtmv.data.item()))

            if iter % 10 == 0:
                # draw results
                showimg = np.zeros((height, width*2, 3), np.uint8)
                for i in range(len(p2d)):
                    gtx = int(gt_rep[i][0] * width) + width
                    gty = int(gt_rep[i][1] * height)
                    showimg = cv2.circle(showimg, (gtx, gty), 3, (0, 0, 255), -1, cv2.LINE_AA)
                    px = int((p2d).data[i][0] * width)
                    py = int((p2d).data[i][1] * height)
                    # print(px, py)
                    showimg = cv2.circle(showimg, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)

                    wv = wt.view(-1)[i]
                    assert( wv >= 0 and wv <= 1)
                    if wv > 0.1:
                        # draw line
                        pixv = int(wv * 255)
                        showimg = cv2.line(showimg, (gtx, gty), (px, py), (pixv, pixv, pixv),  1, cv2.LINE_AA)

                showimg = draw_axis(showimg, K.detach().numpy(), torch.cat((gt_r, gt_t.view(-1, 1)), 1).detach().numpy(), 1)
                showimg = draw_axis(showimg, K.detach().numpy(), torch.cat((R, T.view(-1, 1)), 1).detach().numpy(), 1)

                cv2.imshow("results", showimg)
                cv2.waitKey(10)

        cv2.waitKey(0)

    def test_update3d():
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt

        control_points = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # virtual_points = torch.FloatTensor([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        #                                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
        virtual_points = 2 * (torch.rand(1000, 3) - 0.5)

        K = torch.FloatTensor([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # intrinsic matrix
        width = 640
        height = 480

        # random.seed(2)
        # torch.manual_seed(0)

        gt_r = torch.from_numpy(RandomRotation()).float()
        gt_t = torch.from_numpy(RandomTranslation()).float()
        gt_v = (control_points.mm(gt_r.t()) + gt_t.view(1, -1)).view(-1, 1)
        gt_v = gt_v / gt_v.norm()

        ptCnt = len(virtual_points)
        p3d = torch.rand((ptCnt, 3))
        p3d.requires_grad = True

        # initialize the 2D points
        # p2d = torch.rand((ptCnt, 2))
        p2d = vertices_reprojection_tensor(virtual_points.data, gt_r, gt_t, K, width, height) + 0.0 * torch.rand(ptCnt, 2)
        # p2d.requires_grad = True

        # initialize the weights
        weights = torch.ones(ptCnt, 1)

        IterCnt = 10000
        # learning_rate = 1e-5
        # learning_rate = 1e-4
        # learning_rate = 1e-3
        # learning_rate = 1e-2
        learning_rate = 1e-1
        # learning_rate = 1e0
        for iter in range(IterCnt):
            gt_rep = vertices_reprojection_tensor(p3d.data, gt_r, gt_t, K, width, height)
            gt_3dt = (virtual_points.data.mm(gt_r.t()) + gt_t.view(1, -1))

            t1 = time.time()

            mtm = ComputeMtM(control_points, K, p2d, p3d, weights, width, height)
            R, T, ev = ComputePnP(control_points, mtm)

            pred_3dt = (virtual_points.data.mm(R.t()) + T.view(1, -1))

            losseig = ev
            # loss3dt = (pred_3dt-gt_3dt).pow(2).sum(dim=1).mean()
            loss3dt = (pred_3dt - gt_3dt).abs().sum(dim=1).mean()

            lossvmtmv = gt_v.t().mm(mtm).mm(gt_v)  # optimize xmtmx to 0
            # lossrep = (p2d+delta2d-gt_rep).pow(2).sum(dim=1).mean() # reprojection loss directly
            # losspnp = losseig + loss3dt
            # losspnp = losseig
            # losspnp = loss3dt

            loss = lossvmtmv
            # loss = loss3dt
            loss.backward()

            # update weights using gradient descent

            t2 = time.time()
            # print('%f' % (t2-t1))

            with torch.no_grad():
                # p2d.data.sub_(learning_rate * p2d.grad.data)
                # p2d.grad.data.zero_()
                # delta2d.data.sub_(learning_rate * delta2d.grad.data)
                # delta2d.grad.data.zero_()
                p3d.data.sub_(learning_rate * p3d.grad.data)
                p3d.grad.data.zero_()
                # weights.data.sub_(learning_rate * weights.grad.data)
                # weights.grad.data.zero_()
                if iter % 10 == 0:
                    print("At iter %d, the loss: %f, %f, %f" % (iter, losseig.data.item(), loss3dt.data.item(),
                                                                lossvmtmv.data.item()))

            if iter % 10 == 0:
                # draw results
                showimg = np.zeros((height, width, 3), np.uint8)
                p3d_rep = vertices_reprojection_tensor(p3d.data, gt_r, gt_t, K, width, height)
                for i in range(len(p2d)):
                    x3d = int(p3d_rep[i][0] * width)
                    y3d = int(p3d_rep[i][1] * height)
                    showimg = cv2.circle(showimg, (x3d, y3d), 5, (0, 0, 255), -1, cv2.LINE_AA)
                    x2d = int((p2d).data[i][0] * width)
                    y2d = int((p2d).data[i][1] * height)
                    # print(px, py)
                    showimg = cv2.circle(showimg, (x2d, y2d), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    # draw line
                    # showimg = cv2.line(showimg, (x3d, y3d), (x2d, y2d), (255,255,255), 1, cv2.LINE_AA)

                showimg = draw_axis(showimg, K.detach().numpy(), torch.cat((gt_r, gt_t.view(-1, 1)), 1).detach().numpy(), 1)
                showimg = draw_axis(showimg, K.detach().numpy(), torch.cat((R, T.view(-1, 1)), 1).detach().numpy(), 1)

                showimg = cv2.putText(showimg, "update 3d", (int(width/2-20), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))

                cv2.imshow("results", showimg)
                cv2.waitKey(10)

        #
        real3dnp = virtual_points.detach().numpy()
        pred3dnp = p3d.detach().numpy()
        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.scatter(real3dnp[:, 0], real3dnp[:, 1], real3dnp[:, 2], color='r', marker='^')
        ax3d.scatter(pred3dnp[:, 0], pred3dnp[:, 1], pred3dnp[:, 2], color='b', marker='o')

        # ax3d.plot_trisurf(real3dnp[:, 0], real3dnp[:, 1], real3dnp[:, 2])
        # ax3d.plot_trisurf(pred3dnp[:, 0], pred3dnp[:, 1], pred3dnp[:, 2])
        # plt.xlim(0, 640)
        # plt.ylim(0, 480)
        plt.show()

        cv2.waitKey(0)

    def test_updateRT():
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt

        control_points = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # virtual_points = torch.FloatTensor([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        #                                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
        virtual_points = 2 * (torch.rand(100, 3) - 0.5)

        K = torch.FloatTensor([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # intrinsic matrix
        width = 640
        height = 480

        # random.seed(2)
        # torch.manual_seed(0)

        gt_r = torch.from_numpy(RandomRotation()).float()
        gt_t = torch.from_numpy(RandomTranslation()).float()
        gt_v = (control_points.mm(gt_r.t()) + gt_t.view(1, -1)).view(-1, 1)
        gt_v = gt_v / gt_v.norm()

        ptCnt = len(virtual_points)
        p3d = torch.rand((ptCnt, 3))
        p3d.requires_grad = True

        # initialize the 2D points
        # p2d = torch.rand((ptCnt, 2))
        p2d = vertices_reprojection_tensor(virtual_points.data, gt_r, gt_t, K, width, height) + 0.0 * torch.rand(ptCnt, 2)
        # p2d.requires_grad = True

        # initialize the weights
        weights = torch.ones(ptCnt, 1)

        IterCnt = 10000
        # learning_rate = 1e-5
        learning_rate = 1e-4
        # learning_rate = 1e-3
        # learning_rate = 1e-2
        # learning_rate = 1e-1
        # learning_rate = 1e0
        for iter in range(IterCnt):
            gt_rep = vertices_reprojection_tensor(p3d.data, gt_r, gt_t, K, width, height)
            gt_3dt = (virtual_points.data.mm(gt_r.t()) + gt_t.view(1, -1))

            t1 = time.time()
            p2d[:,0] *= width
            p2d[:,1] *= height
            mtm = ComputeMtM(control_points, K, p2d, p3d, weights)
            R, T, eval, evec, scale = ComputePnP(control_points, mtm)

            pred_3dt = (virtual_points.data.mm(R.t()) + T.view(1, -1))

            losseig = eval
            loss3dt = (pred_3dt-gt_3dt).pow(2).sum(dim=1).mean()
            # loss3dt = (pred_3dt - gt_3dt).abs().sum(dim=1).mean()

            lossvmtmv = gt_v.t().mm(mtm).mm(gt_v)  # optimize xmtmx to 0
            # lossrep = (p2d+delta2d-gt_rep).pow(2).sum(dim=1).mean() # reprojection loss directly
            # losspnp = losseig + loss3dt
            # losspnp = losseig

            # loss = lossvmtmv
            # loss = loss3dt + lossvmtmv
            loss = loss3dt
            loss.backward()

            # update weights using gradient descent

            t2 = time.time()
            # print('%f' % (t2-t1))

            with torch.no_grad():
                # p2d.data.sub_(learning_rate * p2d.grad.data)
                # p2d.grad.data.zero_()
                # delta2d.data.sub_(learning_rate * delta2d.grad.data)
                # delta2d.grad.data.zero_()
                p3d.data.sub_(learning_rate * p3d.grad.data)
                p3d.grad.data.zero_()
                # weights.data.sub_(learning_rate * weights.grad.data)
                # weights.grad.data.zero_()
                if iter % 10 == 0:
                    print("At iter %d, the loss: %f, %f, %f" % (iter, losseig.data.item(), loss3dt.data.item(), lossvmtmv.data.item()))

            if iter % 10 == 0:
                # draw results
                showimg = np.zeros((height, width, 3), np.uint8)
                p3d_rep = vertices_reprojection_tensor(p3d.data, gt_r, gt_t, K, width, height)
                for i in range(len(p2d)):
                    x3d = int(p3d_rep[i][0] * width)
                    y3d = int(p3d_rep[i][1] * height)
                    showimg = cv2.circle(showimg, (x3d, y3d), 5, (0, 0, 255), -1, cv2.LINE_AA)
                    x2d = int((p2d).data[i][0] * width)
                    y2d = int((p2d).data[i][1] * height)
                    # print(px, py)
                    showimg = cv2.circle(showimg, (x2d, y2d), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    # draw line
                    # showimg = cv2.line(showimg, (x3d, y3d), (x2d, y2d), (255,255,255), 1, cv2.LINE_AA)

                showimg = draw_axis(showimg, K.detach().numpy(), torch.cat((gt_r, gt_t.view(-1, 1)), 1).detach().numpy(), 1)
                showimg = draw_axis(showimg, K.detach().numpy(), torch.cat((R, T.view(-1, 1)), 1).detach().numpy(), 1)

                # showimg = cv2.putText(showimg, "update 3d", (int(width/2-20), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255))

                cv2.imshow("results", showimg)
                cv2.waitKey(10)

        #
        real3dnp = virtual_points.detach().numpy()
        pred3dnp = p3d.detach().numpy()
        fig = plt.figure()
        ax3d = fig.add_subplot(111, projection='3d')
        ax3d.scatter(real3dnp[:, 0], real3dnp[:, 1], real3dnp[:, 2], color='r', marker='^')
        ax3d.scatter(pred3dnp[:, 0], pred3dnp[:, 1], pred3dnp[:, 2], color='b', marker='o')

        # ax3d.plot_trisurf(real3dnp[:, 0], real3dnp[:, 1], real3dnp[:, 2])
        # ax3d.plot_trisurf(pred3dnp[:, 0], pred3dnp[:, 1], pred3dnp[:, 2])
        # plt.xlim(0, 640)
        # plt.ylim(0, 480)
        plt.show()

        cv2.waitKey(0)

    # test_update3d()
    # test_updateweights()
    test_updateRT()