import torch
from torch.autograd import Variable

import numpy as np

# Caculate M and Alphas
def PrepareData(pws, us):
    # point count
    N = len(pws)
    assert (len(us) == N)

    # virtual control points, in world frame
    Cw = Variable(torch.from_numpy(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])))

    # get the representation of 3D world points in
    # linear combination of the control points
    nm = 1.0 - (pws[:, 0] + pws[:, 1] + pws[:, 2])
    Alphas = torch.cat((pws, nm.view(-1,1)), 1)

    # get M (2N x 12)
    z = Variable(torch.zeros(N, 1).double())
    M = None
    for i in range(4):
        ta = (Alphas[:, i]).contiguous().view(-1, 1)
        tu = (-us[:, 0] * Alphas[:, i]).contiguous().view(-1, 1)
        tv = (-us[:, 1] * Alphas[:, i]).contiguous().view(-1, 1)
        t0 = torch.cat((ta, z), 1)
        t0 = t0.view(-1, 1)
        t1 = torch.cat((z, ta), 1)
        t1 = t1.view(-1, 1)
        t2 = torch.cat((tu, tv), 1)
        t2 = t2.view(-1, 1)
        if M is None:
            M = torch.cat((t0, t1, t2), 1)
        else:
            M = torch.cat((M, t0, t1, t2), 1)

    # for i in range(N):
    #     M[2*i][0] = Alphas[i][0]
    #     M[2*i][3] = Alphas[i][1]
    #     M[2*i][6] = Alphas[i][2]
    #     M[2*i][9] = Alphas[i][3]
    #     # //
    #     M[2*i][2] = Alphas[i][0] * -us[i][0]
    #     M[2*i][5] = Alphas[i][1] * -us[i][0]
    #     M[2*i][8] = Alphas[i][2] * -us[i][0]
    #     M[2*i][11] = Alphas[i][3] * -us[i][0]
    #
    #     M[2*i+1][1] = Alphas[i][0]
    #     M[2*i+1][4] = Alphas[i][1]
    #     M[2*i+1][7] = Alphas[i][2]
    #     M[2*i+1][10] = Alphas[i][3]
    #     # //
    #     M[2*i+1][2] = Alphas[i][0] * -us[i][1]
    #     M[2*i+1][5] = Alphas[i][1] * -us[i][1]
    #     M[2*i+1][8] = Alphas[i][2] * -us[i][1]
    #     M[2*i+1][11] = Alphas[i][3] * -us[i][1]
    # print(M)

    return Cw, Alphas, M

def GetKernel(M):
    mtm = M.t().mm(M)
    u, s, v = torch.svd(mtm)
    # here s is just the eigenvalus
    # and u and v are exactly the same

    # get the 4 eigenvectors with smallest eigenvalues
    Km = u[:, 8:]
    return Km

def Procrustes(X, Y):
    #
    # input: each line of X or Y is a point
    #
    # will return R, T, r
    # that's find these 3 best matrix to let XR+T = rY,
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

    R = U.mm(V.t())

    d = np.linalg.det(R.data.numpy())
    if d < 0:
        # use this term cause worse accuracy ?
        R = U.mm(torch.diag(torch.DoubleTensor([1,1,-1]))).mm(V.t())

        # use this term cause jitter?
        # R = -R

    # identErr, det = CheckRatation(R)
    # print(identErr)
    # print(det)

    # print(R)

    # check R
    # err = nx*R-ny
    # print("CheckR:")
    # print(err)

    r = S.sum() * (nx_factor / ny_factor)
    t = torch.mm(-ox.view(1, -1), R) + r * oy.view(1, -1)

    # check the result
    err = X.mm(R) + t - r * Y
    # print("Error in Procrustes: %f" % np.linalg.norm(err))
    # print(err)

    return R, t, r

def EPPnP1(K, Pw, uv, norm = 1):
    #normalize the uv
    if norm:
        uv1 = torch.cat((uv, torch.ones(len(uv),1.0).double()), 1)
        uv = torch.inverse(K).mm(uv1.t()).t()
        uv = uv[:,:2]

    Cw, Alpha, M = PrepareData(Pw, uv)
    Km = GetKernel(M)
    # print(Km)

    vK = Km[:, 3]  # take the last column: the one with smallest eigenvalues
    vK = vK.view(4, 3)
    m = vK[:, 2].mean()
    if (m <= 0).data.numpy():  # TODO: check if the numpy mixed Variables differentiable in PyTorch?
        vK = -vK

    return vK.view(1,-1)

def EPPnP(K, Pw, uv, norm = 1):
    #normalize the uv
    if norm:
        uv1 = torch.cat((uv, torch.ones(len(uv),1.0).double()), 1)
        uv = torch.inverse(K).mm(uv1.t()).t()
        uv = uv[:,:2]

    Cw, Alpha, M = PrepareData(Pw, uv)
    Km = GetKernel(M)
    # print(Km)

    vK = Km[:, 3]  # take the last column: the one with smallest eigenvalues
    vK = vK.view(4, 3)
    m = vK[:, 2].mean()
    if (m <= 0).data.numpy():  # TODO: check if the numpy mixed Variables differentiable in PyTorch?
        vK = -vK

    R, T, r = Procrustes(Cw, vK)

    for i in range(10):
        # err = ProjError(Pw, uv, R, T)
        # print("reproj error:")
        # print(err)
        # err = ProjError(Pw, R, T, gtR, gtT)
        # print(err)

        rep = Cw.mm(R) + T
        rep = rep.view(-1, 1)

        # get the linear combination of rep from Km

        # b = torch.mm(Km.t(), rep)
        # A = torch.mm(Km.t(), Km)
        # x, LU = torch.gesv(b, A)

        # closed form of least square
        x = torch.inverse(torch.mm(Km.t(), Km)).mm(Km.t()).mm(rep)

        newY = Km.mm(x)

        # newErr = (rep - newY).norm()
        # newErr.backward()
        # print('reproj error', newErr)

        newY = newY.view(4, 3)

        R, T, r = Procrustes(Cw, newY)

    return R, T

def PPnP(K, Pw, uv, norm = 1):
    #normalize the uv
    if norm:
        uv1 = torch.cat((uv, torch.ones(len(uv),1.0).double()), 1)
        uv = torch.inverse(K).mm(uv1.t()).t()
        uv = uv[:,:2]

    n = len(Pw)

    S = Pw
    P = torch.cat((uv, torch.ones(n,1).double()), 1)

    Z = torch.ones(n, n).double()

    e = torch.ones(n, 1).double()
    A = torch.eye(n).double() - e.mm(e.t())/n
    II = e/n
    E_old = 1e3*torch.ones(n, 3).double()

    iter = 150
    # iter = 1
    for i in range(iter):
        U, _, V = torch.svd(P.t().mm(Z).mm(A).mm(S))

        R = U.mm(V.t())
        d = np.linalg.det(R.data.numpy())
        if d < 0:
            # R = -R
            R = U.mm(torch.diag(torch.DoubleTensor([1, 1, -1]))).mm(V.t())

        PR = P.mm(R)
        c = (S-Z.mm(PR)).t().mm(II)
        Y = S-e.mm(c.t())
        Zmindiag = torch.diag(PR.mm(Y.t())) / torch.sum(P*P, 1)
        Zmindiag[Zmindiag < 0] = 0
        Z = torch.diag(Zmindiag)

        E = Y-Z.mm(PR)
        err = torch.norm(E-E_old)
        E_old = E
        if err < 1e-5:
            break

    T = -R.mm(c)

    return R.t(), T.t()


def quater2mat(Q):
    # ensure unit quaternion
    assert(abs(Q.norm()-1.0) < 1e-10)

    w = Q[0]
    x = Q[1]
    y = Q[2]
    z = Q[3]
    w2 = w.pow(2)
    x2 = x.pow(2)
    y2 = y.pow(2)
    z2 = z.pow(2)
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m0 = w2 + x2 - y2 - z2
    m1 = 2 * (xy - wz)
    m2 = 2 * (wy + xz)
    m3 = 2 * (wz + xy)
    m4 = w2 - x2 + y2 - z2
    m5 = 2 * (yz - wx)
    m6 = 2 * (xz - wy)
    m7 = 2 * (wx + yz)
    m8 = w2 - x2 - y2 + z2

    m = torch.stack((m0,m1,m2,m3,m4,m5,m6,m7,m8))

    return m.view(-1,3)

def mat2quater(M):
    tr = torch.trace(M)
    m = M.contiguous().view(1,-1)[0]
    if tr > 0:
        s = torch.sqrt(tr+1.0) * 2
        w = 0.25 * s
        x = (m[7]-m[5]) / s
        y = (m[2]-m[6]) / s
        z = (m[3]-m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = torch.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7]-m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = torch.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = torch.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = torch.stack((w,x,y,z))
    return Q

def quaterDistance(Q1, Q2):
    # theta = torch.acos(2*torch.pow(Q1.dot(Q2), 2) - 1)
    theta = 1.0 - torch.pow(Q1.dot(Q2), 2) # equals (1-cos_x)/2
    return theta

def rtLoss(K, Pw, r, t, rg, tg, type=1):
    if type == 0:
        q0 = mat2quater(r)
        q1 = mat2quater(rg)
        theta = quaterDistance(q0, q1)
        lt = (tg-t).norm() / tg.norm()
        return theta, lt
        # return lr
    elif type == 1:
        # re-projection error
        ty = Pw.mm(r) + t
        ty = ty.mm(K.t())
        tu = ty[:, 0] / ty[:, 2]
        tv = ty[:, 1] / ty[:, 2]

        ty = Pw.mm(rg) + tg
        ty = ty.mm(K.t())
        gu = ty[:, 0] / ty[:, 2]
        gv = ty[:, 1] / ty[:, 2]

        err = torch.stack((tu-gu,tv-gv), 1)
        nm = err.norm(2, 1)
        return nm.mean()

    elif type == 2:
        # return (r - rg).norm() + (t - tg).norm()
        return (r-rg).pow(2).sum() + (t-tg).pow(2).sum()
