# from plyfile import PlyData, PlyElement
import numpy as np
import cv2

import PnP
import torch
from torch.autograd import Variable

# draft code

# intrinsics
K=[[572.4114, 0.0, 325.2611],
 [0.0, 573.5704, 242.0489],
 [0.0, 0.0, 1.0]]
K = np.matrix(K, dtype='float64')

objects = ['ape', 'benchvise', 'bowl', 'cam', 'can',
           'cat', 'cup', 'driller', 'duck', 'eggbox',
           'glue', 'holepuncher', 'iron', 'lamp', 'phone']

def GenWorldPoints(num):
    ptList = [[0,0,0]]
    for i in range(num-1):
        pt = np.random.uniform(-0.1, 0.1, 3)
        ptList.append(pt.tolist())
    return np.array(ptList)

# plydata = PlyData.read("/home/yinlin/data/LINEMOD/models/ape.ply")
# vertexList = plydata.elements[0].data

pws = GenWorldPoints(100)

for obj in objects:
    pf = open('/home/yinlin/data/LINEMOD/objects/' + obj + '/pose/0001.txt', 'r')
    line = pf.readline()
    pstr = ''
    while line:
        pstr += line
        line = pf.readline()
    pf.close()
    RT = np.array(pstr.split(), dtype='float64').reshape((4, 4))
    R = RT[:3, :3]
    T = RT[:3, 3]

    cv2.destroyAllWindows()
    mask = cv2.imread('/home/yinlin/data/LINEMOD/objects/' + obj + '/mask/0001.png')
    img = cv2.imread('/home/yinlin/data/LINEMOD/objects/' + obj + '/rgb/0001.jpg')
    img[mask < 128] = 255

    xy = []
    for pt in pws:
        p = np.matmul(np.array(pt), R) + T
        print(p)
        p = np.matmul(p, K.transpose()).tolist()[0]

        x=p[0]/p[2]
        y=p[1]/p[2]
        xy.append([x,y])
        # print(x)
        # print(y)

        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)

    xy = Variable(torch.from_numpy(np.array(xy)))
    xy += 50
    xy *= 1.2
    er, et = PnP.EPPnP(Variable(torch.from_numpy(K)), Variable(torch.from_numpy(pws)), xy)
    er = er.data.numpy()
    et = et.data.numpy()[0]

    for pt in pws:
        p = np.matmul(np.array(pt), er) + et
        print(p)
        p = np.matmul(p, K.transpose()).tolist()[0]

        x=p[0]/p[2]
        y=p[1]/p[2]
        # xy.append([x,y])
        # print(x)
        # print(y)
        cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)

    cv2.imshow(obj, img)
    cv2.waitKey(0)
