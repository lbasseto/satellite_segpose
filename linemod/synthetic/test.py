import numpy as np
import cv2

def ReadMatrix(matFile):
    with open(matFile, 'r') as f:
        lines = f.read().splitlines()
    pstr = ''
    for l in lines:
        pstr += l
        pstr += ' '
    mat = np.array(pstr.split(), dtype='float64')
    return mat

def draw_2d_proj(img, pw, K, R, T, color):
    p = np.matmul(pw, R) + T
    p = np.matmul(p, K.transpose()).tolist()
    for i in range(len(p)):
        tx = p[i][0] / p[i][2]
        ty = p[i][1] / p[i][2]
        img = cv2.circle(img, (int(tx), int(ty)), 1, color, -1)
    return img

# pose = ReadMatrix('pose.txt').reshape(4, 4)
# K = ReadMatrix('K.txt').reshape(3, 3)
# img = cv2.imread('out.png')
pose = ReadMatrix('/home/yhu/data/LINEMOD/training/benchvise/00001.txt').reshape(4, 4)
K = ReadMatrix('/home/yhu/data/LINEMOD/rendering/benchvise/K.txt').reshape(3, 3)
img = cv2.imread('/home/yhu/data/LINEMOD/training/benchvise/00001.png')
R = pose[:3, :3].transpose() # transpose R as XR+T
T = pose[:3, 3]

pw = np.load('./vertex.npy')[1]
p = np.matmul(pw, R) + T
p = np.matmul(p, K.transpose()).tolist()
uv = []
for i in range(len(p)):
    tx = p[i][0] / p[i][2]
    ty = p[i][1] / p[i][2]
    uv.append([tx,ty])
uv = np.array(uv)

retval, rot, trans = cv2.solvePnP(pw, uv, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
rot = cv2.Rodrigues(rot)[0] # convert to rotation matrix
trans = trans.reshape(-1)

img = draw_2d_proj(img, pw, K, R, T, (0,255,0))
# img = draw_2d_proj(img, pw, K, rot.T, trans, (0,255,0))

cv2.imshow("img", img)
cv2.waitKey(0)

