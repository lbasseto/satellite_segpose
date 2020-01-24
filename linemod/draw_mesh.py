import os
import random
from linemod_aux import *
import cv2
import numpy as np

def draw_mesh(listfile, meshpath, desiredObjName):
    fileset = read_dataset_filelist(listfile)
    for objname in fileset:
        if objname != desiredObjName:
            continue
        for imgpath in fileset[objname]:
            posepath = imgpath.replace('.jpg', '.txt').replace('.png', '.txt').replace('/rgb', '/mask')
            rt = np.loadtxt(posepath)
            img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
            img = draw_3d_meshes(meshpath + '/' + objname + '.ply', rt, img)
            cv2.imshow("mesh", img)
            cv2.waitKey(0)

if __name__ == "__main__":
    listfile = '/home/yhu/workspace/data/LINEMOD/train_mask.txt'
    # listfile = '/home/yhu/workspace/data/LINEMOD/rendering.txt'
    meshpath = '/home/yhu/workspace/data/LINEMOD/models/'
    draw_mesh(listfile, meshpath, 'eggbox')
