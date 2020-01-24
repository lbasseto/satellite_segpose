import os
import random
from linemod_aux import *
import cv2
import numpy as np
from sklearn.cluster import KMeans

def collect_z_distributions(listfile):
    fileset = read_dataset_filelist(listfile)
    for objname in fileset:
        minZ = 1e10
        maxZ = 0
        averZ = 0
        for imgpath in fileset[objname]:
            posepath = imgpath.replace('.jpg', '.txt').replace('.png', '.txt').replace('rgb', 'pose')
            rt = np.loadtxt(posepath)
            z = rt[2][3]
            if z < minZ:
                minZ = z
            if z > maxZ:
                maxZ = z
            averZ += z
        averZ /= len(fileset[objname])
        # print(objname + ' :')
        print(minZ, maxZ, averZ)

def collet_anchors(listfile, modelVertexPath):
    fileset = read_dataset_filelist(listfile)
    whs = []
    for objname in fileset:
        vertex = np.loadtxt(modelVertexPath + objname + '.txt')
        for imgpath in fileset[objname]:
            img = cv2.imread(imgpath)
            imgwidth = img.shape[1]
            imgheight = img.shape[0]
            posepath = imgpath.replace('.jpg', '.txt').replace('.png', '.txt')
            rt = np.loadtxt(posepath)

            # op = vertices_reprojection(np.array([[0, 0, 0]]), rt, k_linemod)[0]
            vp = vertices_reprojection(vertex, rt, k_linemod)
            w = (vp[:, 0].max() - vp[:, 0].min()) / imgwidth
            h = (vp[:, 1].max() - vp[:, 1].min()) / imgheight
            print(w, h)
            whs.append([w, h])

    nClusters = 5
    kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(np.array(whs))
    # print(kmeans.labels_)
    for i in range(nClusters):
        w = kmeans.cluster_centers_[i][0]
        h = kmeans.cluster_centers_[i][1]
        print('%.3f, %.3f, %.3f' % (w, h, w*h))

if __name__ == "__main__":
    listfile = '/data/LINEMOD/all.txt'
    collect_z_distributions(listfile)
    # modelVertexPath = '/home/yhu/workspace/data/LINEMOD/model_vertex/'
    # collet_anchors(listfile, modelVertexPath)
