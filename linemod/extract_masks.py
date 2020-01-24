import os
import random
from linemod_aux import *
import cv2
import numpy as np

def extract_masks(listfile, modelpath, outpath):
    fileset = read_dataset_filelist(listfile)
    objvertices = {}
    for objname in fileset:
        suboutpath = outpath + '/' + objname
        if not os.path.exists(suboutpath):
            os.makedirs(suboutpath)  # recursive
        for imgpath in fileset[objname]:
            baseName, extName = os.path.splitext(imgpath[imgpath.rfind('/') + 1:])
            posepath = imgpath.replace('.jpg', '.txt')
            rt = np.loadtxt(posepath)
            if objname not in objvertices:
                meshpath = modelpath + '/' + objname + '.ply'
                mesh = trimesh.load(meshpath)
                # make the mesh denser
                vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, 1)
                objvertices[objname] = vertices

            pts = vertices_reprojection(objvertices[objname], rt)
            img = cv2.imread(imgpath)
            msk = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) # for alpha channel
            for p in pts:
                x = int(p[0] + 0.5)
                y = int(p[1] + 0.5)
                if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
                    msk[y][x] = 255
            msk3 = np.repeat(msk, 3).reshape(img.shape)
            img[msk3 < 128] = 0
            foregroundimg = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            foregroundimg[:,:,3] = msk # assign the mask to the last channel of the image

            maskname = suboutpath + '/' + baseName + '.png'
            cv2.imwrite(maskname, foregroundimg)
            # cv2.imwrite(maskname, foregroundimg, [cv2.IMWRITE_PNG_COMPRESSION, 9,
            #                                       cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY])
            print(maskname)

if __name__ == "__main__":
    listfile = '/home/yhu/workspace/data/LINEMOD/all.txt'
    modelpath = '/home/yhu/workspace/data/LINEMOD/models/'
    outpath = '/home/yhu/workspace/data/LINEMOD/mask'
    extract_masks(listfile, modelpath, outpath)
