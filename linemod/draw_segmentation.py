import os
import random
import numpy as np
import cv2
from linemod_aux import *
from utils import get_class_colors

def show_segmentation(annotationdir):
    annotlist = [f for f in os.listdir(annotationdir) if f.endswith('.npz')]
    random.shuffle(annotlist)
    for item in annotlist:
        annofile = annotationdir + item
        imgfile = annofile.replace('.npz', '.png')

        annot = np.load(annofile)
        seg = annot['segmentation']
        poses = annot['poses']
        objIds = annot['objectsID']
        objCnt = len(objIds)

        img = cv2.imread(imgfile)
        cv2.imshow("img", img)
        # cv2.waitKey(0)

        meshimg = np.copy(img)
        segimg = np.copy(img)
        segimg[seg == 0] = get_class_colors(139) # random choose for background
        for i in range(objCnt):
            id = objIds[i]
            color = get_class_colors(id)
            segimg[seg == (i+1)] = color

            meshfile = '/home/yhu/workspace/data/LINEMOD/models/' + target_objects[id] + '.ply'
            meshimg = draw_3d_meshes(meshfile, poses[i], meshimg, k_linemod, color)
            meshimg = draw_axis_1(meshimg, k_linemod, poses[i], 0.1)

        cv2.imshow("seg", segimg)
        cv2.imshow("mesh", meshimg)

        cv2.waitKey(0)


if __name__ == "__main__":
    annotationdir = '/home/yhu/workspace/data/LINEMOD/synthetic_train/'
    show_segmentation(annotationdir)
