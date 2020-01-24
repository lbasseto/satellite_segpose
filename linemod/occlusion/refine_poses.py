import os
import numpy as np
import trimesh
import cv2
import math
import shutil
from linemod_aux import *

def update_pose_according_to_new_mesh(flist, outdir, meshTrans):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # recursive
    for imgpath in flist:
        prepath = imgpath[:imgpath.rfind('/') + 1]
        baseName, extName = os.path.splitext(imgpath[imgpath.rfind('/') + 1:])
        rotpath = prepath + baseName.replace('color', 'rot') + '.rot'
        trapath = prepath + baseName.replace('color', 'tra') + '.tra'
        rot = read_gt_matrix(rotpath)
        tra = read_gt_matrix(trapath) * 10  # the raw translation is in cm, change to mm

        # copy the images to new place
        shutil.copyfile(imgpath, outdir + '/' + baseName + extName)

        # recompute the pose and copy them to new place
        newrot = np.matmul(rot, meshTrans[0:3, 0:3].T)
        newtra = tra - np.matmul(newrot, meshTrans[0:3, 3])
        outpose = np.concatenate((newrot, newtra.T), 1)
        outpose = np.concatenate((outpose, np.array([[0, 0, 0, 1]])), 0)  # make it to 4x4
        outposefilename = outdir + '/' + baseName + '.txt'
        np.savetxt(outposefilename, outpose)
        print(outposefilename)

def refine_poses(linemod_path, outmodelpath, outimgpath, listfile):
    if not os.path.exists(outmodelpath):
        os.makedirs(outmodelpath)  # recursive
    fileset = read_dataset_filelist(listfile)
    for objname in objects:
        if objname == 'bowl' or objname == 'cup':  # don't consider these objects (symmetric)
            continue
        meshfile = linemod_path + '/' + objname + '/mesh.ply'
        outmeshfile = outmodelpath + '/' + objname + '.ply'
        mesh = trimesh.load(meshfile)
        # rotate the raw mesh
        qx = trimesh.transformations.quaternion_about_axis(math.pi, [1,0,0])
        R = trimesh.transformations.quaternion_matrix(qx)
        T = mesh.bounding_box.centroid # make the mesh center to origin
        meshTrans = R
        meshTrans[0:3, 3] = T # merge RT to 4x4
        mesh.apply_transform(meshTrans)
        export_option = {}
        export_option['file_obj'] = outmeshfile
        # export_option['vertex_normal'] = False
        # export_option['encoding'] = 'ascii'
        mesh.export(**export_option)
        print(outmeshfile)

        # refine the raw pose
        update_pose_according_to_new_mesh(fileset[objname], outimgpath + objname, meshTrans)


if __name__ == "__main__":
    linemod_path = '/home/yhu/data/LINEMOD/raw/'
    outmodelpath = '/home/yhu/data/LINEMOD/models/'
    outimgpath = '/home/yhu/data/LINEMOD/rgb/'
    listfile = '/home/yhu/data/LINEMOD/raw_all.txt'
    refine_poses(linemod_path, outmodelpath, outimgpath, listfile)
