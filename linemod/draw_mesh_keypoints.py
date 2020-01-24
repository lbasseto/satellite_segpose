import os
import random
from linemod_aux import *
import trimesh
import cv2
import numpy as np

def collect_mesh_vertex(meshpath, outdir, vNum):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # recursive
    # for objname in objects:
    # mp = meshpath + objname + '.ply'
    mp = meshpath + 'textured.obj'
    mesh = trimesh.load(mp)
    # make it sparse (random choose)
    vertices, faceidx = trimesh.sample.sample_surface(mesh, vNum)
    # outName = outdir + objname + '.txt'
    # np.savetxt(outName, vertices)
    # write new color for the keypoints
    for idx in faceidx:
        mesh.visual.face_colors[idx] = np.array([0,255,0,255])
    # save new mesh
    export_option = {}
    # export_option['file_obj'] = outdir + objname + '.ply'
    export_option['file_obj'] = outdir + 'out.ply'
    # export_option['vertex_normal'] = False
    # export_option['encoding'] = 'ascii'
    mesh.export(**export_option)

if __name__ == "__main__":
    outdir = './out/'
    meshpath = '/home/yhu/workspace/data/YCB_Video_Dataset/models/003_cracker_box/'
    collect_mesh_vertex(meshpath, outdir, 1000)
