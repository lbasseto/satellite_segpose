import os
import numpy as np
import trimesh
import cv2

def load_transform(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    lines = lines[1:] # get rid of the first line
    str = ''
    for l in lines:
        data = l.split()
        str += data[1] # only fetch the second column
        str += ' '
    rt = np.fromstring(str, sep=' ')
    rt = rt.reshape(3, 4)
    rt[:,3] *= 1000 # translation metric from m to mm
    return rt

def refind_old_mesh(oldmeshfile, tranformfile, outmeshfile):
    mesh = trimesh.load(oldmeshfile)
    transform = load_transform(tranformfile)
    transform = np.concatenate((transform, np.array([[0,0,0,1]])), 0) # make it to 4x4
    mesh.apply_transform(transform)
    export_option = {}
    export_option['file_obj'] = outmeshfile
    # export_option['vertex_normal'] = False
    # export_option['encoding'] = 'ascii'
    mesh.export(**export_option)

if __name__ == "__main__":
    oldmeshfile = '/home/yhu/data/LINEMOD/raw/cam/OLDmesh.ply'
    tranformfile = '/home/yhu/data/LINEMOD/raw/cam/transform.dat'
    outmeshfile = '/home/yhu/data/LINEMOD/raw/cam/mesh.ply'
    refind_old_mesh(oldmeshfile, tranformfile, outmeshfile)

    oldmeshfile = '/home/yhu/data/LINEMOD/raw/eggbox/OLDmesh.ply'
    tranformfile = '/home/yhu/data/LINEMOD/raw/eggbox/transform.dat'
    outmeshfile = '/home/yhu/data/LINEMOD/raw/eggbox/mesh.ply'
    refind_old_mesh(oldmeshfile, tranformfile, outmeshfile)

