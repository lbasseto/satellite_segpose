import os
import numpy as np
import trimesh
import cv2

# 8 objects for LINEMOD-Occlusion dataset
target_objects = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']

# intrinsics of LINEMOD dataset
k_linemod = np.array([[572.41140, 0.0, 325.26110],
                 [0.0, 573.57043, 242.04899],
                 [0.0, 0.0, 1.0]])

def get_linemod_obj_id(objName):
    if objName not in target_objects:
        return -1
    else:
        return target_objects.index(objName)

def write_linemod_objects_name(filename):
    with open(filename, 'w') as f:
        for obj in target_objects:
            f.write(obj + '\n')

def filename_to_objectname(filename):
    for objname in target_objects:
        if objname in filename:
            return objname
    return None

def read_dataset_filelist(listfile):
    with open(listfile, 'r') as f:
        flist = f.read().splitlines()
    fileset = {}
    for fname in flist:
        objname = filename_to_objectname(fname)
        if objname in fileset:
            fileset[objname].append(fname)
        else:
            fileset[objname] = [fname]
    return fileset

def read_gt_matrix(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    row, col = lines[0].split()
    rot = ' '.join(lines[1:])
    rot = np.fromstring(rot, sep=' ')
    rot = rot.reshape(int(row), int(col))
    return rot

def vertices_reprojection(vertices, rt, k):
    p = np.matmul(k, np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def vertices_3d_transform(vertices, rt):
    p = np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1)
    return p.T

def draw_3d_meshes(meshfile, rt, img, k, color = [0,255,0]):
    mesh = trimesh.load(meshfile)
    # make it sparse
    vertices, faceidx = trimesh.sample.sample_surface(mesh, 1000)
    pts = vertices_reprojection(vertices, rt, k)
    for p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 1, color, -1)
    return img

def draw_axis_1(img, k, rt, scale = 1, linewidth=2, xcolor = [0, 0, 255], ycolor= [0, 255, 0], zcolor = [255, 0, 0]):
    # X Y Z corresponding to R G B
    k = k.reshape(3, 3)
    rt = rt.reshape(3, 4)
    anchors = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]])

    p = np.matmul(k, np.matmul(rt[:3, 0:3], anchors.T) + rt[:3, 3].reshape(-1, 1))
    x = p[0]/p[2]
    y = p[1]/p[2]

    # origin
    # img = cv2.circle(img, (x[0], y[0]), 5, (0, 0, 255), -1)

    img = cv2.line(img, (int(x[0]+0.5), int(y[0]+0.5)),
                   (int(x[1]+0.5), int(y[1]+0.5)), xcolor, linewidth, cv2.LINE_AA)
    img = cv2.line(img, (int(x[0]+0.5), int(y[0]+0.5)),
                   (int(x[2]+0.5), int(y[2]+0.5)), ycolor, linewidth, cv2.LINE_AA)
    img = cv2.line(img, (int(x[0]+0.5), int(y[0]+0.5)),
                   (int(x[3]+0.5), int(y[3]+0.5)), zcolor, linewidth, cv2.LINE_AA)

    return img

def read_label_colors(labelcolorfile = '/home/yhu/data/LINEMOD/label_colors.txt'):
    with open(labelcolorfile, 'r') as f:
        lines = f.read().splitlines()
    labelcolors = {}
    for l in lines:
        data = l.split()
        labelcolors[data[3]] = [int(data[0]), int(data[1]), int(data[2])]
    return labelcolors
