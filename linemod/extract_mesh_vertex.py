import os
import random
import trimesh
import cv2
import numpy as np
from linemod_aux import *

def occlinemod_collect_mesh_bbox(meshpath, objectlist, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # recursive
    allv = []

    for objname in objectlist:
        mppath = meshpath + objname.capitalize()
        mp = [f for f in os.listdir(mppath) if f.endswith('.obj')][0]
        mesh = trimesh.load(mppath + '/' + mp)
        bbox = mesh.bounding_box_oriented.vertices
        allv.append(bbox)
    outName = outdir + 'LINEMOD_bbox.npy'
    np.save(outName, allv)

def occlinemod_print_mesh_diameter(meshpath, objectlist):
    for objname in objectlist:
        mppath = meshpath + objname.capitalize()
        mp = [f for f in os.listdir(mppath) if f.endswith('.obj')][0]
        mesh = trimesh.load(mppath + '/' + mp)
        v = mesh.bounding_sphere.volume
        r = np.power(3*v/(4*np.pi), 1/3)
        print(2*r)

def occlinemod_collect_mesh_vertex(meshpath, objectlist, outdir, vNum):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # recursive
    allv = []

    for objname in objectlist:
        mppath = meshpath + objname.capitalize()
        mp = [f for f in os.listdir(mppath) if f.endswith('.obj')][0]
        mesh = trimesh.load(mppath + '/' + mp)

        # make it sparse (random choose)
        vertices, faceidx = trimesh.sample.sample_surface(mesh, vNum)
        # vertices, faceidx = trimesh.sample.sample_surface_even(mesh, vNum)
        # outName = outdir + objname + '.txt'
        # np.savetxt(outName, vertices)

        np.random.shuffle(vertices)
        allv.append(vertices)
    outName = outdir + 'LINEMOD_vertex.npy'
    np.save(outName, allv)

def linemod_collect_mesh_bbox(meshpath, objectlist, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # recursive
    allv = []

    for objname in objectlist:
        mppath = meshpath + objname + '.ply'
        mesh = trimesh.load(mppath)
        bbox = mesh.bounding_box_oriented.vertices
        allv.append(bbox)
    outName = outdir + 'LINEMOD_bbox.npy'
    np.save(outName, allv)

def linemod_print_mesh_diameter(meshpath, objectlist):
    for objname in objectlist:
        mppath = meshpath + objname + '.ply'
        mesh = trimesh.load(mppath)
        v = mesh.bounding_sphere.volume
        r = np.power(3*v/(4*np.pi), 1/3)
        print(2*r)

def linemod_collect_mesh_vertex(meshpath, objectlist, outdir, vNum):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # recursive
    allv = []

    for objname in objectlist:
        mppath = meshpath + objname + '.ply'
        mesh = trimesh.load(mppath)
        # make it sparse (random choose)
        vertices, faceidx = trimesh.sample.sample_surface(mesh, vNum)
        # vertices, faceidx = trimesh.sample.sample_surface_even(mesh, vNum)
        # outName = outdir + objname + '.txt'
        # np.savetxt(outName, vertices)

        np.random.shuffle(vertices)
        allv.append(vertices)

    outName = outdir + 'LINEMOD_vertex.npy'
    np.save(outName, allv)

def mesh_transform_LM_to_OCC(linemod_path, occlinemod_path, objectlist, outdir):
    '''
    TODO: the transform is wrong !
    :param linemod_path:
    :param occlinemod_path:
    :param objectlist:
    :param outdir:
    :return:
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)  # recursive
    all_transforms = []

    for objname in objectlist:
        mppath = occlinemod_path + objname.capitalize()
        mp = [f for f in os.listdir(mppath) if f.endswith('.obj')][0]
        occ_mesh = trimesh.load(mppath + '/' + mp)

        mppath = linemod_path + objname + '.ply'
        lm_mesh = trimesh.load(mppath)

        transform, cost = trimesh.registration.mesh_other(lm_mesh, occ_mesh)
        all_transforms.append(transform)

        print(objname, cost)
    outName = outdir + 'To_OccLINEMOD.npy'
    np.save(outName, all_transforms)

diameter_occlinemod = [0.104259213249,0.204827320837,0.15462857733,0.264124227464,0.110829372822,0.164649609483,0.178355374558,0.163178609322]
def generate_uniform_bbox(diameters, out_name):
    bbox = np.array([1,1,1, 1,1,-1, 1,-1,1, 1,-1,-1, -1,1,1, -1,1,-1, -1,-1,1, -1,-1,-1])
    uniform_r = diameters.mean() / (2.0*np.sqrt(3.0))
    objCnt = len(diameters)
    uniform_bbox = (bbox*uniform_r).reshape(1,-1).repeat(objCnt, axis=0).reshape(objCnt,-1,3)
    np.save(out_name, uniform_bbox)

if __name__ == "__main__":
    occ_outdir = '/data/OcclusionChallengeICCV2015/models_vertex/'
    occ_meshpath = '/data/OcclusionChallengeICCV2015/models/'
    # occlinemod_collect_mesh_vertex(meshpath, target_objects, outdir, 10000)
    # occlinemod_collect_mesh_bbox(occ_meshpath, target_objects, occ_outdir)
    # occlinemod_print_mesh_diameter(meshpath, target_objects)

    print('-------------------')
    lm_outdir = '/data/LINEMOD/models_vertex/'
    lm_meshpath = '/data/LINEMOD/models/'
    # linemod_collect_mesh_vertex(meshpath, target_objects, outdir, 10000)
    # linemod_collect_mesh_bbox(lm_meshpath, target_objects, lm_outdir)
    # linemod_print_mesh_diameter(meshpath, target_objects)
    generate_uniform_bbox(np.array(diameter_occlinemod), lm_outdir + 'LINEMOD_bbox_u.npy')
    # generate_uniform_axis(np.array(diameter_occlinemod), lm_outdir + 'LINEMOD_axis4.npy')
    # generate_uniform_axis6(np.array(diameter_occlinemod), lm_outdir + 'LINEMOD_axis6.npy')

    outdir = '/data/LINEMOD/models_vertex/'
    # mesh_transform_LM_to_OCC(lm_meshpath, occ_meshpath, target_objects, outdir)
