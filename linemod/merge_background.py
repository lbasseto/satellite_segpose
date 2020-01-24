import os
import numpy as np
import trimesh
import cv2
import math
import shutil
from linemod.linemod_aux import *
import random
from itertools import combinations

def recompute_pose(vertex, rt, k, transform2d):
    # vCnt = min(1000, len(vertex))
    vCnt = len(vertex)
    v3dPt = vertex[:vCnt]

    # compute 2d reprojection
    uvz = np.matmul(k, np.matmul(rt[:3, 0:3], v3dPt.T) + rt[:3, 3].reshape(-1, 1))
    uvz[0] = uvz[0] / uvz[2]
    uvz[1] = uvz[1] / uvz[2]
    uvz[2] = 1.0 # get homography coordinates

    # adjust the 2D points according to transformation
    uv = (np.matmul(transform2d, uvz).T)[:, :2]

    # recompute the pose (use RANSAC EPNP in OpenCV)
    # retval, rot, trans = cv2.solvePnP(vertices.reshape(ptCnt, 1, -1), uv.reshape(ptCnt, 1, -1), k,
    #                                   None, None, None, False, cv2.SOLVEPNP_EPNP)
    # retval, rot, trans = cv2.solvePnP(v3dPt, uv, k, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
    retval, rot, trans, inliers = cv2.solvePnPRansac(v3dPt, uv, k, None, flags=cv2.SOLVEPNP_EPNP)
    assert (retval == True)
    R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
    T = trans.reshape(-1, 1)

    return np.concatenate((R, T), 1)


def generate_one_sample(vertex, targetK, k, width, height, x, y, z, angle, foreImg, forePose):
    rt = forePose

    # change image to target intrinsics
    img = foreImg
    Mi = np.matmul(targetK, np.linalg.inv(k))
    img = cv2.warpAffine(img, Mi[:2], (width, height))

    # recompute the pose according to new xyz
    p = vertices_reprojection(np.array([[0, 0, 0]]), rt, targetK)[0]
    cx = p[0]
    cy = p[1]
    dx = x - cx
    dy = y - cy
    scale = rt[2][3] / z

    m1 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])  # translation
    rs = cv2.getRotationMatrix2D((cx, cy), angle, scale)  # rotation and scale
    m2 = np.concatenate((rs, [[0, 0, 1]]), axis=0)
    M = np.matmul(m1, m2)
    newrt = recompute_pose(vertex, rt, targetK, M)

    # change image accordingly
    outImg = cv2.warpAffine(img, M[:2], (width, height))

    return outImg, newrt

def resolve_occlusion(imglist, max_occ_ratio = 0.5):
    itemCnt = len(imglist)
    selected = [0] * itemCnt # the flag if current item is choose
    pixelCnt = [0] * itemCnt # raw pixel count of each item
    occlusionMap = None
    for i in range(itemCnt):
        mi = np.copy(imglist[i][:,:,3]) # get alpha channel
        mi[mi < 128] = 0
        mi[mi >= 128] = i + 1
        pixelCnt[i] = int(mi.sum() / (i + 1))
        if occlusionMap is None:
            occlusionMap = np.copy(mi)
        else:
            occlusionMap[mi > 0] = mi[mi > 0]

    # tmpMap = np.copy(occlusionMap)
    # cv2.normalize(occlusionMap, tmpMap, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow("occ", tmpMap)
    # cv2.waitKey(0)

    # check occlusions
    for i in range(itemCnt):
        tMap = np.copy(occlusionMap)
        tMap[tMap != i+1] = 0
        tMap[tMap == i+1] = 1

        visCnt = tMap.sum()
        occRatio = 1.0 - visCnt/pixelCnt[i]
        # print(occRatio, visCnt, pixelCnt[i])

        if occRatio > max_occ_ratio:
            selected[i] = 0
        else:
            selected[i] = 1

        # cv2.normalize(tMap, tMap, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("occ", tMap)
        # cv2.waitKey(0)

    # recompute the segmentation
    segMap = None
    segIdx = 1
    for i in range(itemCnt):
        if selected[i]:
            mi = np.copy(imglist[i][:, :, 3])  # get alpha channel
            mi[mi < 128] = 0
            mi[mi >= 128] = segIdx
            if segMap is None:
                segMap = np.copy(mi)
            else:
                segMap[mi > 0] = mi[mi > 0]
            segIdx += 1

    return selected, segMap

def generate_synthetic_images(vertex, targetK, renderingK, renderinglist, foreimglist, outpath, syntheticratio, poseCnt):
    '''
    :param syntheticratio: the percentage of foreground using rendering set
    :return:
    '''
    width = 640
    height = 480
    margin = 0.2
    minx = int(0 + width * margin)
    maxx = int(width - width * margin) # in pixel
    miny = int(0 + height * margin)
    maxy = int(height - height * margin) # in pixel
    minz = 0.6 # in meter, from the statistics of raw dataset
    maxz = 1.1 # in meter
    minangle_syn = -70 # for synthetic samples
    maxangle_syn = 70
    minangle_real = -20 # for real samples
    maxangle_real = 20
    minobjcnt = 3
    maxobjcnt = 8
    objectindex = [0,1,2,3,4,5,6,7]
    if not os.path.exists(outpath):
        os.makedirs(outpath)  # recursive

    realmaskset = read_dataset_filelist(foreimglist)
    renderingset = None
    k_rendering = None
    if syntheticratio > 0:
        k_rendering = np.loadtxt(renderingK)
        renderingset = read_dataset_filelist(renderinglist)

    outIdx = 0
    while outIdx < poseCnt:
        backimg = 0
        #
        foreimgset = {}

        candiobjcnt = random.randint(minobjcnt, maxobjcnt)
        combStore = list(combinations(objectindex, candiobjcnt))
        candiObjIndex = combStore[random.randint(0, len(combStore) - 1)]

        while True:
            foreimgset['image'] = []
            foreimgset['pose'] = []
            foreimgset['object'] = []
            for iobj in range(candiobjcnt):
                objID = candiObjIndex[iobj]
                objName = target_objects[objID]

                # random position
                x = random.randint(minx, maxx)
                y = random.randint(miny, maxy)

                # random choose from synthetic or real set
                dT = random.uniform(0, 1)
                chooseFromReal = True
                angle = 0
                if dT >= syntheticratio:
                    k = targetK
                    foregroundset = realmaskset
                    angle = random.randint(minangle_real, maxangle_real)
                    chooseFromReal = True
                else:
                    k = k_rendering
                    foregroundset = renderingset
                    # random choose angle
                    angle = random.randint(minangle_syn, maxangle_syn)
                    chooseFromReal = False

                # random choose depth
                z = random.uniform(minz, maxz)
                # random choose pose
                candiCnt = len(foregroundset[objName])
                idx = random.randint(0, candiCnt - 1)
                if chooseFromReal:
                    # extract the foreground image from mask
                    imgname = foregroundset[objName][idx]
                    maskname = imgname.replace('rgb', 'mask').replace('.jpg', '.png')
                    rawimg = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
                    maskimg = cv2.imread(maskname, cv2.IMREAD_UNCHANGED)
                    rawimg = (rawimg * (maskimg / 255)).astype(np.uint8)
                    foreImg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2BGRA)
                    foreImg[:, :, 3] = maskimg.mean(axis=2).astype(np.uint8)
                    posename = imgname.replace('rgb','pose').replace('.jpg', '.txt')
                else:
                    imgname = foregroundset[objName][idx]
                    foreImg = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
                    posename = imgname.replace('.png', '.txt')
                forePose = np.loadtxt(posename)[:3, :]
                #
                img, rt = generate_one_sample(vertex[objID], targetK, k, width, height, x, y, z, angle, foreImg, forePose)

                foreimgset['image'].append(img)
                foreimgset['pose'].append(rt)
                foreimgset['object'].append(objName)

                # rgb = img[:,:,:3]
                # alpha = img[:,:,3]
                # backimg[alpha>128] = rgb[alpha>128]
                # meshfile = '/data/LINEMOD/models/' + obj + '.ply'
                # backimg += draw_3d_meshes(meshfile, rt, img, k_linemod)
                # backimg += draw_3d_meshes(meshfile, forePose, foreImg, k)

                # cv2.imshow("img", backimg)
                # cv2.waitKey(0)

            selected, segmap = resolve_occlusion(foreimgset['image'], 0.7)
            if np.array(selected).sum() == candiobjcnt: # desired object count
                break

        # cv2.normalize(segmap, segmap, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("occ", segmap)
        # cv2.waitKey(0)
        # collect selected items
        selectobjects = []
        selectposes = []
        mergedimg = np.zeros((height, width, 4), np.uint8)
        for i in range(len(selected)):
            if selected[i]:
                forergb = foreimgset['image'][i][:, :, :3]
                alpha = foreimgset['image'][i][:, :, 3] / 255.0
                alpha = np.repeat(alpha, 3).reshape(height, width, 3)
                mergedimg[:, :, :3] = np.uint8(mergedimg[:,:,:3] * (1 - alpha) + forergb * alpha)
                mergedimg[:, :, 3] = np.maximum(mergedimg[:,:,3], foreimgset['image'][i][:, :, 3])
                # mergedimg[alpha>128] = foreimgset['image'][i][alpha>128]
                # backimg[alpha > 128] = rgb[alpha > 128]
                selectobjects.append(get_linemod_obj_id(foreimgset['object'][i]))
                selectposes.append(foreimgset['pose'][i])
        # cv2.imshow("img", backimg)
        # cv2.waitKey(0)
        # write
        assert(len(selectobjects) > 0)
        outName = ("%s/%06d" % (outpath, outIdx))
        cv2.imwrite(outName + '.png', mergedimg)
        np.savez_compressed(outName + '.npz',
                            segmentation = segmap,
                            poses = selectposes,
                            objectsID = selectobjects,
                            intrinsics = targetK)
        outIdx += 1
        print(outName)

if __name__ == "__main__":
    #############################
    renderinglist = '/data/LINEMOD/all_render.txt'
    renderingK = '/data/LINEMOD/render/K.txt'
    targetK = k_linemod
    vertex = np.load('/data/LINEMOD/models_vertex/LINEMOD_vertex.npy')
    # backgroundpath = '/home/yhu/workspace/data/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/JPEGImages/'
    foreimglist = '/data/LINEMOD/train.txt'
    outpath = '/data/LINEMOD/synthetic_train/'

    # backimgs = [f for f in os.listdir(backgroundpath) if f.endswith('.jpg') or f.endswith('.png')]
    # backimgs.sort()
    # for i in range(len(backimgs)):
    #     backimgs[i] = backgroundpath + backimgs[i]

    # write_linemod_objects_name('/home/yhu/workspace/data/LINEMOD/linemod.names')

    generate_synthetic_images(vertex, targetK, renderingK, renderinglist, foreimglist, outpath, 0, 20000)

    foreimglist = '/data/LINEMOD/valid.txt'
    outpath = '/data/LINEMOD/synthetic_valid/'
    generate_synthetic_images(vertex, targetK, renderingK, renderinglist, foreimglist, outpath, 0, 2000)
