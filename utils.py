import sys
import os
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from linemod.linemod_aux import *
from torchvision import transforms
from sklearn.cluster import MeanShift, estimate_bandwidth

import itertools
import struct  # get_image_size
import imghdr  # get_image_size
import operator
import random
import numpy
import cv2


def get_class_colors(class_id):
    colordict = {'gray': [128, 128, 128], 'silver': [192, 192, 192], 'black': [0, 0, 0],
                 'maroon': [128, 0, 0], 'red': [255, 0, 0], 'purple': [128, 0, 128], 'fuchsia': [255, 0, 255],
                 'green': [0, 128, 0],
                 'lime': [0, 255, 0], 'olive': [128, 128, 0], 'yellow': [255, 255, 0], 'navy': [0, 0, 128],
                 'blue': [0, 0, 255],
                 'teal': [0, 128, 128], 'aqua': [0, 255, 255], 'orange': [255, 165, 0], 'indianred': [205, 92, 92],
                 'lightcoral': [240, 128, 128], 'salmon': [250, 128, 114], 'darksalmon': [233, 150, 122],
                 'lightsalmon': [255, 160, 122], 'crimson': [220, 20, 60], 'firebrick': [178, 34, 34],
                 'darkred': [139, 0, 0],
                 'pink': [255, 192, 203], 'lightpink': [255, 182, 193], 'hotpink': [255, 105, 180],
                 'deeppink': [255, 20, 147],
                 'mediumvioletred': [199, 21, 133], 'palevioletred': [219, 112, 147], 'coral': [255, 127, 80],
                 'tomato': [255, 99, 71], 'orangered': [255, 69, 0], 'darkorange': [255, 140, 0], 'gold': [255, 215, 0],
                 'lightyellow': [255, 255, 224], 'lemonchiffon': [255, 250, 205],
                 'lightgoldenrodyellow': [250, 250, 210],
                 'papayawhip': [255, 239, 213], 'moccasin': [255, 228, 181], 'peachpuff': [255, 218, 185],
                 'palegoldenrod': [238, 232, 170], 'khaki': [240, 230, 140], 'darkkhaki': [189, 183, 107],
                 'lavender': [230, 230, 250], 'thistle': [216, 191, 216], 'plum': [221, 160, 221],
                 'violet': [238, 130, 238],
                 'orchid': [218, 112, 214], 'magenta': [255, 0, 255], 'mediumorchid': [186, 85, 211],
                 'mediumpurple': [147, 112, 219], 'blueviolet': [138, 43, 226], 'darkviolet': [148, 0, 211],
                 'darkorchid': [153, 50, 204], 'darkmagenta': [139, 0, 139], 'indigo': [75, 0, 130],
                 'slateblue': [106, 90, 205],
                 'darkslateblue': [72, 61, 139], 'mediumslateblue': [123, 104, 238], 'greenyellow': [173, 255, 47],
                 'chartreuse': [127, 255, 0], 'lawngreen': [124, 252, 0], 'limegreen': [50, 205, 50],
                 'palegreen': [152, 251, 152],
                 'lightgreen': [144, 238, 144], 'mediumspringgreen': [0, 250, 154], 'springgreen': [0, 255, 127],
                 'mediumseagreen': [60, 179, 113], 'seagreen': [46, 139, 87], 'forestgreen': [34, 139, 34],
                 'darkgreen': [0, 100, 0], 'yellowgreen': [154, 205, 50], 'olivedrab': [107, 142, 35],
                 'darkolivegreen': [85, 107, 47], 'mediumaquamarine': [102, 205, 170], 'darkseagreen': [143, 188, 143],
                 'lightseagreen': [32, 178, 170], 'darkcyan': [0, 139, 139], 'cyan': [0, 255, 255],
                 'lightcyan': [224, 255, 255],
                 'paleturquoise': [175, 238, 238], 'aquamarine': [127, 255, 212], 'turquoise': [64, 224, 208],
                 'mediumturquoise': [72, 209, 204], 'darkturquoise': [0, 206, 209], 'cadetblue': [95, 158, 160],
                 'steelblue': [70, 130, 180], 'lightsteelblue': [176, 196, 222], 'powderblue': [176, 224, 230],
                 'lightblue': [173, 216, 230], 'skyblue': [135, 206, 235], 'lightskyblue': [135, 206, 250],
                 'deepskyblue': [0, 191, 255], 'dodgerblue': [30, 144, 255], 'cornflowerblue': [100, 149, 237],
                 'royalblue': [65, 105, 225], 'mediumblue': [0, 0, 205], 'darkblue': [0, 0, 139],
                 'midnightblue': [25, 25, 112],
                 'cornsilk': [255, 248, 220], 'blanchedalmond': [255, 235, 205], 'bisque': [255, 228, 196],
                 'navajowhite': [255, 222, 173], 'wheat': [245, 222, 179], 'burlywood': [222, 184, 135],
                 'tan': [210, 180, 140],
                 'rosybrown': [188, 143, 143], 'sandybrown': [244, 164, 96], 'goldenrod': [218, 165, 32],
                 'darkgoldenrod': [184, 134, 11], 'peru': [205, 133, 63], 'chocolate': [210, 105, 30],
                 'saddlebrown': [139, 69, 19],
                 'sienna': [160, 82, 45], 'brown': [165, 42, 42], 'snow': [255, 250, 250], 'honeydew': [240, 255, 240],
                 'mintcream': [245, 255, 250], 'azure': [240, 255, 255], 'aliceblue': [240, 248, 255],
                 'ghostwhite': [248, 248, 255], 'whitesmoke': [245, 245, 245], 'seashell': [255, 245, 238],
                 'beige': [245, 245, 220], 'oldlace': [253, 245, 230], 'floralwhite': [255, 250, 240],
                 'ivory': [255, 255, 240],
                 'antiquewhite': [250, 235, 215], 'linen': [250, 240, 230], 'lavenderblush': [255, 240, 245],
                 'mistyrose': [255, 228, 225], 'gainsboro': [220, 220, 220], 'lightgrey': [211, 211, 211],
                 'darkgray': [169, 169, 169], 'dimgray': [105, 105, 105], 'lightslategray': [119, 136, 153],
                 'slategray': [112, 128, 144], 'darkslategray': [47, 79, 79], 'white': [255, 255, 255]}

    colornames = list(colordict.keys())
    assert (class_id < len(colornames))

    r, g, b = colordict[colornames[class_id]]

    return b, g, r  # for OpenCV

def draw_axis(img, k, rt, scale=1, linewidth=1, xcolor=[0, 0, 255], ycolor=[0, 255, 0], zcolor=[255, 0, 0]):
    # X Y Z corresponding to R G B
    k = k.reshape(3, 3)
    rt = rt.reshape(3, 4)
    anchors = np.array([[0,0,0], [scale, 0, 0], [0, scale, 0], [0, 0, scale]])

    p = np.matmul(k, np.matmul(rt[:3, 0:3], anchors.T) + rt[:3, 3].reshape(-1, 1))
    x = p[0] / (p[2] + 1e-5)
    y = p[1] / (p[2] + 1e-5)

    # origin
    # img = cv2.circle(img, (x[0], y[0]), 5, (0, 0, 255), -1)

    # check nan
    if x[0] != x[0] or x[1] != x[1] or x[2] != x[2] or x[3] != x[3]:
        return img
    if y[0] != y[0] or y[1] != y[1] or y[2] != y[2] or y[3] != y[3]:
        return img

    # white borders
    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[1] + 0.5), int(y[1] + 0.5)), (255, 255, 255),
                   linewidth + 1, cv2.LINE_AA)
    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[2] + 0.5), int(y[2] + 0.5)), (255, 255, 255),
                   linewidth + 1, cv2.LINE_AA)
    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[3] + 0.5), int(y[3] + 0.5)), (255, 255, 255),
                   linewidth + 1, cv2.LINE_AA)

    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[1] + 0.5), int(y[1] + 0.5)), xcolor, linewidth,
                   cv2.LINE_AA)
    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[2] + 0.5), int(y[2] + 0.5)), ycolor, linewidth,
                   cv2.LINE_AA)
    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[3] + 0.5), int(y[3] + 0.5)), zcolor, linewidth,
                   cv2.LINE_AA)

    return img

def draw_axis2(img, k, rt, scale=1, linewidth=1, xcolor=[0, 0, 255], ycolor=[0, 255, 0], zcolor=[255, 0, 0]):
    # X Y Z corresponding to R G B
    k = k.reshape(3, 3)
    rt = rt.reshape(3, 4)
    anchors = np.array([[scale, 0, 0], [-scale, 0, 0], [0, scale, 0], [0, -scale, 0], [0, 0, scale], [0, 0, -scale]])

    p = np.matmul(k, np.matmul(rt[:3, 0:3], anchors.T) + rt[:3, 3].reshape(-1, 1))
    x = p[0] / (p[2] + 1e-5)
    y = p[1] / (p[2] + 1e-5)

    # origin
    # img = cv2.circle(img, (x[0], y[0]), 5, (0, 0, 255), -1)

    # check nan
    if x[0] != x[0] or x[1] != x[1] or x[2] != x[2] or x[3] != x[3] or x[4] != x[4] or x[5] != x[5]:
        return img
    if y[0] != y[0] or y[1] != y[1] or y[2] != y[2] or y[3] != y[3] or y[4] != y[4] or y[5] != y[5]:
        return img

    # end points
    for i in range(len(anchors)):
        img = cv2.circle(img, (int(x[i] + 0.5), int(y[i] + 0.5)), 7, (255, 255, 255), -1, cv2.LINE_AA)
        img = cv2.circle(img, (int(x[i] + 0.5), int(y[i] + 0.5)), 5, get_class_colors(i), -1, cv2.LINE_AA)

    # white borders
    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[1] + 0.5), int(y[1] + 0.5)), (255, 255, 255),
                   linewidth + 1, cv2.LINE_AA)
    img = cv2.line(img, (int(x[2] + 0.5), int(y[2] + 0.5)), (int(x[3] + 0.5), int(y[3] + 0.5)), (255, 255, 255),
                   linewidth + 1, cv2.LINE_AA)
    img = cv2.line(img, (int(x[4] + 0.5), int(y[4] + 0.5)), (int(x[5] + 0.5), int(y[5] + 0.5)), (255, 255, 255),
                   linewidth + 1, cv2.LINE_AA)

    img = cv2.line(img, (int(x[0] + 0.5), int(y[0] + 0.5)), (int(x[1] + 0.5), int(y[1] + 0.5)), xcolor, linewidth,
                   cv2.LINE_AA)
    img = cv2.line(img, (int(x[2] + 0.5), int(y[2] + 0.5)), (int(x[3] + 0.5), int(y[3] + 0.5)), ycolor, linewidth,
                   cv2.LINE_AA)
    img = cv2.line(img, (int(x[4] + 0.5), int(y[4] + 0.5)), (int(x[5] + 0.5), int(y[5] + 0.5)), zcolor, linewidth,
                   cv2.LINE_AA)

    return img


def sigmoid(x):
    return 1.0 / (math.exp(-x) + 1.)


def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x / x.sum()
    return x


def reproj_confidence0(x, y, gtx, gty):
    dis = ((x - gtx).pow(2) + (y - gty).pow(2)).sqrt().mean(dim=1)
    conf = torch.exp(-10 * dis)
    return conf


def reproj_confidence1(x, y, gtx, gty):
    dis = ((x - gtx).pow(2) + (y - gty).pow(2)).sqrt()
    conf = torch.exp(-10 * dis)
    return conf


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    # print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes


def NonMaximumSuppression(output, width, height, intrinsics, conf_thresh, nms_thresh, batchIdx):
    layerCnt = len(output)
    rawPred = []
    for l in range(layerCnt):
        layerId = output[l][0]
        out_data = output[l][2]

        conf = out_data[0][batchIdx]
        predx = out_data[1][batchIdx]
        predy = out_data[2][batchIdx]
        cls_confs = out_data[3][batchIdx]
        cls_ids = out_data[4][batchIdx]
        vpoints = out_data[5]
        keypoints = None
        if len(out_data) > 6:
            keypoints = out_data[6]
        nH, nW, nV = predx.shape
        assert (len(vpoints) == nV)

        for cy in range(nH):
            for cx in range(nW):
                confd = conf[cy][cx]
                confc = cls_confs[cy][cx]
                totalConf = confd * confc
                # totalConf = confd
                id = int(cls_ids[cy][cx])

                if totalConf > conf_thresh:
                    # if totalConf > 0:
                    px = predx[cy][cx].reshape(-1, 1) * width
                    py = predy[cy][cx].reshape(-1, 1) * height
                    pxy = np.concatenate((px, py), 1)
                    vp3d = vpoints
                    candiPred = [[totalConf], pxy, vpoints, [0, 0, 0], [id], [id], [-1], [(cx + 0.5) / nW],
                                 [(cy + 0.5) / nH], [layerId]]
                    if not keypoints is None:
                        num_keypoint = keypoints[0].shape[0]
                        clsid = int(id / num_keypoint)
                        partid = id - clsid * num_keypoint
                        vp3d = keypoints[clsid][partid] + vpoints
                        candiPred = [[totalConf], pxy, vpoints, keypoints[clsid][partid], [id], [clsid], [partid],
                                     [(cx + 0.5) / nW], [(cy + 0.5) / nH], [layerId]]

                    # retval, rot, trans = cv2.solvePnP(vpoints.reshape(nV, 1, -1), pxy.reshape(nV, 1, -1),
                    #                                   intrinsics, None, None, None, False, cv2.SOLVEPNP_EPNP)



                    retval, rot, trans = cv2.solvePnP(vp3d, pxy,
                                                      intrinsics, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                    assert (retval == True)
                    R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
                    T = trans.reshape(-1, 1)
                    rt = np.concatenate((R, T), 1)
                    candiPred.append(rt)
                    # vxy = vertices_reprojection(vpoints, rt, intrinsics)
                    # candiPred.append(vxy)
                    #
                    rawPred.append(candiPred)

    newOut = rawPred

    if len(rawPred) > 0:
        # clustering by MeanShift
        if True:
            # if False:
            vp2ds = []
            for i in range(len(rawPred)):
                # pt = vertices_reprojection(refpoints, rawPred[i][-1], intrinsics)
                pt = vertices_reprojection(np.array([[0, 0, 0]]), rawPred[i][-1], intrinsics)
                cid = rawPred[i][5][0]
                clsPenalty = cid * 1e6
                vp2ds.append(np.concatenate((np.array([clsPenalty]), pt.reshape(-1)), 0))
            # bandwidth = estimate_bandwidth(vp2ds)
            bandwidth = 10000
            # bandwidth = 100
            ms = MeanShift(bandwidth=bandwidth)
            ms.fit(vp2ds)
            labels = ms.labels_
            # cluster_centers = ms.cluster_centers_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            # print(n_clusters_)

        # fusion information in clustering
        # if True:
        if False:
            # choose best
            newOut = []
            for lab in labels_unique:
                predcluster = [rawPred[i] for i in range(len(rawPred)) if labels[i] == lab]
                if len(predcluster) < 5:
                    continue
                maxPred = None
                for p in predcluster:
                    if maxPred is None or p[0] > maxPred[0]:
                        maxPred = p
                newOut.append(maxPred)
        if True:
            # if False:
            # fusion in clustering
            newOut = []
            for lab in labels_unique:
                predcluster = [rawPred[i] for i in range(len(rawPred)) if labels[i] == lab]
                if len(predcluster) < 2:
                    continue
                p3d = None
                p2d = None
                mergedConf = []
                mergedClsId = []
                mergedCx = []
                mergedCy = []
                for p in predcluster:
                    conf, puv, pxyz, opoint, id, clsid, partid, cx, cy, layerId, rt = p
                    #
                    mergedConf.append(conf[0])
                    mergedClsId.append(clsid[0])
                    mergedCx.append(cx[0])
                    mergedCy.append(cy[0])
                    #
                    if p3d is None or p2d is None:
                        p3d = pxyz
                        p2d = puv
                    else:
                        p3d = np.concatenate((p3d, pxyz), axis=0)
                        p2d = np.concatenate((p2d, puv), axis=0)

                sumConf = np.array(mergedConf).sum()

                retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)
                # retval, rot, trans = cv2.solvePnP(p3d, p2d, intrinsics, None, flags=cv2.SOLVEPNP_ITERATIVE)
                if not retval:
                    continue

                R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
                T = trans.reshape(-1, 1)
                rt = np.concatenate((R, T), 1)

                # # remove outlier
                # clusterMaxConf = np.array(mergedConf).max()
                # if clusterMaxConf < 0.4:
                #     continue

                newOut.append(
                    [mergedConf, None, None, None, mergedClsId, mergedClsId, -1, mergedCx, mergedCy, None, rt])
        # if True:
        if False:
            # non-maximum suppression
            newOut = []
            selectionFlag = [1] * len(rawPred)
            rawPred.sort(key=lambda x: x[0][0], reverse=True)  # sort by the first element (confidence)
            for i in range(len(rawPred)):
                if selectionFlag[i] == 1:
                    newOut.append(rawPred[i])
                    p1 = rawPred[i][1]
                    clsPenalty1 = rawPred[i][5][0] * 1e5
                    for j in range(i + 1, len(rawPred)):
                        p2 = rawPred[j][1]
                        clsPenalty2 = rawPred[j][5][0] * 1e5
                        pixdis = np.sqrt(np.power((p1 - p2), 2).sum(axis=1)).mean() + abs(clsPenalty1 - clsPenalty2)
                        # print(pixdis)
                        if pixdis < nms_thresh:
                            selectionFlag[j] = 0
        #
    return newOut

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) / num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert (output.size(1) == (5 + num_classes) * num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []
    output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,
                                                                                                        batch * num_anchors * h * w)

    grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch * num_anchors, 1, 1).view(
        batch * num_anchors * h * w).type_as(output)  # cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h * w).view(batch * num_anchors * h * w).type_as(output)  # cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])

    cls_confs = F.softmax(Variable(output[5:5 + num_classes].transpose(0, 1)), dim=1).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx / w, bcy / h, bw / w, bh / h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('      boxes filter : %f' % (t3 - t2))
        print('---------------------------------')
    return all_boxes


def get_prediction_candidates(output, num_classes, num_vpoints,
                              anchors, num_anchors, only_objectness=1, validation=False):
    anchor_step = len(anchors) / num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    nB = output.size(0)
    nA = num_anchors
    nC = num_classes
    nV = num_vpoints
    assert (output.size(1) == nA * (1 + 2 * nV + nC))
    nH = output.size(2)
    nW = output.size(3)

    t0 = time.time()

    output = output.view(nB * nA, (1 + 2 * nV + nC), nH * nW).transpose(0, 1). \
        contiguous().view((1 + 2 * nV + nC), nB * nA * nH * nW)

    conf = F.sigmoid(output[0].view(nB, nA, nH, nW))
    x = output[1:1 + nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
    y = output[1 + nV:1 + 2 * nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
    cls = output[1 + 2 * nV:1 + 2 * nV + nC].transpose(0, 1)
    t1 = time.time()

    grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA * nV, 1, 1). \
        view(nB, nA, nV, nH, nW).type_as(output)
    grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA * nV, 1, 1). \
        view(nB, nA, nV, nH, nW).type_as(output)
    grid_x = grid_x.permute(0, 1, 3, 4, 2).contiguous()
    grid_y = grid_y.permute(0, 1, 3, 4, 2).contiguous()
    anchor_w = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW * nV).view(nB, nA, nH, nW, nV).type_as(output)
    anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW * nV).view(nB, nA, nH, nW, nV).type_as(output)

    predx = (x.data * anchor_w + grid_x) / nW
    predy = (y.data * anchor_h + grid_y) / nH

    cls_confs, cls_ids = torch.max(F.softmax(cls, 1), 1)
    cls_confs = cls_confs.view(nB, nA, nH, nW)
    cls_ids = cls_ids.view(nB, nA, nH, nW)
    t1 = time.time()

    conf_max, conf_max_id = torch.max(conf, dim=1)
    maxmask = torch.zeros(nB * nH * nW, nA).long()
    d0 = torch.linspace(0, nB * nH * nW - 1, nB * nH * nW).long()
    maxmask[d0, conf_max_id.view(-1)] = 1
    maxmask = (maxmask == 1)
    assert ((conf.permute(0, 2, 3, 1).contiguous().view(-1, nA)[maxmask] == conf_max.view(-1)).sum() == nB * nH * nW)

    conf = conf.permute(0, 2, 3, 1).contiguous().view(-1, nA)[maxmask].view(nB, nH, nW)
    cls_confs = cls_confs.permute(0, 2, 3, 1).contiguous().view(-1, nA)[maxmask].view(nB, nH, nW)
    cls_ids = cls_ids.permute(0, 2, 3, 1).contiguous().view(-1, nA)[maxmask].view(nB, nH, nW)

    maskxy = maxmask.view(-1, 1).repeat(1, nV).view(nB, nH, nW, nA, nV)
    predx = predx.permute(0, 2, 3, 1, 4)[maskxy].contiguous().view(nB, nH, nW, nV)
    predy = predy.permute(0, 2, 3, 1, 4)[maskxy].contiguous().view(nB, nH, nW, nV)

    # copy to CPU
    conf = convert2cpu(conf).numpy()
    predx = convert2cpu(predx).numpy()
    predy = convert2cpu(predy).numpy()
    cls_confs = convert2cpu(cls_confs).numpy()
    cls_ids = convert2cpu_long(cls_ids).numpy()

    t2 = time.time()

    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1 - t0))
        print('        gpu to cpu : %f' % (t2 - t1))
        print('---------------------------------')

    return [conf, predx, predy, cls_confs, cls_ids]


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(((box[0] - box[2] / 2.0) * width) + 0.5)
        y1 = int(((box[1] - box[3] / 2.0) * height) + 0.5)
        x2 = int(((box[0] + box[2] / 2.0) * width) + 0.5)
        y2 = int(((box[1] + box[3] / 2.0) * height) + 0.5)

        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def plot_boxes(img, boxes, savename=None, class_names=None):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]);

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.width
    height = img.height
    draw = ImageDraw.Draw(img)
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            rgb = (red, green, blue)
            draw.text((x1, y1), class_names[cls_id], fill=rgb)
        draw.rectangle([x1, y1, x2, y2], outline=rgb)
    if savename:
        print("save plot results to %s" % savename)
        img.save(savename)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        if truths[i][3] < min_box_scale:
            continue
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
    return np.array(new_truths)


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def image2torch(img):
    width = img.width
    height = img.height
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
    img = img.view(1, 3, height, width)
    img = img.float().div(255.0)
    return img


def do_detect(model, rawimg, intrinsics, gt_poses, bestCnt, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    t0 = time.time()

    height, width, _ = rawimg.shape
    # scale
    img = cv2.resize(rawimg, (model.width, model.height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if isinstance(img, Image.Image):
        width = img.width
        height = img.height
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        img = img.view(height, width, 3).transpose(0, 1).transpose(0, 2).contiguous()
        img = img.view(1, 3, height, width)
        img = img.float().div(255.0)
    elif type(img) == np.ndarray:  # cv2 image
        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    else:
        print("unknow image type")
        exit(-1)

    t1 = time.time()

    if use_cuda:
        img = img.cuda()
    img = Variable(img)
    t2 = time.time()

    out_preds = model(img, None)

    t3 = time.time()

    #predPose, conf, deviation, p2d = NonMaximumSuppression_sep_seg_wta(rawimg, out_preds, width, height, intrinsics, gt_poses, conf_thresh, nms_thresh, 0, bestCnt)
    predPose, conf, p2d = NonMaximumSuppression_sep_seg_wta(rawimg, out_preds, width, height, intrinsics, gt_poses, conf_thresh, nms_thresh, 0, bestCnt)
    t4 = time.time()

    #return predPose, conf, deviation, p2d
    return predPose, conf, p2d


def NonMaximumSuppression_sep_seg_wta(inputimg, output, width, height, intrinsics, gt_poses, conf_thresh, nms_thresh, batchIdx, bestCnt):
    layerCnt = len(output)
    assert(layerCnt == 2)

    segshowimg = np.copy(inputimg)

    cls_confs = output[0][2][0][batchIdx]
    cls_ids = output[0][2][1][batchIdx]
    predx = output[1][2][0][batchIdx]
    predy = output[1][2][1][batchIdx]
    det_confs = output[1][2][2][batchIdx]
    keypoints = output[1][2][3]

    #print('cls_confs', cls_confs)
    #print('cls_ids', cls_ids)
    #print('predx', predx)
    #print('predy', predy)
    #print('det_confs', det_confs)
    #print('keypoints', keypoints)

    nH, nW, nV = predx.shape
    nC = cls_ids.max() + 1

    outPred = []
    showRepImg = np.copy(inputimg)
    segshowimg = np.copy(inputimg)
    inlierImg = np.copy(inputimg)

    mx = predx.mean(axis=2) # average x positions
    my = predy.mean(axis=2) # average y positions
    mdConf = det_confs.mean(axis=2) # average 2D confidence
    csi_mean = 0
    for cidx in range(nC):
        # skip background
        if cidx == 0:
            continue
        foremask = (cls_ids == cidx)
        cidx -= 1

        foreCnt = foremask.sum()
        if foreCnt < 1:
            continue

        xs = predx[foremask]
        ys = predy[foremask]
        ds = det_confs[foremask]
        cs = cls_confs[foremask]
        centerxys = np.concatenate((mx[foremask].reshape(-1,1), my[foremask].reshape(-1,1)), 1)

        # choose the item with maximum detection confidence
        # actually, this will choose only one object instance for each type, this is true for OccludedLINEMOD and YCB-Video dataset
        maxIdx = np.argmax(mdConf[foremask])
        refxys = centerxys[maxIdx].reshape(1,-1).repeat(foreCnt, axis=0)
        selected = (np.linalg.norm(centerxys - refxys, axis=1) < 0.2)

        xsi = xs[selected] * width
        ysi = ys[selected] * height
        dsi = ds[selected]
        csi = cs[selected]

        csi_mean = csi.mean()

        gridCnt = len(xsi)
        assert(gridCnt > 0)

        # show segmentation image
        if False:
            nH, nW = foremask.shape
            labIdx = np.argwhere(foremask == 1)

            minx, maxx, miny, maxy = width,0,height,0
            for i in range(len(labIdx)):
                tmpy = int(((labIdx[i][0] + 0.5) / nH) * height + 0.5)
                tmpx = int(((labIdx[i][1] + 0.5) / nW) * width + 0.5)
                tmpr = 7
                segshowimg = cv2.rectangle(segshowimg, (tmpx-tmpr,tmpy-tmpr), (tmpx+tmpr,tmpy+tmpr), get_class_colors(cidx), -1)
                if tmpx < minx:
                    minx = tmpx
                if tmpx > maxx:
                    maxx = tmpx
                if tmpy < miny:
                    miny = tmpy
                if tmpy > maxy:
                    maxy = tmpy
            print(minx, maxx, miny, maxy)
            name = 'img/segshowimg' + str(csi.mean()) + '.png'
            cv2.imwrite(name, segshowimg)
        # compute ground truth 2D reprojections
        if not gt_poses is None and not gt_poses[cidx] is None:
            gt_2ds = vertices_reprojection(keypoints[cidx], gt_poses[cidx], intrinsics)


        # show all 2D reprojections
        if False:
            for i in range(len(xsi)):
                for j in range(nV):
                    x = xsi[i][j]
                    y = ysi[i][j]
                    showRepImg = cv2.circle(showRepImg, (int(x), int(y)), 2, [255, 0, 0], -1)

            name = 'img/pnpall' + str(csi.mean()) + '.png'
            cv2.imwrite(name, showRepImg)



        # choose best N count
        p2d = None
        p3d = None
        candiBestCnt = min(gridCnt, bestCnt)
        for i in range(candiBestCnt):
            bestGrids = dsi.argmax(axis=0)
            validmask = (dsi[bestGrids, list(range(nV))] > 0.5)
            xsb = xsi[bestGrids, list(range(nV))][validmask]
            ysb = ysi[bestGrids, list(range(nV))][validmask]
            t2d = np.concatenate((xsb.reshape(-1, 1), ysb.reshape(-1, 1)), 1)
            t3d = keypoints[cidx][validmask]
            if p2d is None:
                p2d = t2d
                p3d = t3d
            else:
                p2d = np.concatenate((p2d, t2d), 0)
                p3d = np.concatenate((p3d, t3d), 0)
            dsi[bestGrids, list(range(nV))] = 0

        if len(p3d) < 6:
            continue

        # show the selected 2D reprojections
        if False:
            for i in range(len(p2d)):
                x = p2d[i][0]
                y = p2d[i][1]
                inlierImg = cv2.circle(inlierImg, (int(x), int(y)), 2, [0, 0, 255], -1)
            name = 'img/pnp_inlierImg' + str(csi.mean()) + '.png'
            cv2.imwrite(name, inlierImg)

        coord_1 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==0])
        coord_2 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==1])
        coord_3 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==2])
        coord_4 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==3])
        coord_5 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==4])
        coord_6 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==5])
        coord_7 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==6])
        coord_8 = numpy.std([p[0] for i,p in enumerate(p2d) if i % 8 ==7])
        coord_11 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==0])
        coord_21 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==1])
        coord_31 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==2])
        coord_41 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==3])
        coord_51 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==4])
        coord_61 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==5])
        coord_71 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==6])
        coord_81 = numpy.std([p[1] for i,p in enumerate(p2d) if i % 8 ==7])

        #deviation=coord_1+coord_2+coord_3+coord_4+coord_5+coord_6+coord_7+coord_8+coord_11+coord_21+coord_31+coord_41+coord_51+coord_61+coord_71+coord_81


        retval, rot, trans, inliers = cv2.solvePnPRansac(p3d, p2d, intrinsics, None,
                                                         iterationsCount = 100, reprojectionError=32.0,
                                                         flags=cv2.SOLVEPNP_ITERATIVE)

        if not retval:
            continue

        R = cv2.Rodrigues(rot)[0]  # convert to rotation matrix
        T = trans.reshape(-1, 1)
        rt = np.concatenate((R, T), 1)

        outPred.append([cidx, rt, 1, None, None, None, [cidx], -1, [0], [0], None])

    #return outPred, csi_mean, deviation, p2d
    #print(str(outPred) + ' ' + str(csi_mean) + ' ' + str(p2d))
    return outPred, csi_mean, p2d


def visualize_predictions(predPose, image, vertex, intrinsics):
    height, width, _ = image.shape
    visPreds = []
    confImg = np.copy(image)
    contourImg = np.copy(image)
    for p in predPose:
        outid, rt, conf, puv, pxyz, opoint, clsid, partid, cx, cy, layerId = p

        if True:
            # if False:
            vp = vertices_reprojection(vertex[outid][:], rt, intrinsics)
            for p in vp:
                if p[0] != p[0] or p[1] != p[1]:  # check nan
                    continue
                confImg = cv2.circle(confImg, (int(p[0]), int(p[1])), 10, get_class_colors(outid), -1, cv2.LINE_AA)

    return confImg


def parse_predictions(predPose, image, vertex, intrinsics):
    height, width, _ = image.shape
    detCnt = len(predPose)
    rois = np.zeros((detCnt, 6), np.float32)
    poses = np.zeros((detCnt, 12), np.float32)
    for i in range(detCnt):
        conf, pxy, vpoints, opoint, id, clsid, partid, cx, cy, layerId, rt = predPose[i]

        # choose the best counter
        cid = max(clsid, key=clsid.count)
        # surface reprojection
        vp = vertices_reprojection(vertex[cid][0:500], rt, intrinsics).T
        rois[i] = np.array([0, cid + 1, vp[0].min(), vp[1].min(), vp[0].max(), vp[1].max()])
        poses[i] = rt.reshape(1, -1)

    return rois, poses


def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options


def scale_bboxes(bboxes, width, height):
    import copy
    dets = copy.deepcopy(bboxes)
    for i in range(len(dets)):
        dets[i][0] = dets[i][0] * width
        dets[i][1] = dets[i][1] * height
        dets[i][2] = dets[i][2] * width
        dets[i][3] = dets[i][3] * height
    return dets


def file_lines(thefilepath):
    with open(thefilepath, 'r') as file:
        linelist = file.readlines()
    return len(linelist)


def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg' or imghdr.what(fname) == 'jpg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height


def logging(message):
    print('%s %s' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))
