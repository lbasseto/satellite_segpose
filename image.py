#!/usr/bin/python
# encoding: utf-8
import random
import os
import numpy as np
from linemod.linemod_aux import *
from PIL import Image
from utils import *
from numpy import linalg as LA
from utils_satellite import zoom_on_satellite

def distort_image(im, hue, sat, val):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) # convert to HSV first
    h, s, v = cv2.split(hsv)

    nh = h + hue * 179
    nh[nh > 179] = (nh - 179)[nh > 179]
    nh[nh < 0] = (nh + 179)[nh < 0]
    ns = s * sat
    ns[ns > 255] = 255
    nv = v * val
    nv[nv > 255] = 255

    h = np.uint8(nh)
    s = np.uint8(ns)
    v = np.uint8(nv)

    # change back
    im = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2):
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure, noise, smooth):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    #res = distort_image(im, dhue, dsat, dexp)

    # add noise
    noisesigma = random.uniform(0, noise)
    gauss = np.random.normal(0, noisesigma, res.shape) * 255
    res = res + gauss

    res[res > 255] = 255
    res[res < 0] = 0

    # add smooth
    smoothsigma = random.uniform(0.001, smooth)
    res = cv2.GaussianBlur(res, (7, 7), smoothsigma, cv2.BORDER_DEFAULT)

    return np.uint8(res)

def data_augmentation(img, segImg, shape, jitter, withFlip = False, withRotation=False):
    oh = img.shape[0]
    ow = img.shape[1]
    assert(segImg.shape[0] == oh and segImg.shape[1] == ow)

    sx = 1
    sy = 1
    dx = 0
    dy = 0

    # sized = cropped.resize(shape)
    sized = cv2.resize(img, shape)
    sizedSeg = cv2.resize(segImg, shape)

    # random rotation
    rM = None
    if withRotation:
        cx = 0.5 * shape[0]
        cy = 0.5 * shape[1]
        ang = random.uniform(-45, 45)
        scale = random.uniform(0.8, 1.2)
        # scale = 1.0
        rs = cv2.getRotationMatrix2D((cx, cy), ang, scale)  # rotation with scale
        rM = np.concatenate((rs, [[0, 0, 1]]), axis=0)
        # rotate image accordingly
        sized = cv2.warpAffine(sized, rM[:2], shape)
        sizedSeg = cv2.warpAffine(sizedSeg, rM[:2], shape)

    flip=0

    smoothsigma = random.uniform(0.001, 1)
    #sized = cv2.GaussianBlur(sized, (7, 7), smoothsigma, cv2.BORDER_DEFAULT)

    return sized, sizedSeg, flip, dx,dy,sx,sy, rM

def fill_truth_detection(poses, objsID, rawK,
                         rw, rh, cw, ch, flip, dx, dy, sx, sy, rM):
    max_objects = 40
    columnNum = 26
    label = np.zeros((max_objects, columnNum), np.float32)
    label.fill(-1)

    objCnt = len(objsID)
    assert (objCnt <= max_objects)

    # compute the new K according to transform
    trans = np.array([[cw / rw * sx, 0, -cw * dx],
                      [0, ch / rh * sy, -ch * dy],
                      [0, 0, 1]])

    k = np.matmul(trans, rawK)  # new intrinsic
    if rM is not None:
        k = np.matmul(rM, k) # multiply rotation matrix

    curridx = 0
    for i in range(objCnt):
        id = int(objsID[i])
        assert(id >= 0)
        op = vertices_reprojection(np.array([[0,0,0]]), poses[i], k)[0]
        l = np.concatenate(([id], [op[0] / cw], [op[1] / ch], [cw, ch],
                            k.reshape(-1), poses[i].reshape(-1)), 0).reshape(1, -1)
        label[curridx] = l
        curridx += 1
    return label

def load_data_detection(backImg, foreImg, segImg, poses, objsID, rawk, bounds, shape,
                        hue, saturation, exposure, noise, smooth, jitter, withFlip, withRotation, withDistortion, withZoom):

    # TODO add prob zoom
    if withZoom:
        foreImg, segImg, rawk = zoom_on_satellite(foreImg, segImg, poses[0], rawk, bounds)

    ## data augmentation
    rh = foreImg.shape[0]
    rw = foreImg.shape[1]

    img, segImg, flip, dx, dy, sx, sy, rM = data_augmentation(foreImg, segImg, shape, jitter, withFlip, withRotation)
    label = fill_truth_detection(poses, objsID, rawk,
                                 rw, rh, img.shape[1], img.shape[0],
                                 flip, dx, dy, 1./sx, 1./sy, rM)
    # random distort the image
    if withDistortion:
        img = random_distort_image(img, hue, saturation, exposure, noise, smooth)

    return img, label, segImg
