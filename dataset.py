#!/usr/bin/python
# encoding: utf-8
from utils_satellite import quaternion2rotation
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths
from image import *
import scipy.io as sio
import cv2
import json


class RandomBackgroundDataset(Dataset):
    def __init__(self, backRoot, foreRoot, shape=None, shuffle=True,
                 transform=None, target_transform=None, train=False, seen=0,
                 batch_size=64, num_workers=4):
        with open(foreRoot, 'r') as file:
            self.fore_lines = file.readlines()

        if shuffle:
            random.shuffle(self.fore_lines)

        self.nSamples = len(self.fore_lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.rawShape = shape
        self.currShape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.poses = {}

        # fix pose
        with open('/cvlabdata1/cvlab/datasets_kgerard/speed/train.json') as example:
            d = json.load(example)
            for line in d:
                qs = line['q_vbs2tango']
                rs = line['r_Vo2To_vbs_true']
                name, q, r = line['filename'], np.array([float(qs[0]), float(qs[1]), float(qs[2]), float(qs[3])]), np.array([float(rs[0]), float(rs[1]), float(rs[2])])
                self.poses[name[:-4]] = np.array([np.hstack((np.transpose(quaternion2rotation(q)), np.expand_dims(r, 1)))])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        # no flip
        withFlip = False
        jitter = 0
        withDistortion = False
        withRotation = True
        withZoom = True

        assert index <= len(self), 'index range error'
        imgpath = self.fore_lines[index].rstrip()

        annotPath = imgpath.replace('.png', '.npz')
        annot = np.load(annotPath)
        segImg = annot['segmentation']
        poses = annot['poses']
        #poses = self.poses[annotPath[-13:-4]]
        objsID = annot['objectsID']
        rawk = annot['intrinsics']

        bounds = np.load(annotPath.replace('train', 'large_info'))['satellite_bounds']

        hue = 0.2
        saturation = 1.5
        exposure = 1.5
        noise = 0.1
        smooth = 1
        foreImg = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        foreImg = np.stack((foreImg,)*3, axis=-1)
        backImg = foreImg
        img, label, seg = load_data_detection(backImg, foreImg, segImg, poses, objsID, rawk, bounds,
                                         self.currShape, hue, saturation, exposure, noise, smooth, jitter, withFlip, withRotation, withDistortion, withZoom)


        label = torch.from_numpy(label)
        seg = torch.from_numpy(seg)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
            seg = self.target_transform(seg)


        self.seen = self.seen + self.num_workers

        return (img, label, seg)
