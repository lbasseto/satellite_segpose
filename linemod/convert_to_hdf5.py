import os
import random
import h5py
import numpy as np
import cv2

def save_backgrounds(filelist, outhdf):
    with open(filelist, 'r') as file:
        backfiles = file.readlines()

    max_objects = 50

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(outhdf, mode='w')

    hdf5_file.create_dataset("images", (len(backfiles), ),
                             dtype = h5py.special_dtype(vlen=np.dtype('uint8')))
    hdf5_file.create_dataset("shapes", (len(backfiles), ),
                             dtype = h5py.special_dtype(vlen=np.dtype('int32')))

    for i in range(len(backfiles)):
        imgName = backfiles[i].rstrip()
        print(imgName)
        img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
        hdf5_file["images"][i] = img.reshape(-1)
        hdf5_file["shapes"][i] = img.shape

def save_linemod_foreground(trainlist, validlist, outhdf):
    with open(trainlist, 'r') as file:
        trainfiles = file.readlines()
    with open(validlist, 'r') as file:
        validfiles = file.readlines()

    # open a hdf5 file and create earrays
    hdf5_file = h5py.File(outhdf, mode='w')

    #############################
    hdf5_file.create_dataset("train/images", (len(trainfiles), 480, 640, 4), np.uint8)
    hdf5_file.create_dataset("train/labels/segmentation", (len(trainfiles), 480, 640), np.uint8)
    hdf5_file.create_dataset("train/labels/intrinsics", (len(trainfiles), 3, 3), np.float32)
    hdf5_file.create_dataset("train/labels/poses", (len(trainfiles),), dtype = h5py.special_dtype(vlen=np.dtype('float32')))
    hdf5_file.create_dataset("train/labels/objectsID", (len(trainfiles),), dtype = h5py.special_dtype(vlen=np.dtype('int32')))

    for i in range(len(trainfiles)):
        imgName = trainfiles[i].rstrip()
        print(imgName)

        baseName, extName = os.path.splitext(imgName[imgName.rfind('/') + 1:])
        annotPath = imgName.replace('.png', '.npz')

        img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
        hdf5_file["train/images"][i] = img
        annot = np.load(annotPath)
        hdf5_file["train/labels/segmentation"][i] = annot['segmentation']
        hdf5_file["train/labels/intrinsics"][i] = annot['intrinsics']
        hdf5_file["train/labels/poses"][i] = annot['poses'].reshape(-1)
        hdf5_file["train/labels/objectsID"][i] = annot['objectsID']

    ##########################
    hdf5_file.create_dataset("valid/images", (len(validfiles), 480, 640, 4), np.uint8)
    hdf5_file.create_dataset("valid/labels/segmentation", (len(validfiles), 480, 640), np.uint8)
    hdf5_file.create_dataset("valid/labels/intrinsics", (len(validfiles), 3, 3), np.float32)
    hdf5_file.create_dataset("valid/labels/poses", (len(validfiles),), dtype = h5py.special_dtype(vlen=np.dtype('float32')))
    hdf5_file.create_dataset("valid/labels/objectsID", (len(validfiles),), dtype = h5py.special_dtype(vlen=np.dtype('int32')))

    for i in range(len(validfiles)):
        imgName = validfiles[i].rstrip()
        print(imgName)

        baseName, extName = os.path.splitext(imgName[imgName.rfind('/') + 1:])
        annotPath = imgName.replace('.png', '.npz')

        img = cv2.imread(imgName, cv2.IMREAD_UNCHANGED)
        hdf5_file["valid/images"][i] = img
        annot = np.load(annotPath)
        hdf5_file["valid/labels/segmentation"][i] = annot['segmentation']
        hdf5_file["valid/labels/intrinsics"][i] = annot['intrinsics']
        hdf5_file["valid/labels/poses"][i] = annot['poses'].reshape(-1)
        hdf5_file["valid/labels/objectsID"][i] = annot['objectsID']


if __name__ == "__main__":
    # backfile = '/home/yhu/workspace/data/LINEMOD/backgrounds.txt'
    # # outhdf = '/home/yhu/workspace/backgrounds.h5'
    # outhdf = '/mnt/sda2/backgrounds.h5'
    # save_backgrounds(backfile, outhdf)

    trainfile = '/home/yhu/workspace/data/LINEMOD/synthetic_train.txt'
    validfile = '/home/yhu/workspace/data/LINEMOD/synthetic_valid.txt'
    # outhdf = '/mnt/sda2/linemod.h5'
    outhdf = '/home/yhu/workspace/linemod.h5'
    save_linemod_foreground(trainfile, validfile, outhdf)
