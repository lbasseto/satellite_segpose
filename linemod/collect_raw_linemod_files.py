import os
import random
from linemod_aux import *

def collect_rgb_images_linemod(linemod_path, outname):
    imgset = []

    objlist = os.listdir(linemod_path + 'objects')
    objlist.sort()

    for obj in objlist:
        if not obj in target_objects:
            continue
        path = linemod_path + 'objects/' + obj + '/rgb/'
        imglist = os.listdir(path)
        imglist.sort()
        for imgpath in imglist:
            imgset.append(path + imgpath)
            print(path + imgpath)

    # write sets
    allf = open(outname, 'w')
    for imgpath in imgset:
        allf.write(imgpath +'\n')

def collect_render_images_linemod(render_img_path, outname):
    imgset = []

    objlist = os.listdir(render_img_path)
    objlist.sort()

    for obj in objlist:
        path = render_img_path + obj + '/'
        if not os.path.isdir(path):
            continue
        # if not obj in target_objects:
        #     continue
        imglist = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.jpg')]
        imglist.sort()
        for imgpath in imglist:
            imgset.append(path + imgpath)
            print(path + imgpath)

    # write sets
    allf = open(outname, 'w')
    for imgpath in imgset:
        allf.write(imgpath +'\n')

def collect_images(path, outname):
    imgs = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    imgs.sort()
    # write sets
    allf = open(outname, 'w')
    for i in imgs:
        allf.write(path + i +'\n')

def collect_images2(datapath, outname):
    filelist = []

    itemlist = os.listdir(datapath)
    itemlist.sort()

    for obj in itemlist:
        path = datapath + obj
        if os.path.isdir(path):
            imgpath = path + '/RGB'
            if not os.path.isdir(imgpath):
                imgpath = path
        else:
            continue

        imgs = [f for f in os.listdir(imgpath) if f.endswith('.jpg') or f.endswith('.png')]
        for i in imgs:
            filelist.append(imgpath + '/' + i)

    filelist.sort()
    # write sets
    allf = open(outname, 'w')
    for i in filelist:
        allf.write(i + '\n')

if __name__ == "__main__":
    # imgpath = '/data/OcclusionChallengeICCV2015/RGB-D/rgb_noseg/'
    # outname = '/data/OcclusionChallengeICCV2015/test.txt'
    # collect_images(imgpath, outname)

    # raw_linemod_path = '/data/LINEMOD/'
    # outname = '/data/LINEMOD/all.txt'
    # collect_rgb_images_linemod(raw_linemod_path, outname)
    #
    # renderimg_path = '/data/LINEMOD/render/'
    # outname = '/data/LINEMOD/all_render.txt'
    # collect_render_images_linemod(renderimg_path, outname)

    #synthetic_train_path = '/data/LINEMOD/synthetic_train/'
    #outname = '/data/LINEMOD/synthetic_train.txt'
    #collect_images(synthetic_train_path, outname)

    #synthetic_valid_path = '/data/LINEMOD/synthetic_valid/'
    #outname = '/data/LINEMOD/synthetic_valid.txt'
    #collect_images(synthetic_valid_path, outname)

    raw_linemod_path = '/cvlabdata1/home/yhu/data/PASCAL3D+_release1.1/PASCAL/VOCdevkit/VOC2012/JPEGImages/'
    outname = '/cvlabdata1/home/yhu/data/backgrounds.txt'
    collect_images(raw_linemod_path, outname)

    # raw_linemod_path = '/home/yhu/data/LINEMOD2/'
    # outname = '/home/yhu/data/LINEMOD/backgrounds2.txt'
    # collect_images2(raw_linemod_path, outname)

    # renderimg_path = '/data/YCB_Video_Dataset/render/'
    # outname = '/data/YCB_Video_Dataset/all_render.txt'
    # collect_render_images_linemod(renderimg_path, outname)
