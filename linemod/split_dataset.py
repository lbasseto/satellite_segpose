import os
import random

target_objects = ['ape', 'benchvise', 'cam', 'can',
           'cat', 'driller', 'duck', 'eggbox',
           'glue', 'holepuncher', 'iron', 'lamp', 'phone']

def arrange_list_by_object(allimgfiles, desired_objects):
    with open(allimgfiles, 'r') as file:
        allimgs = file.readlines()
    imgallset = []
    for obj in desired_objects:
        imgobjset = []
        for imgname in allimgs:
            if obj in imgname:
                imgobjset.append(imgname)
        imgallset.append(imgobjset)
    return imgallset

def split_linemod(allimgfiles, out_path, train_propation):
    vaf = open(out_path + '/valid.txt', 'w')
    trf = open(out_path + '/train.txt', 'w')
    tef = open(out_path + '/test.txt', 'w')

    imgset = arrange_list_by_object(allimgfiles, target_objects)
    for i in range(len(imgset)):
        random.shuffle(imgset[i])
        totalCnt = len(imgset[i])
        trainCnt = int(totalCnt * train_propation + 0.5)
        validCnt = int(trainCnt * 0.05 + 0.5)
        trainset = imgset[i][:trainCnt-validCnt]
        validset = imgset[i][trainCnt-validCnt:trainCnt]
        testset = imgset[i][trainCnt:]
        trainset.sort()
        validset.sort()
        testset.sort()

        for f in trainset:
            trf.write(f)
        for f in validset:
            vaf.write(f)
        for f in testset:
            tef.write(f)

if __name__ == "__main__":
    allimgfiles = '/data/LINEMOD/all.txt'
    out_path = '/data/LINEMOD/'
    split_linemod(allimgfiles, out_path, 1.0)
