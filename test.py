import sys
import time
from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet
import scipy.io as sio
import json
from submission import SubmissionWriter
from navpy import dcm2quat
from image import *

fx = 0.0176  # focal length[m]
fy = 0.0176  # focal length[m]
nu = 1920  # number of horizontal[pixels]
nv = 1200  # number of vertical[pixels]
ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
ppy = ppx  # vertical pixel pitch[m / pixel]
fpx = fx / ppx  # horizontal focal length[pixels]
fpy = fy / ppy  # vertical focal length[pixels]
k = [[fpx,   0, nu / 2],
     [0,   fpy, nv / 2],
     [0,     0,      1]]
K_tango = np.array(k)


def evaluate_with_gt_pos(cfgfile, weightfile, listfile, append, bestCnt, conf_thresh, linemod_index=False, use_cuda=True):
    import cv2
    m = Darknet(cfgfile)
    m.print_network()
    m.load_state_dict(torch.load(weightfile))
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    with open(listfile, 'r') as file:
        imglines = file.readlines()
    failed = 0

    for idx in range(len(imglines)):
        imgfile = imglines[idx].rstrip()
        img = cv2.imread(imgfile)
        dirname, filename = os.path.split(imgfile)
        baseName, _ = os.path.splitext(filename)

        dirname = os.path.splitext(dirname[dirname.rfind('/') + 1:])[0]
        outFileName = dirname+'_'+baseName

        start = time.time()
        gtPoses = [None] * 3
        predPose = do_detect(m, img, rawk, gtPoses, bestCnt, conf_thresh, 0, use_cuda)
        finish = time.time()
        print('%s: Predict %d objects in %f seconds.' % (imgfile, len(predPose), (finish - start)))

        name = 'img/' + filename + '.png'
        if (len(predPose)  != 0):
            pose = predPose[0][1]
            r = pose[:, 3]
            if not np.isnan(r[0]):
                quat = np.delete(pose, 3, axis=1).T
                q0,qvec = dcm2quat(quat)
                q = [q0, qvec[0], qvec[1],qvec[2]]
                f = filename[:-3] + 'jpg'
                append(f, q, r)


withFlip = True
jitter = 0.3
withDistortion = True
withRotation = True
imgpath = '/cvlabdata1/cvlab/datasets_kgerard/speed/images/train/img000043.png'


annotPath = imgpath.replace('.png', '.npz')
annot = np.load(annotPath)
segImg = annot['segmentation']
poses = annot['poses']
objsID = annot['objectsID']
rawk = annot['intrinsics']

hue = 0.2
saturation = 1.5
exposure = 1.5
noise = 0.1
smooth = 1
foreImg = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
foreImg = np.stack((foreImg,)*3, axis=-1)
backImg = foreImg
img, label, seg = load_data_detection(backImg, foreImg, segImg, poses, objsID, rawk,
                                 (928, 576), hue, saturation, exposure, noise, smooth, jitter, withFlip, withRotation, withDistortion)

cv2.imwrite('distort_image.png', img)
# YCB-Video
rawk = K_tango
submission_writer = SubmissionWriter()
append_test = submission_writer.append_test
conf= 'cfg/yolov3-vis-YCB-deconv2-608.cfg'
#weights= '/cvlabdata1/home/kgerard/satellite_segpose/logs_YCB_high_res/yolov3-vis-YCB-deconv2-608.cfg/000360.pth'
weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs/satellite_louis.cfg/000123.pth'
test_paths='/cvlabdata1/cvlab/datasets_kgerard/speed/test_paths_to_img.txt'
real_test_paths='/cvlabdata1/cvlab/datasets_kgerard/speed/real_test_paths_to_img.txt'
append_real = submission_writer.append_real_test
evaluate_with_gt_pos(conf, weights,test_paths, append_test, bestCnt=15, conf_thresh=0.1, use_cuda=True)
evaluate_with_gt_pos(conf, weights,real_test_paths, append_real, bestCnt=15, conf_thresh=0.1, use_cuda=True)
submission_writer.export(suffix='submission')
