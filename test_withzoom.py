import sys
import time
from PIL import Image, ImageDraw
from image import *
from utils import *
from utils_satellite import *
from darknet import Darknet
import scipy.io as sio
import json
from submission import SubmissionWriter
from navpy import dcm2quat
from numpy import linalg as LA
import random

from cfg import parse_cfg

K_tango = Camera.K
losses =  []

def evaluate_with_gt_pos(cfgfile, weightfile, listfile, append, bestCnt, withZoom=True, use_cuda=True, zoom_type=None):
    import cv2

    all_channels = [32, 64, 32, 64, 128, 64, 128, 64, 128, 256, 128, 256, 128, 256, 128, 256, 128, 256, 128, 256, 128, 256, 128, 256, 128, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 256, 512, 1024, 512, 1024, 512, 1024, 512, 1024, 512, 1024, 512, 1024, 512, 1024, 512, 256, 256, 512, 256, 512, 256, 128, 128, 256, 128, 256, 128, 256, 2, 512, 1024, 512, 1024, 512, 256, 256, 512, 256, 512, 256, 128, 128, 256, 128, 256, 128, 256, 24]

    m = Darknet(cfgfile, all_channels)
    m.print_network()
    m.load_state_dict(torch.load(weightfile))
    print('Loading weights from %s... Done!' % (weightfile))
    if use_cuda:
        m.cuda()

    m.print_bn_weights()

    with open(listfile, 'r') as file:
        imglines = file.readlines()

    failed_pred = 0
    total_pred = 0
    for idx in range(len(imglines)):
        max_conf = 0
        imgfile = imglines[idx].rstrip()
        img = cv2.imread(imgfile)
        dirname, filename = os.path.split(imgfile)

        baseName, _ = os.path.splitext(filename)
        dirname = os.path.splitext(dirname[dirname.rfind('/') + 1:])[0]
        outFileName = dirname+'_'+baseName

        start = time.time()
        gtPoses = [None] * 3
        rawk = K_tango
        target_shape=(704, 704)

        max_conf = 0
        best_pred = None
        best_border = None
        save=False

        print('imgfile', imgfile)
        print(str(failed_pred) + ' ' + str(total_pred))
        #predPose, conf, deviation, p2d = do_detect(m, img, rawk, gtPoses, bestCnt, 0, 0, use_cuda)
        try:
            total_pred = total_pred + 1
            predPose, conf, p2d = do_detect(m, img, rawk, gtPoses, bestCnt, 0, 0, use_cuda)
        except Exception:
            failed_pred = failed_pred + 1
            pass

        finish = time.time()
        name = 'img/' + filename + '.png'
        if predPose is not None and (len(predPose)  != 0):
            pose = predPose[0][1]
            print(str(pose))
            r = pose[:, 3]

            name = 'img/' + filename + '_' + str(conf) + '.png'
            #save_img_with_label(img, pose, rawk, name)
            print(name)
            quat = np.delete(pose, 3, axis=1).T
            q0,qvec = dcm2quat(quat)
            q = [q0, qvec[0], qvec[1],qvec[2]]
            print(conf)
        else:
            name = 'img/missing' + filename + '.png'
            print('problem', name, save)
    print(str(failed_pred) + ' ' + str(total_pred))



final_predictions = {}

# YCB-Video
rawk = K_tango
submission_writer = SubmissionWriter()
append_test = submission_writer.append_test
conf= 'cfg/satellite.cfg'
old_conf= 'cfg/old_satellite.cfg'
high_conf= 'cfg/sat_high_res.cfg'
#weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_hd_001_ind53_bigbackgroundweight/satellite_louis.cfg/000052.pth'
#weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_hd_001_ind53to100/satellite_louis.cfg/000065.pth'
#weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_hd_001/satellite_louis.cfg/000101.pth'
#weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_bnpruning/satellite_louis.cfg/000105.pth'
#weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_hd_01/satellite_louis.cfg/000095.pth'
weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_bnpruning_lr1e-5_fromzero/satellite_louis.cfg/000130.pth'
#weights= '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_hd_01_ind53/satellite_louis.cfg/000065.pth'
#weights= '/cvlabdata1/home/kgerard/satellite_segpose/logs_kube/satellite.cfg/000135.pth'
#weights = '/cvlabdata2/home/basseto/satellite_segpose/louis_logs_bnpruning/satellite_louis.cfg/000005.pth'
#old weights= '/cvlabdata1/home/kgerard/satellite_segpose/satellite_logs/satellite.cfg/000080.pth'
unzoomed_weights= '/cvlabdata1/home/kgerard/satellite_segpose/saved_weights/000360.pth'
zoomed_weights= '/cvlabdata1/home/kgerard/satellite_segpose/saved_weights/000240.pth'
test_paths='/cvlabdata1/cvlab/datasets_kgerard/speed/test_paths_to_img.txt'
train_path='/cvlabdata1/cvlab/datasets_kgerard/speed/full_train_paths_to_img.txt'
valid_path='/cvlabdata1/cvlab/datasets_kgerard/speed/valid_train_paths_to_img.txt'
real_test_paths='/cvlabdata1/cvlab/datasets_kgerard/speed/real_test_paths_to_img.txt'
validation_path='/cvlabdata1/cvlab/datasets_kgerard/speed/valid_train_paths_to_img.txt'
append_real = submission_writer.append_real_test
evaluate_with_gt_pos(conf, weights, valid_path, append_test, bestCnt=10000, use_cuda=True, withZoom=True)
#evaluate_with_gt_pos(old_conf, unzoomed_weights, test_paths, append_test, bestCnt=15, conf_thresh=0.9, use_cuda=True, withZoom=False)
#evaluate_with_gt_pos(old_conf, zoomed_weights, train_path, append_test, bestCnt=15, conf_thresh=0.9, use_cuda=True, withZoom=True, zoom_type='old')

submission_writer.export(suffix='29_new')
