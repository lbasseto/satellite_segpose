import sys
import time
from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet

# intrinsics of LINEMOD dataset
k_linemod = np.array([[572.41140, 0.0, 325.26110],
                 [0.0, 573.57043, 242.04899],
                 [0.0, 0.0, 1.0]])

vertex = np.load('/data/YCB_Video_Dataset/model_vertex/YCB_vertex.npy')
# vertex = np.load('/data/LINEMOD/models_vertex/LINEMOD_vertex.npy')
use_cuda = 0

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    # m.load_weights(weightfile)
    m.load_state_dict(torch.load(weightfile))
    print('Loading weights from %s... Done!' % (weightfile))

    num_classes = 20
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)

    rawk = k_linemod
    gtObjsID = None
    gtPoses = None

    baseName, extName = os.path.splitext(imgfile[imgfile.rfind('/') + 1:])
    annotPath = imgfile.replace('.png', '.npz').replace('.jpg', '.npz')
    if os.path.exists(annotPath):
        annot = np.load(annotPath)
        segImg = annot['segmentation']
        gtPoses = annot['poses']
        gtObjsID = annot['objectsID']
        rawk = annot['intrinsics']

        # show gt
        gtPoseImg = np.copy(img)
        objCnt = len(gtObjsID)
        for i in range(objCnt):
            id = int(gtObjsID[i])
            gt_rt = gtPoses[i]
            # show vertex reprojection
            vp = vertices_reprojection(vertex[id][:500], gt_rt, rawk)
            for p in vp:
                gtPoseImg = cv2.circle(gtPoseImg, (int(p[0]), int(p[1])), 2, get_class_colors(id), -1)
            # show axis
            gtPoseImg = draw_axis(gtPoseImg, rawk, gt_rt, 0.05)
        cv2.imshow("img", img)
        segImg = cv2.normalize(segImg, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("gt label", segImg)
        cv2.imshow("gt", gtPoseImg)
        # cv2.waitKey(0)

    start = time.time()
    predPose = do_detect(m, img, rawk, None, 5, 0.3, 300, use_cuda)
    finish = time.time()
    print('%s: Predict %d objects in %f seconds.' % (imgfile, len(predPose), (finish - start)))

    # show predictions
    visImg = visualize_predictions(predPose, img, vertex, rawk)
    cv2.imshow("pred", visImg)
    cv2.waitKey(0)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

detect_cv2('cfg/yolov3-vis-YCB-deconv2-bbox-u.cfg', 'logs_YCB_L1_param2/yolov3-vis-YCB-deconv2-bbox-u.cfg/000010.pth',
           '/home/yhu/workspace/data/YCB_Video_Dataset/data/0065/000083-color.png')

detect_cv2('cfg/yolov3-vis-LINEMOD-deconv2-bbox-u.cfg', 'logs_LINEMOD_L1/yolov3-vis-LINEMOD-deconv2-bbox-u.cfg/000200.pth',
           '/home/yhu/workspace/data/OcclusionChallengeICCV2015/RGB-D/rgb_noseg/color_00781.png')

exit(0)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
