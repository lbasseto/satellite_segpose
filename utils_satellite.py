import numpy as np
import json
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mplPath
from joblib import Parallel, delayed
from numpy import linalg as LA
import multiprocessing
import time
from navpy import dcm2quat
import math
import cv2
import random
# deep learning framework imports
try:
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.preprocessing import image as keras_image
    has_tf = True
except ModuleNotFoundError:
    has_tf = False

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    has_pytorch = True
except ImportError:
    has_pytorch = False


class Camera:

    """" Utility class for accessing camera parameters. """

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
    K = np.array(k)


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'name': image_ann['filename'], 'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def mat2quater(M):
    tr = torch.trace(M)
    m = M.contiguous().view(1,-1)[0]
    if tr > 0:
        s = torch.sqrt(tr+1.0) * 2
        w = 0.25 * s
        x = (m[7]-m[5]) / s
        y = (m[2]-m[6]) / s
        z = (m[3]-m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = torch.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7]-m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = torch.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = torch.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = torch.stack((w,x,y,z))
    return  Q

def quaternion2rotation(quat):
    '''
    Do not use the quat2dcm() function in the SPEED utils.py, it is not rotation
    '''
    assert (len(quat) == 4)
    # normalize first
    quat = quat / np.linalg.norm(quat)
    a, b, c, d = quat

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)


def project(q, r):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))

        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y


def project_test(pose_mat):
        return project_test_with_k(pose_mat, Camera.K)

def project_test_with_k(pose_mat, k):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = k.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y

def save_img_with_label(img, pose, k, path):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.imshow(img)
    xa, ya = project_test_with_k(pose, k)
    ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
    ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
    ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

    fig.savefig(path)
    plt.close(fig)
def get_img_with_label(img, pose, k):
    fig, ax = plt.subplots( nrows=1, ncols=1 )
    ax.imshow(img)
    xa, ya = project_test_with_k(pose, k)
    ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
    ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
    ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')
    return fig

class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, 'images', split, img_name)
        image = Image.open(img_name).convert('RGB')
        return image

    def get_image_name(self, i=0):
        """ Getting pimage name for image. """
        img_id = self.partitions['train'][i]
        return self.labels[img_id]['name'][0:-4]

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def convert_one_image(self, i, start):
        print(i, 'time remmaining:', ((time.time() - start) * 12000 / (i + 1) ))

        q, r = self.get_pose(i)
        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quaternion2rotation(q)), np.expand_dims(r, 1)))

        #intrisincs
        k=Camera.K

        polys, centers = self.get_polys_and_centers(i)

        width = 1920
        height = 1200
        paths = []
        minx, maxx, miny, maxy = width,0,height,0
        for poly in polys:
            for coord in poly:
                if coord[0] < minx:
                    minx = coord[0]
                if coord[0] > maxx:
                    maxx = coord[0]
                if coord[1] < miny:
                    miny = coord[1]
                if coord[1] > maxy:
                    maxy = coord[1]
            paths.append(mplPath.Path(poly))

        if (miny > 0 and minx > 0) and i > 200:
            return

        seg_img = np.zeros((height, width), dtype=np.uint8)
        for y in range(max(int(miny), 0), min(int(maxy + 1), height)):
            for x in range(max(int(minx), 0), min(int(maxx + 1), width)):
                for path in paths:
                    if path.contains_point((x, y)):
                        seg_img[y][x] = 1
                        break
        outfile = '/cvlabdata1/cvlab/datasets_kgerard/speed/images/train_info/' + self.get_image_name(i) + '.npz'
        np.savez(outfile, segmentation=seg_img, poses=pose_mat, objectsID=np.array([1]), centers=centers, intrinsics=k)
        return

    def convert_input(self):
        start = time.time()
        training_set_size = len(self.partitions['train'])
        print('training_set_size',training_set_size)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.convert_one_image)(i,start) for i in range(training_set_size))



    def get_polys_and_centers(self, i):
        q, r = self.get_pose(i)
        xa, ya = project(q, r)
        red = [(xa[1] - xa[0]), (ya[1] - ya[0])]
        green = [(xa[2] - xa[0]), (ya[2] - ya[0])]
        blue = [(xa[3] - xa[0]), (ya[3] - ya[0])]

        a = [xa[0] + red[0]/2.5 + green[0]/3, ya[0] + red[1]/2.5 + green[1]/3]
        b = [xa[0] + red[0]/2.5 - green[0]/3, ya[0] + red[1]/2.5 - green[1]/3]
        c = [xa[0] - red[0]/2.5 - green[0]/3, ya[0] - red[1]/2.5 - green[1]/3]
        d = [xa[0] - red[0]/2.5 + green[0]/3, ya[0] - red[1]/2.5 + green[1]/3]
        e = [xa[0] + red[0]/1.65 + green[0]/1.8 + blue[0]/3.1, ya[0] + red[1]/1.65 + green[1]/1.8 + blue[1]/3.1]
        f = [xa[0] + red[0]/2.3 - green[0]/1.7 + blue[0]/3.1, ya[0] + red[1]/2.3 - green[1]/1.7 + blue[1]/3.1]
        g = [xa[0] - red[0]/2.3 - green[0]/2.5 + blue[0]/3.1, ya[0] - red[1]/2.3 - green[1]/2.5 + blue[1]/3.1]
        h = [xa[0] - red[0]/1.65 + green[0]/1.8 + blue[0]/3.1, ya[0] - red[1]/1.65 + green[1]/1.8 + blue[1]/3.1]

        rects = [[e,f,g,h,e], [a,b,c,d,a], [a,b,f,e,a],[a,d,h,e,a],[c,b,f,g,c],[c,d,h,g,c]]

        centers = np.array([[xa[0], ya[0]]])

        return rects, centers



if has_pytorch:
    class PyTorchSatellitePoseEstimationDataset(Dataset):

        """ SPEED dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, split='train', speed_root='', transform=None):

            if not has_pytorch:
                raise ImportError('Pytorch was not imported successfully!')

            if split not in {'train', 'test', 'real_test'}:
                raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

            with open(os.path.join(speed_root, split + '.json'), 'r') as f:
                label_list = json.load(f)

            self.sample_ids = [label['filename'] for label in label_list]
            self.train = split == 'train'

            if self.train:
                self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                               for label in label_list}
            self.image_root = os.path.join(speed_root, 'images', split)

            self.transform = transform

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, idx):
            sample_id = self.sample_ids[idx]
            img_name = os.path.join(self.image_root, sample_id)

            # note: despite grayscale images, we are converting to 3 channels here,
            # since most pre-trained networks expect 3 channel input
            pil_image = Image.open(img_name).convert('RGB')

            if self.train:
                q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
                y = np.concatenate([q, r])
            else:
                y = sample_id

            if self.transform is not None:
                torch_image = self.transform(pil_image)
            else:
                torch_image = pil_image

            return torch_image, y
else:
    class PyTorchSatellitePoseEstimationDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError('Pytorch is not available!')



def get_vertices(q,r):
    xa, ya = project(q, r)
    red = [(xa[1] - xa[0]), (ya[1] - ya[0])]
    green = [(xa[2] - xa[0]), (ya[2] - ya[0])]
    blue = [(xa[3] - xa[0]), (ya[3] - ya[0])]

    a = [xa[0] + red[0]/2.5 + green[0]/3, ya[0] + red[1]/2.5 + green[1]/3]
    b = [xa[0] + red[0]/2.5 - green[0]/3, ya[0] + red[1]/2.5 - green[1]/3]
    c = [xa[0] - red[0]/2.5 - green[0]/3, ya[0] - red[1]/2.5 - green[1]/3]
    d = [xa[0] - red[0]/2.5 + green[0]/3, ya[0] - red[1]/2.5 + green[1]/3]
    e = [xa[0] + red[0]/1.65 + green[0]/1.8 + blue[0]/3.1, ya[0] + red[1]/1.65 + green[1]/1.8 + blue[1]/3.1]
    f = [xa[0] + red[0]/2.3 - green[0]/1.7 + blue[0]/3.1, ya[0] + red[1]/2.3 - green[1]/1.7 + blue[1]/3.1]
    g = [xa[0] - red[0]/2.3 - green[0]/2.5 + blue[0]/3.1, ya[0] - red[1]/2.3 - green[1]/2.5 + blue[1]/3.1]
    h = [xa[0] - red[0]/1.65 + green[0]/1.8 + blue[0]/3.1, ya[0] - red[1]/1.65 + green[1]/1.8 + blue[1]/3.1]

    vertices = [a,b,c,d,e,f,g,h]

    centers = np.array([[xa[0], ya[0]]])
    return vertices, centers

def calculate_loss(truth, actual):
    score_position = LA.norm(truth[1] - actual[1]) / LA.norm(truth[1])
    score_pose = 2 * np.arccos(min(1,abs(truth[0].dot(actual[0]))))
    return score_position + score_pose, score_position, score_pose

def get_sat_bound(q, r, width, height):
    vertices, centers = get_vertices(q,r)
    minx, maxx, miny, maxy = width,0,height,0
    for coord in vertices:
        if coord[0] < minx:
            minx = coord[0]
        if coord[0] > maxx:
            maxx = coord[0]
        if coord[1] < miny:
            miny = coord[1]
        if coord[1] > maxy:
            maxy = coord[1]

    return minx, maxx, miny, maxy

def zoom_on_satellite(img, segImg, pose, k, bounds, with_fixed_border=False, border_size=0):
    width = img.shape[1]
    height = img.shape[0]
    smallest_Res = min(width, height)
    biggest_Res = max(width, height)

    img = cv2.copyMakeBorder(img, 0, biggest_Res - smallest_Res, 0, 0, cv2.BORDER_CONSTANT)
    segImg = cv2.copyMakeBorder(segImg, 0, biggest_Res - smallest_Res, 0, 0, cv2.BORDER_CONSTANT)

    width = img.shape[1]
    height = img.shape[0]

    minx, maxx, miny, maxy = bounds[0], bounds[1], bounds[2], bounds[3]

    sat_width = math.ceil(maxx - minx)
    sat_height = math.ceil(maxy - miny)
    biggest_dim = max(sat_width, sat_height)

    left_border = random.randint(int(biggest_dim/9), int(biggest_dim/5))
    top_border = random.randint(int(biggest_dim/9), int(biggest_dim/5))
    new= random.randint(max(left_border + sat_width, top_border + sat_height), int(biggest_dim*1.25))

    if with_fixed_border:
        left_border = int(biggest_dim /border_size)
        top_border = left_border
        new = top_border * 2 + biggest_dim


    new= min(new, biggest_Res)

    pleft  = max(0, minx - left_border)
    ptop   = max(0, miny - top_border)
    pright = -pleft
    pbot = -ptop
    sx = 1
    sy = 1
    rM2 = np.array([[1.0, 0.0, -pleft], [0.0, 1.0, -ptop], [0.0, 0.0, 1.0]])  # translation
    img = cv2.warpAffine(img, rM2[:2], (width, height))
    segImg = cv2.warpAffine(segImg, rM2[:2], (width, height))
    dx = float(pleft)/ width
    dy = float(ptop) / height
    img = img[0:new, 0:new]
    segImg = segImg[0:new, 0:new]
    cw = width
    ch = height

    # compute the new K according to transform
    trans = np.array([[1, 0, -cw * dx],
                      [0, 1, -ch * dy],
                      [0, 0, 1]])
    k = np.matmul(trans, k)  # new intrinsic

    return img, segImg, k


def old_zoom_on_satellite(img, q, r, rawk):
    width = img.shape[1]
    height = img.shape[0]
    minx, maxx, miny, maxy =    (q, r, width, height)

    sat_width = maxx - minx
    sat_height = maxy - miny
    border = 50
    #border = 30
    new_width = max(2*border + sat_width, 928)
    new_height = max(2*border + sat_height,576)
    ratio = max(new_width/width, new_height/height)
    new_width = math.ceil(ratio*width)
    new_height = math.ceil(ratio*height)

    pleft  = max(0, minx - (new_width-sat_width)/2)
    ptop   = max(0, miny - (new_height-sat_height)/2)


    pright = -pleft
    pbot = -ptop
    sx = 1
    sy = 1

    rM2 = np.array([[1.0, 0.0, -pleft], [0.0, 1.0, -ptop], [0.0, 0.0, 1.0]])  # translation
    img = cv2.warpAffine(img, rM2[:2], (width, height))

    dx = float(pleft)/ width
    dy = float(ptop) / height

    img = img[:new_height, :new_width]
    cw = width
    ch = height

    # compute the new K according to transform
    trans = np.array([[1, 0, -cw * dx],
                      [0, 1, -ch * dy],
                      [0, 0, 1]])
    k = np.matmul(trans, rawk)  # new intrinsic

    return img, k
