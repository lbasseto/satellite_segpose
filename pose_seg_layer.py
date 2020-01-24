import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
from PnP import *
from FocalLoss import *

def ComputePnPLoss(cpoints, keypoints, target, px, py):
    cpoints = cpoints.type_as(px)
    keypoints = keypoints.type_as(px)
    label = target[0].data.type_as(px)
    gtseg = target[1].data

    nB, nA, nH, nW, nV = px.shape

    cellWidth = int(gtseg.shape[2] / nW)
    cellHeight = int(gtseg.shape[1] / nH)
    subgtseg = gtseg[:, int(cellHeight / 2 + 0.5)::cellHeight, int(cellWidth / 2 + 0.5)::cellWidth]


    nGT = 0
    mtmloss = 0
    pnploss = 0
    evloss = 0
    for b in range(nB):
        for t in range(label.shape[1]):
            id = int(label[b][t][0])
            if id < 0:
                break
            nGT = nGT + 1

            width = label[b][t][3]
            height = label[b][t][4]
            curr_k = label[b][t][5:14].view(-1, 3)
            cur_gt_rt = label[b][t][14:26].view(-1, 4)
            cur_gt_r = cur_gt_rt[:, 0:3]
            cur_gt_t = cur_gt_rt[:, 3].view(-1, 1)

            curr_gtv = (cpoints.mm(cur_gt_r.t()) + cur_gt_t.view(1, -1)).view(4, 3)
            curr_gtv = curr_gtv / curr_gtv.norm()
            zpos = 10
            gt3dx = zpos * curr_gtv[:, 0] / curr_gtv[:, 2]
            gt3dy = zpos * curr_gtv[:, 1] / curr_gtv[:, 2]

            labelmask = (subgtseg[b] == t + 1)
            labelCnt = int(labelmask.sum())
            if labelCnt < 1:
                continue

            # GT reprojection
            p = curr_k.mm(cur_gt_r.mm(keypoints[id].t()) + cur_gt_t)
            curr_gtx = (p[0] / p[2]) / width
            curr_gty = (p[1] / p[2]) / height
            curr_gtx = curr_gtx.repeat(labelCnt, 1)
            curr_gty = curr_gty.repeat(labelCnt, 1)

            # construct p3d
            p3d = keypoints[id].view(1,-1).repeat(labelCnt,1).view(-1, 3)
            p2dx = px[b][0][labelmask]
            p2dy = py[b][0][labelmask]
            # # for debug
            mtm = ComputeMtM(cpoints, curr_k,
                             torch.cat(((p2dx * width).view(-1, 1), (p2dy * height).view(-1, 1)), 1),
                             p3d, None)
            curr_mtmloss = curr_gtv.view(-1, 1).t().mm(mtm).mm(curr_gtv.view(-1, 1)) / (2 * len(p3d))

            curr_pnploss = curr_mtmloss
            curr_evloss = curr_mtmloss

            pnploss = pnploss + curr_pnploss
            evloss = evloss + curr_evloss
            mtmloss = mtmloss + curr_mtmloss

    pnploss = pnploss / nGT
    evloss = evloss / nGT
    mtmloss = mtmloss / nGT
    return pnploss, evloss, mtmloss

def ComputeNormReprojectionLoss(keypoints, target, px, py):
    keypoints = keypoints.type_as(px)
    label = target[0].data.type_as(px)
    gtseg = target[1].data

    nB, nA, nH, nW, nV = px.shape

    cellWidth = int(gtseg.shape[2] / nW)
    cellHeight = int(gtseg.shape[1] / nH)
    subgtseg = gtseg[:, int(cellHeight / 2 + 0.5)::cellHeight, int(cellWidth / 2 + 0.5)::cellWidth]

    nGT = 0
    repXloss = 0
    repYloss = 0
    for b in range(nB):
        for t in range(label.shape[1]):
            id = int(label[b][t][0])
            if id < 0:
                break
            nGT = nGT + 1

            width = label[b][t][3]
            height = label[b][t][4]
            curr_k = label[b][t][5:14].view(-1, 3)
            cur_gt_rt = label[b][t][14:26].view(-1, 4)
            cur_gt_r = cur_gt_rt[:, 0:3]
            cur_gt_t = cur_gt_rt[:, 3].view(-1, 1)

            labelmask = (subgtseg[b] == t + 1)
            labelCnt = int(labelmask.sum())
            if labelCnt < 1:
                continue

            # GT reprojection
            p = curr_k.mm(cur_gt_r.mm(keypoints[id].t()) + cur_gt_t)
            curr_gtx = (p[0] / p[2]) / width
            curr_gty = (p[1] / p[2]) / height
            curr_gtx = curr_gtx.repeat(labelCnt, 1)
            curr_gty = curr_gty.repeat(labelCnt, 1)

            p2dx = px[b][0][labelmask]
            p2dy = py[b][0][labelmask]

            curr_xloss = (curr_gtx - p2dx).pow(2).mean()
            curr_yloss = (curr_gty - p2dy).pow(2).mean()

            repXloss = repXloss + curr_xloss
            repYloss = repYloss + curr_yloss

    repXloss = repXloss / nGT
    repYloss = repYloss / nGT
    return repXloss + repYloss

def build_targets(keypoints, target, anchors, anchor_step, coord_weight,
                  num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    label = target[0].data
    gtseg = target[1].data

    nB = label.size(0)
    nA = int(len(anchors) / anchor_step)
    nC = num_classes
    nV = keypoints.shape[1]
    conf_scale  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW, nV)
    tx         = torch.zeros(nB, nA, nH, nW, nV)
    ty         = torch.zeros(nB, nA, nH, nW, nV)
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) # background for default
    cls_mask   = torch.ones(nB, nA, nH, nW) # all positions

    cellWidth = int(gtseg.shape[2] / nW)
    cellHeight = int(gtseg.shape[1] / nH)
    subgtseg = gtseg[:, int(cellHeight / 2 + 0.5)::cellHeight, int(cellWidth / 2 + 0.5)::cellWidth]

    #
    nGT = 0
    for b in range(nB):
        for t in range(label.shape[1]):
            id = int(label[b][t][0])
            if id < 0:
                break
            nGT = nGT + 1

            width = label[b][t][3]
            height = label[b][t][4]
            curr_k = label[b][t][5:14].view(-1, 3)
            cur_gt_rt = label[b][t][14:26].view(-1, 4)

            # reprojection
            p = curr_k.mm(cur_gt_rt[:, 0:3].mm(keypoints[id].t()) + cur_gt_rt[:, 3].view(-1, 1))
            gtx = (p[0] / p[2]) / width
            gty = (p[1] / p[2]) / height

            best_n = 0
            if False:
                # compute bounding box from virtual points
                gbbw = gtx.max() - gtx.min()
                gbbh = gty.max() - gty.min()

                # find the best anchor for the ground truth
                gt_box = [0, 0, gbbw, gbbh]
                best_iou = 0.0
                best_n = -1
                for n in range(nA):
                    aw = anchors[int(anchor_step*n)]
                    ah = anchors[int(anchor_step*n+1)]
                    anchor_box = [0, 0, aw, ah]
                    iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n

            labelmask = (subgtseg[b] == t + 1)
            labelCnt = int(labelmask.sum())
            if labelCnt < 1:
                continue

            tcls[b][best_n][labelmask] = id + 1
            cls_mask[b][best_n][labelmask] = 1

            tx[b][best_n][labelmask] = gtx
            ty[b][best_n][labelmask] = gty

    cls_mask = (cls_mask==1)
    nRecall = int((tconf > 0.5).sum().data.item())
    return nGT, nRecall, coord_mask, conf_scale, tx, ty, tconf, tcls, cls_mask

class PoseSegLayer(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1, vpoints=[], cpoints=[],
                 class_weights=[], keypointsfile=[],
                 num_keypoints=1, alpha_class=1, alpha_coord=1):
        super(PoseSegLayer, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = int(len(anchors)/num_anchors)
        self.coord_scale = float(alpha_coord)
        self.coord_norm_factor = 10
        self.noobject_scale = 10
        self.object_scale = 50
        self.class_scale = float(alpha_class)
        self.pnp_scale = 1
        self.mtm_scale = 3000
        self.thresh = 0.6
        self.stride = 32
        self.class_weight = torch.from_numpy(class_weights).float()
        self.coord_weight = self.class_weight[1:]
        self.cpoints = torch.from_numpy(cpoints).float()
        self.keypoints = torch.from_numpy(np.load(keypointsfile)).float()
        self.num_keypoints = num_keypoints
        self.keypoints = self.keypoints[:,:num_keypoints,:]

    def forward(self, output, target, param = None):
        if param:
            seen = param[0]

        nB = output.data.size(0)
        nA = self.num_anchors
        assert(nA == 1)
        nC = self.num_classes
        nV = self.num_keypoints
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB * nA, (nC), nH * nW).transpose(0, 1). \
            contiguous().view((nC), nB * nA * nH * nW)
        cls = output[0:nC].transpose(0, 1)

        if self.training:
            cls = convert2cpu(cls)

            nGT, nRecall, coord_mask, conf_scale, tx, ty, tconf, tcls, cls_mask = \
                build_targets(self.keypoints, target, self.anchors, self.anchor_step, self.coord_weight,
                              nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, seen)
            assert(nGT > 0)

            tcls  = Variable(tcls[cls_mask].type_as(cls).view(-1).long())

            class_weight = Variable(self.class_weight.type_as(cls))

            cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC))
            cls        = cls[cls_mask].view(-1, nC)

            loss_cls = self.class_scale * FocalLoss(class_num=nC, alpha=class_weight, gamma=2, size_average=False)(cls, tcls)

            loss = loss_cls

            return loss, [loss.data.item(), loss_cls.data.item(), nGT, 0, 0, 0, 0]
        else:
            print('cls', cls)
            cls_confs, cls_ids = torch.max(F.softmax(cls, 1), 1)

            cls_confs = cls_confs.view(nB, nH, nW)
            cls_ids = cls_ids.view(nB, nH, nW)


            # copy to CPU
            cls_confs = convert2cpu(cls_confs).detach().numpy()
            cls_ids = convert2cpu_long(cls_ids).detach().numpy()


            out_preds = [cls_confs, cls_ids]
            return out_preds
