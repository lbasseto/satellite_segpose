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
            # p2dx = curr_gtx * xyscale
            # p2dy = curr_gty * xyscale
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

def build_targets(predx, predy, keypoints, target, anchors, anchor_step, class_weight,
                  num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    label = target[0].data
    gtseg = target[1].data

    nB = label.size(0)
    nA = int(len(anchors) / anchor_step)
    nC = num_classes
    nV = keypoints.shape[1]
    # conf_scale  = torch.ones(nB, nA, nH, nW) * noobject_scale
    conf_mask = torch.zeros(nB, nA, nH, nW, nV)
    coord_mask = torch.zeros(nB, nA, nH, nW, nV)
    tx         = torch.zeros(nB, nA, nH, nW, nV)
    ty         = torch.zeros(nB, nA, nH, nW, nV)
    tconf      = torch.zeros(nB, nA, nH, nW, nV)
    tcls       = torch.zeros(nB, nA, nH, nW) # zero for background

    cellWidth = int(gtseg.shape[2] / nW)
    cellHeight = int(gtseg.shape[1] / nH)
    subgtseg = gtseg[:, int(cellHeight / 2 + 0.5)::cellHeight, int(cellWidth / 2 + 0.5)::cellWidth]
    # debug
    # if True:
    if False:
        segimg = seg[0]
        segimg = cv2.normalize(segimg, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("seg", segimg)
        cv2.waitKey(0)

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

            labelmask = (subgtseg[b] == t + 1)
            labelCnt = int(labelmask.sum())
            if labelCnt < 1:
                continue

            tcls[b][best_n][labelmask] = id+1

            tx[b][best_n][labelmask] = gtx
            ty[b][best_n][labelmask] = gty
            tconf[b][best_n][labelmask] = reproj_confidence1(predx[b][best_n][labelmask], predy[b][best_n][labelmask], tx[b][best_n][labelmask], ty[b][best_n][labelmask])

            #
            coord_mask[b][best_n][labelmask] = class_weight[id+1]
            conf_mask[b][best_n][labelmask] = class_weight[id+1]

    nRecall = int((tconf > 0.5).sum().data.item())
    return nGT, nRecall, coord_mask, conf_mask, tx, ty, tconf, tcls

class Pose2DLayer(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1, vpoints=[], cpoints=[],
                 class_weights=[], keypointsfile=[],
                 num_keypoints=1, alpha_class=1, alpha_coord=1, alpha_conf=1):
        super(Pose2DLayer, self).__init__()
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
        self.conf_scale = float(alpha_conf)
        self.mtm_scale = 3000
        self.thresh = 0.6
        self.stride = 32

        self.class_weight = torch.from_numpy(class_weights).float()
        self.cpoints = torch.from_numpy(cpoints).float()

        self.keypoints = torch.from_numpy(np.load(keypointsfile)).float()
        self.num_keypoints = num_keypoints
        self.keypoints = self.keypoints[:,:num_keypoints,:]

    def forward(self, output, target, param = None):
        if param:
            seen = param[0]

        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nV = self.num_keypoints
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB * nA, (3 * nV), nH * nW).transpose(0, 1). \
            contiguous().view((3 * nV), nB * nA * nH * nW)

        conf = torch.sigmoid(output[0:nV].transpose(0, 1).view(nB, nA, nH, nW, nV))
        x = output[nV:2*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)
        y = output[2*nV:3*nV].transpose(0, 1).view(nB, nA, nH, nW, nV)

        t1 = time.time()

        grid_x = ((torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nW ) * self.coord_norm_factor
        grid_y = ((torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA * nV, 1, 1). \
            view(nB, nA, nV, nH, nW).type_as(output) + 0.5) / nH) * self.coord_norm_factor
        grid_x = grid_x.permute(0, 1, 3, 4, 2).contiguous()
        grid_y = grid_y.permute(0, 1, 3, 4, 2).contiguous()


        predx = x + grid_x
        predy = y + grid_y

        if self.training:
            #
            predx = convert2cpu(predx)
            predy = convert2cpu(predy)
            grid_x = convert2cpu(grid_x)
            grid_y = convert2cpu(grid_y)
            conf = convert2cpu(conf)

            nGT, nRecall, coord_mask, conf_mask, tx, ty, tconf, tcls = \
                build_targets(predx.data/self.coord_norm_factor, predy.data/self.coord_norm_factor,
                              self.keypoints, target, self.anchors, self.anchor_step, self.class_weight,
                              nC, nH, nW, self.noobject_scale, self.object_scale, self.thresh, seen)
            assert(nGT > 0)

            nProposals = int((conf > 0.5).sum().data.item())
            meanConf = tconf[conf_mask>0].mean().data.item()

            tx    = Variable(tx.type_as(predx))
            ty    = Variable(ty.type_as(predy))
            tconf = Variable(tconf.type_as(conf))

            t3 = time.time()

            # L2Loss = True
            L2Loss = False
            if L2Loss:
                coord_mask = Variable(coord_mask.type_as(predx).sqrt())
                conf_mask = Variable(conf_mask.type_as(conf).sqrt())

                loss_x = self.coord_scale * nn.SmoothL1Loss(size_average=False)(predx*coord_mask, self.coord_norm_factor*tx*coord_mask)
                loss_y = self.coord_scale * nn.SmoothL1Loss(size_average=False)(predy*coord_mask, self.coord_norm_factor*ty*coord_mask)
                loss_conf = self.conf_scale * nn.SmoothL1Loss(size_average=False)(self.coord_norm_factor*conf*conf_mask, self.coord_norm_factor*tconf*conf_mask)
            else:
                coord_mask = Variable(coord_mask.type_as(predx))
                conf_mask = Variable(conf_mask.type_as(conf))

                loss_x = self.coord_scale * nn.L1Loss(size_average=False)(predx*coord_mask, self.coord_norm_factor*tx*coord_mask)
                loss_y = self.coord_scale * nn.L1Loss(size_average=False)(predy*coord_mask, self.coord_norm_factor*ty*coord_mask)
                loss_conf = self.conf_scale * nn.L1Loss(size_average=False)(self.coord_norm_factor*conf*conf_mask, self.coord_norm_factor*tconf*conf_mask)
            loss = loss_x + loss_y + loss_conf

            return loss, [loss.data.item(),
                          loss_conf.data.item(),
                          (loss_x + loss_y).data.item(),
                          meanConf, nProposals, nRecall,
                          nGT]
        else:
            predx = predx.view(nB, nH, nW, nV) / self.coord_norm_factor
            predy = predy.view(nB, nH, nW, nV) / self.coord_norm_factor

            # copy to CPU
            conf = convert2cpu(conf.view(nB,nH,nW,nV)).detach().numpy()
            px = convert2cpu(predx).detach().numpy()
            py = convert2cpu(predy).detach().numpy()
            keypoints = convert2cpu(self.keypoints).detach().numpy()


            out_preds = [px, py, conf, keypoints]
            return out_preds
