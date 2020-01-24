import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pose_2d_layer import Pose2DLayer
from pose_seg_layer import PoseSegLayer
from utils import convert2cpu
from cfg import *


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, int(H/hs), hs, int(W/ws), ws).transpose(3,4).contiguous()
        x = x.view(B, C, int(H/hs*W/ws), hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, int(H/hs), int(W/ws)).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, int(H/hs), int(W/ws))
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, cfgfile, channels):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks, channels) # merge conv, bn,leaky
        self.loss = None
        self.interParam = []

        self.no_reg_loss = 0.0
        self.l1_reg_only = 0.0
        # Add regularization to batchnorm weights
        self.bn_weight_params = None
        self.bn_regularization_lambda = 0

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        self.header = torch.IntTensor([0,0,0,0,0])
        self.seen = 0

    def forward(self, x, y = None):
        ind = -2
        self.loss = None
        self.interParam = []

        self.bn_weight_params = []

        outputs = dict()
        out_predicts = []
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'deconvolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)

                if block['type'] == 'convolutional'  and ind >= 53:
                    for module_name, module in self.models[ind].named_children():
                        if module_name.startswith('bn'):
                            for param_name, param in module.named_parameters():
                                if param_name == 'weight':
                                    self.bn_weight_params.append(convert2cpu(param))

                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                layerlen = len(layers)
                assert (layerlen >= 1)
                x = outputs[layers[0]]
                if layerlen > 1:
                    for i in range(1, layerlen):
                        x = torch.cat((x,outputs[layers[i]]), 1)
                outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] in ['yolo', 'pose', 'pose-2d', 'pose-ind', 'pose-part', 'pose-seg', 'pose-3dr', 'pose-3drseg', 'pose-pnp']:
                layerId = ("L%03d" % int(ind))
                if self.training:
                    loss, param = self.models[ind](x, y, [self.seen])
                    #self.no_reg_loss = loss.item()

                    # Compute regularization
                    #all_bn_weights = torch.cat([x.view(-1) for x in self.bn_weight_params])
                    #l1_regularization = torch.norm(all_bn_weights, 1)
                    #self.l1_reg_only = l1_regularization.item()

                    #loss = torch.add(loss, l1_regularization, alpha=self.bn_regularization_lambda)

                    #print('no_reg_loss: ' + str(self.no_reg_loss) + '\tl1_reg_only: ' + str(self.l1_reg_only), '\tregularized_loss: ' + str(self.regularized_loss))
                    if self.loss:
                        self.loss = self.loss + loss
                    else:
                        self.loss = loss
                    self.interParam.append([layerId, block['type'], param])
                else:
                    pred = self.models[ind](x, None)
                    out_predicts.append([layerId, block['type'], pred])
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        if self.training:
            return self.loss, self.interParam
        else:
            return out_predicts

    def print_network(self):
        print_cfg(self.blocks)

    def print_bn_weights(self):
        ind = -2
        min_weight = 100
        with open('bn_weights.txt', 'r+') as f:
            for block in self.blocks:
                ind = ind + 1
                if block['type'] == 'convolutional':
                    for module_name, module in self.models[ind].named_children():
                        if module_name.startswith('bn'):
                            for param_name, param in module.named_parameters():
                                if param_name == 'weight':
                                    min_weight = min(min_weight, torch.min(param).item())
                                    weight_str = str(ind) + ' ' + module_name + ' ' + param_name + ' ' + str(param)
                                    print(weight_str)
                                    f.write(weight_str + '\n')
                                    #self.bn_weight_params.append(convert2cpu(param))
            f.write('min_weight ' + str(min_weight))



    def create_network(self, blocks, channels):
        models = nn.ModuleList()
        ind = -2
        prev_filters = 3
        out_filters =[]
        prev_stride = 1
        out_strides = []
        conv_id = 0
        deconv_id = 0
        for block in blocks:
            ind = ind + 1
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] in ['convolutional', 'deconvolutional']:
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = int((kernel_size-1)/2) if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                namesuffix = None
                if block['type'] == 'convolutional':
                    conv_id = conv_id + 1
                    # Override number of filters with the compressed version
                    filters = channels[conv_id - 1]
                    if batch_normalize:
                        model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                        model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                        # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                    else:
                        model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                    namesuffix = conv_id
                elif block['type'] == 'deconvolutional':
                    deconv_id = deconv_id + 1
                    if batch_normalize:
                        model.add_module('deconv{0}'.format(deconv_id), nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                        model.add_module('bn{0}'.format(deconv_id), nn.BatchNorm2d(filters))
                        # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                    else:
                        model.add_module('deconv{0}'.format(deconv_id), nn.ConvTranspose2d(prev_filters, filters, kernel_size, stride, pad))
                    namesuffix = deconv_id
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(namesuffix), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(namesuffix), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = int(stride * stride * prev_filters)
                out_filters.append(prev_filters)
                prev_stride = int(prev_stride * stride)
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = int(prev_stride / stride)
                out_strides.append(prev_stride)
                # models.append(nn.Upsample(scale_factor=stride, mode='nearest'))
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                layerlen = len(layers)
                assert (layerlen >= 1)
                prev_filters = out_filters[layers[0]]
                prev_stride = out_strides[layers[0]]
                if layerlen > 1:
                    assert (layers[0] == ind - 1)
                    for i in range(1, layerlen):
                        prev_filters += out_filters[layers[i]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind-1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLoss()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = int(len(loss.anchors)/loss.num_anchors)
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(loss)
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = int(len(yolo_layer.anchors)/yolo_layer.num_anchors)
                yolo_layer.stride = prev_stride
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            elif block['type'] in ['pose', 'pose-2d', 'pose-ind', 'pose-part', 'pose-seg', 'pose-3dr', 'pose-3drseg']:
                anchors = block['anchors'].split(',')
                anchors = [float(i) for i in anchors]
                anchor_mask = block['mask'].split(',')
                anchor_mask = [int(i) for i in anchor_mask]
                num_classes = int(block['classes'])
                num_anchors = int(block['num_anchors'])
                num_vpoints = int(block['num_vpoints'])
                class_weights = block['class_weights'].split(',')
                class_weights = np.array([float(i) for i in class_weights])
                vpoints_scale = float(block['vpoints_scale'])
                vpoints = block['vpoints'].split(',')
                vpoints = [float(i) for i in vpoints]
                vpoints = np.array(vpoints).reshape(num_vpoints, -1) * vpoints_scale
                cpoints_scale = float(block['cpoints_scale'])
                cpoints = block['cpoints'].split(',')
                cpoints = [float(i) for i in cpoints]
                cpoints = np.array(cpoints).reshape(-1, 3) * cpoints_scale
                anchor_step = int(len(anchors)/ num_anchors)
                masked_anchors = []
                for m in anchor_mask:
                    masked_anchors += anchors[int(m * anchor_step):int((m + 1) * anchor_step)]
                if block['type'] == 'pose':
                    curr_layer = PoseLayer(num_classes, masked_anchors, len(anchor_mask), vpoints)
                elif block['type'] == 'pose-2d':
                    curr_layer = Pose2DLayer(num_classes, masked_anchors, len(anchor_mask), vpoints, cpoints,
                                              class_weights, block['keypointsfile'], int(block['num_keypoints']),
                                              block['alpha_class'],  block['alpha_coord'], block['alpha_conf'])
                elif block['type'] == 'pose-ind':
                    curr_layer = PoseIndLayer(num_classes, masked_anchors, len(anchor_mask), vpoints)
                elif block['type'] == 'pose-part':
                    curr_layer = PosePartLayer(num_classes, masked_anchors, len(anchor_mask), vpoints,
                                               block['keypointsdir'], int(block['num_keypoint']))
                elif block['type'] == 'pose-seg':
                    curr_layer = PoseSegLayer(num_classes, masked_anchors, len(anchor_mask), vpoints, cpoints,
                                             class_weights, block['keypointsfile'], int(block['num_keypoints']),
                                             block['alpha_class'], block['alpha_coord'])
                elif block['type'] == 'pose-3dr':
                    curr_layer = Pose3DrLayer(num_classes, masked_anchors, len(anchor_mask), vpoints)
                elif block['type'] == 'pose-3drseg':
                    curr_layer = Pose3DrSegLayer(num_classes, masked_anchors, len(anchor_mask), vpoints)

                curr_layer.stride = prev_stride
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(curr_layer)
            elif block['type'] in ['pose-pnp']:
                anchors = block['anchors'].split(',')
                anchors = [float(i) for i in anchors]
                anchor_mask = block['mask'].split(',')
                anchor_mask = [int(i) for i in anchor_mask]
                num_classes = int(block['classes'])
                num_anchors = int(block['num_anchors'])
                anchor_step = int(len(anchors) / num_anchors)
                masked_anchors = []
                for m in anchor_mask:
                    masked_anchors += anchors[int(m * anchor_step):int((m + 1) * anchor_step)]
                if block['type'] == 'pose-pnp':
                    cpoints_scale = float(block['cpoints_scale'])
                    cpoints = block['cpoints'].split(',')
                    cpoints = [float(i) for i in cpoints]
                    cpoints = np.array(cpoints).reshape(-1, 3) * cpoints_scale  # scale
                    vertexfile = block['vertexfile']
                    num_vertex = float(block['num_vertex'])
                    curr_layer = PosePnPLayer(num_classes, masked_anchors, len(anchor_mask), cpoints, vertexfile, num_vertex)

                curr_layer.stride = prev_stride
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(curr_layer)
            else:
                print('unknown type %s' % (block['type']))
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'deconvolutional']:
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'pose':
                pass
            elif block['type'] == 'pose-2d':
                pass
            elif block['type'] == 'pose-ind':
                pass
            elif block['type'] == 'pose-part':
                pass
            elif block['type'] == 'pose-seg':
                pass
            elif block['type'] == 'pose-3dr':
                pass
            elif block['type'] == 'pose-3drseg':
                pass
            elif block['type'] == 'pose-pnp':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] in ['convolutional', 'deconvolutional']:
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fc, model)
                else:
                    save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'pose':
                pass
            elif block['type'] == 'pose-2d':
                pass
            elif block['type'] == 'pose-ind':
                pass
            elif block['type'] == 'pose-part':
                pass
            elif block['type'] == 'pose-seg':
                pass
            elif block['type'] == 'pose-3dr':
                pass
            elif block['type'] == 'pose-3drseg':
                pass
            elif block['type'] == 'pose-pnp':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()
