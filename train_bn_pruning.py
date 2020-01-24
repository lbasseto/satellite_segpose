from __future__ import print_function
import sys
import cv2
import h5py

if False:
    if len(sys.argv) != 4:
        print('Usage:')
        print('python train.py datacfg cfgfile weightfile')
        exit()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg
from darknet_bn_pruning import DarknetBnPruning
from darknet import Darknet

from utils_bn_pruning import *
from models.layers import bn
import sgd as bnopt

from tensorboardX import SummaryWriter

# Training settings

datacfg       = 'cfg/satellite.data'

data_options  = read_data_cfg(datacfg)
cfgfile       = data_options['cfgfile']
net_options   = parse_cfg(cfgfile)[0]

weightfile    = ''
startepoch    = 0
if 'weightfile' in data_options.keys():
    weightfile    = data_options['weightfile']
    startepoch    = data_options['startepoch']

trainlist      = data_options['train']
validlist      = data_options['valid']

logdir         = data_options['logdir'] + cfgfile[cfgfile.rfind('/'):]

logWriter = SummaryWriter(logdir)
backgroundlist = ""
nsamples      = file_lines(trainlist)
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
max_epochs    = int(net_options['max_epochs'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
use_cuda      = True
if gpus == 'None':
    use_cuda  = False
seed          = int(time.time())
save_interval = int(net_options['save_interval'])  # epoches


if not os.path.exists(logdir):
    os.mkdir(logdir)

###############
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

print('Loading cfgfile from \'%s\' ... Done!' % cfgfile)
model       = DarknetBnPruning(cfgfile)

model.print_network()
if os.path.exists(weightfile):
    model.load_state_dict(torch.load(weightfile))
    print('Loading weights from \'%s\' ... Done!' % weightfile)

init_width        = model.width
init_height       = model.height
init_epoch        = int(startepoch)

model.seen        = init_epoch * nsamples
processed_epochs  = init_epoch

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    print('gpu', gpus)
    if ngpus > 1:
        print('mutli gpus '+gpus)
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

def scale_gammas(alpha, model, scale_down=True):
    # get pairs of consecutive layers
    layers = expand_model(model, [])

    alpha_ = 1 / alpha

    if not scale_down:
        # after training we want to scale back up so need to invert alpha
        alpha_  = alpha
        alpha   = 1 / alpha

    for l1, l2 in zip(layers,layers[1:]):
        if(isinstance(l1, bn.BatchNorm2dEx) and isinstance(l2, nn.Conv2d)):
            l1.weight.data = l1.weight.data * alpha
            l2.weight.data = l2.weight.data * alpha_

    return model

def switch_to_follow(model):
    first = True # want to skip the first bn layer - only do follow up layers
    for layer in expand_model(model, []):
        if isinstance(layer, bn.BatchNorm2dEx):
            if not first:
                layer.follow = True
            first = False

'''
Equation (2) on page 6
'''
def compute_penalties(model, rho):
    penalties  = []
    image_dims = compute_dims(model) # calculate output sizes of each convolution so we can count penalties

    # TODO change: this won't work for ResNet since a lot of the convs don't have bnx layers after them
    layers = list(filter(lambda l : isinstance(l, nn.Conv2d), expand_model(model, [])))

    # zip xs (tail xs) - need to know kernel size of follow-up layer
    for i in range(len(layers)):
        l    = layers[i]
        tail = layers[i+1:]

        i_w, i_h = init_width, init_height
        k_w, k_h = l.kernel_size[0], l.kernel_size[1]
        c_prev   = l.in_channels
        c_next   = l.out_channels

        follow_up_cost = 0.

        for j, follow_up_conv in enumerate(tail):
            follow_up_cost += follow_up_conv.kernel_size[0] * follow_up_conv.kernel_size[1] * follow_up_conv.in_channels + image_dims[j+i]**2

        ista = ((1 / i_w * i_h) * (k_w * k_h * c_prev)) # + follow_up_cost
        ista = rho * ista

        print(ista)
        penalties.append(ista)

    return penalties

alpha = 1.
rho   = 0.000001

#TODO original is compute_penalties_ not compute_penalties !!!
ista_penalties = compute_penalties(model, rho)
print_layer_ista_pair(model, ista_penalties)

non_bn_params = [p for n, p in model.named_parameters() if 'bnx' not in n]
bn_params     = [p for n, p in model.named_parameters() if 'bnx' in n]

# should weight decay be zero?
optimizer    = optim.SGD([p for n, p in model.named_parameters() if 'bnx' not in n], lr=learning_rate, dampening=0, momentum=momentum, weight_decay=decay)
bn_optimizer = bnopt.BatchNormSGD([p for n, p in model.named_parameters() if 'bnx' in n], lr=learning_rate, ista=ista_penalties, momentum=0.9)

# step two: gamma rescaling trick
scale_gammas(alpha, model=model, scale_down=True)

count_sparse_bn(model, logWriter, 0, w = init_width, h = init_height)
print_sparse_bn(model)

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if epoch >= steps[i]:
            lr = lr * scale
            if epoch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def network_grad_ratio(model):
    '''
    for debug
    :return:
    '''
    gradsum = 0
    datasum = 0
    layercnt = 0
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.abs().mean()
            data = param.data.abs().mean()
            # print(grad)
            gradsum += grad
            datasum += data
            layercnt += 1
    gradsum /= layercnt
    datasum /= layercnt
    return gradsum, datasum

def finetune(model, writer, epochs):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, dampening=0, momentum=momentum, weight_decay=decay)
    bn_optimizer = None
    for epoch in range(1, epochs):
        train(epoch, model, finetune=True)
        count_sparse_bn(model, writer, epoch, w = init_width, h = init_height)
        print_sparse_bn(model)

def train(epoch, model, finetune = False):
    global processed_epochs
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model

    train_loader = torch.utils.data.DataLoader(dataset.RandomBackgroundDataset(backgroundlist, trainlist,
                                        shape=(init_width, init_height), shuffle=True,
                                        transform=transforms.Compose([transforms.ToTensor(),]),
                                        train=True, seen=cur_model.seen, batch_size=batch_size, num_workers=num_workers),
                                        batch_size=batch_size, shuffle=True, **kwargs)

    lr = adjust_learning_rate(optimizer, processed_epochs)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    cur_model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)

    for batch_idx, (data, target, seg) in enumerate(train_loader):
        if use_cuda:
            data = data.cuda()

        data, target, seg = Variable(data), Variable(target), Variable(seg)

        optimizer.zero_grad()
        if not finetune:
            bn_optimizer.zero_grad()

        loss, interPram = cur_model(data, [target, seg])
        cur_model.seen += data.data.size(0)
        loss.backward()

        optimizer.step()
        if not finetune:
            bn_optimizer.step()

        trainParam = interPram
        itemCnt = len(trainParam)

        for i in range(itemCnt):
            layerId = trainParam[i][0]
            lossParamTrain = trainParam[i][2]
            paramStr = ''
            for p in lossParamTrain:
                paramStr += ('%f ' % p)
            print('%d/%s: %s' % (cur_model.seen, layerId, paramStr))

        # gradient debug
        avggrad, avgdata = network_grad_ratio(cur_model)
        print('avg gradiant ratio: %f, %f, %f' % (avggrad, avgdata, avggrad/avgdata))
        print('loss: ' + str(loss.data))

        step = int(cur_model.seen / batch_size)
        if step % 20 == 0:
            _, validParam = validate_single_batch(step, cur_model, False)

            paramlen = len(trainParam[0][2])
            for j in range(paramlen):
                titleStr = ('%d' % j)
                keyset = {}
                for i in range(itemCnt):
                    layerId = trainParam[i][0]
                    assert (len(trainParam[i][2]) == paramlen)
                    keyset[layerId + '_train'] = trainParam[i][2][j]

                    assert (len(validParam) == itemCnt)
                    assert (validParam[i][0] == layerId)
                    assert (len(validParam[i][2]) == paramlen)
                    keyset[layerId + '_valid'] = validParam[i][2][j]

                logWriter.add_scalars(titleStr, keyset, step * batch_size)
            logWriter.add_scalar("settings/learning_rate", lr, step * batch_size)

    processed_epochs = processed_epochs + 1

    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))
    if processed_epochs % save_interval == 0:
        logging('save weights to %s/%06d.pth' % (logdir, processed_epochs))
        cur_model.seen = processed_epochs * len(train_loader.dataset)
        torch.save(cur_model.state_dict(), ('%s/%06d.pth' % (logdir, processed_epochs)))

def validate_single_batch(step, model, savefig=False):
    validation_loader = torch.utils.data.DataLoader(
        dataset.RandomBackgroundDataset(backgroundlist, validlist,
                                        shape=(init_width, init_height), shuffle=True,
                                        transform=transforms.Compose([transforms.ToTensor(),]),
                                        train=False), batch_size=batch_size, shuffle=True) #**kwargs)

    data, target, seg = next(iter(validation_loader))

    if use_cuda:
        data = data.cuda()

    data, target, seg = Variable(data), Variable(target), Variable(seg)

    loss, param = model(data, [target, seg])

    return loss.data, param

if __name__ == "__main__":
    for epoch in range(init_epoch, max_epochs):
        train(epoch, model)
        count_sparse_bn(model, logWriter, epoch, w = init_width, h = init_height)
        new_penalties = compute_penalties(model, rho)
        # rho=0.00000001
        # test(epoch)

    # step four: remove constant channels by switching bn to "follow" mode
    switch_to_follow(model)

    # step five: gamma rescaling trick
    scale_gammas(alpha, model=model, scale_down=False)

    # step six: finetune
    finetune(model, logWriter, epochs = 50)

    # zero out any channels that have a 0 batchnorm weight
    print("Compressing model...")
    sparsify_on_bn(model)

    compressed_model = Darknet
    new_model = compress_convs(model, compressed_model, cfgfile)

    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        print('gpu', gpus)
        if ngpus > 1:
            print('mutli gpus '+gpus)
            new_model = torch.nn.DataParallel(new_model).cuda()
        else:
            new_model = new_model.cuda()

    # step six: finetune
    finetune(new_model, logWriter, epochs = 50)
