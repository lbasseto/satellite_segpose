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
from darknet import Darknet

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
model       = Darknet(cfgfile)

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

params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay}]
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                      dampening=0, weight_decay=decay)

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

def train(epoch):
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
        loss, interPram, no_reg_loss, l1_reg_only = cur_model(data, [target, seg])
        cur_model.seen += data.data.size(0)
        loss.backward()
        optimizer.step()

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
        print('no_reg_loss: ' + str(no_reg_loss) + '\tl1_reg_only: ' + str(l1_reg_only), '\tregularized_loss: ' + str(loss.data))

        step = int(cur_model.seen / batch_size)
        if step % 20 == 0:
            _, validParam = validate_single_batch(step, False)

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

def validate_single_batch(step, savefig=False):
    validation_loader = torch.utils.data.DataLoader(
        dataset.RandomBackgroundDataset(backgroundlist, validlist,
                                        shape=(init_width, init_height), shuffle=True,
                                        transform=transforms.Compose([transforms.ToTensor(),]),
                                        train=False), batch_size=batch_size, shuffle=True) #**kwargs)

    data, target, seg = next(iter(validation_loader))

    if use_cuda:
        data = data.cuda()

    data, target, seg = Variable(data), Variable(target), Variable(seg)

    loss, param, _, _ = model(data, [target, seg])

    return loss.data, param

if __name__ == "__main__":
    for epoch in range(init_epoch, max_epochs):
        train(epoch)
        # test(epoch)
