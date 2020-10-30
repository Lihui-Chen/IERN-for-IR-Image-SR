import functools
import importlib
import torch
import torch.nn as nn
from torch.nn import init
import os


####################
# define network
####################

def create_model(opt):
    if opt['mode'] == 'sr':
        net = define_net(opt['networks'])
        return net
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])


# choose one network
def define_net(opt):
    which_model = opt['which_model']
    print('===> Building network [%s]...' % which_model)
    which_model = 'networks.' + which_model

    netpack = importlib.import_module(which_model)
    net = netpack.Net(opt=opt)

    if torch.cuda.device_count() > 1 and len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
        torch.distributed.init_process_group(backend='nccl', # todo: tcp port need change according to avialbe port.
                                             init_method='tcp://localhost:23456', rank=0, world_size=1)
        net = net.cuda()
        net = nn.parallel.DistributedDataParallel(net)

    elif torch.cuda.is_available():
        net = net.cuda()


    return net







