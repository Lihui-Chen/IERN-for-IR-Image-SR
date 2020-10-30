import os, argparse
from collections import OrderedDict
from datetime import datetime
import yaml
import torch
from utils import util
import shutil

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def add_args():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    parser.add_argument('-nblocks', type=int, default=None, help='number of basic blocks')
    parser.add_argument('-nlayers', type=int, default=None, help='number of layers in each basic block')
    parser.add_argument('-iterations', type=int, default=None, help='number of iterations')
    parser.add_argument('-trained_model', type=str, default=None, help='Path of the trained model')
    parser.add_argument('-lr_path', type=str, default=None, help='Path of the trained model')
    return parser.parse_args()


def parse(opt_path, nblocks=None, nlayers=None, iterations=None, trained_model=None, lr_path=None):
    Loader, Dumper = OrderedYaml()
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    opt['timestamp'] = get_timestamp()
    scale = opt['scale']
    rgb_range = opt['rgb_range']

    # export CUDA_VISIBLE_DEVICES
    if torch.cuda.is_available():
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('===> Export CUDA_VISIBLE_DEVICES = [' + gpu_list + ']')
    else:
        print('===> CPU mode is set (NOTE: GPU is recommended)')

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['rgb_range'] = rgb_range
        if phase =='test' and lr_path is not None:
            dataset['mode'] = 'LR'
            dataset['dataroot_LR'] = lr_path
        
    # for network initialize
    opt['networks']['scale'] = opt['scale']
    network_opt = opt['networks']
    if nblocks:
        network_opt['nBlock'] = nblocks
    if nlayers:
        network_opt['nDenselayer'] = nlayers
    if iterations:
        network_opt['iterations'] = iterations

    config_str = '%s_x%d_in%df%d'%(network_opt['which_model'].upper(), opt['scale'], network_opt['in_channels'],
                                                        network_opt['num_features'], )
    exp_path = os.path.join(os.getcwd(), 'experiments', config_str)

    if opt['is_train'] and opt['solver']['pretrain']:
        if 'pretrained_path' not in list(opt['solver'].keys()):
            raise ValueError("[Error] The 'pretrained_path' does not declarate in *.yml")
        exp_path = os.path.dirname(os.path.dirname(opt['solver']['pretrained_path']))
        if opt['solver']['pretrain'] == 'finetune':
            exp_path += '_finetune'

    exp_path = os.path.relpath(exp_path)
    if trained_model:
        opt['solver']['pretrained_path'] = trained_model

    path_opt = OrderedDict()
    path_opt['exp_root'] = exp_path
    path_opt['epochs'] = os.path.join(exp_path, 'epochs')
    path_opt['visual'] = os.path.join(exp_path, 'visual')
    path_opt['records'] = os.path.join(exp_path, 'records')
    opt['path'] = path_opt

    if opt['is_train']:
        # create folders
        if opt['solver']['pretrain'] == 'resume':
            opt = dict_to_nonedict(opt)
        else:
            util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root'))
            save(opt)
            opt = dict_to_nonedict(opt)

        print("===> Experimental DIR: [%s]"%exp_path)

    return opt


def save(opt):
    dump_dir = opt['path']['exp_root']
    dump_path = os.path.join(dump_dir, 'options.yml')
    network_file = opt["networks"]['which_model'] + '.py'
    shutil.copy('./networks/'+network_file, os.path.join(dump_dir, network_file))
    with open(dump_path, 'w') as dump_file:
        yaml.dump(opt, dump_file, Dumper=Dumper)


class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')