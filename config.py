""" Config class for search/augment """
import argparse
import os
import genotypes as gt
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--num_classes', type=int, default=10, help='number of classes of dataset')
        parser.add_argument('--data_path', default='./data/', help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4, help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. `all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for alpha')
        parser.add_argument('--lat_lamda', type=float, default=0, help='Lamda for alpha latency loss constraint')
        parser.add_argument('--pretrained_path', help='Pretained model path')
        parser.add_argument('--act_type', default='GateAct', help='Non-linear activation type')
        parser.add_argument('--pool_type', default='GatePool', help='Pooling layer type')
        parser.add_argument('--arch', default='vgg16_gated_bn', help='Model architecture type')
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        # self.data_path = './data/'
        self.path = os.path.join(f'searchs_{self.arch}', self.name + str("_lat_lmd_{:.0e}_lr{}ep{}".format(self.lat_lamda, self.w_lr, self.epochs)))
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)

class FinetuneConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Finetune config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--num_classes', type=int, default=10, help='number of classes of dataset')
        parser.add_argument('--data_path', default='./data/', help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--w_lr', type=float, default=0.01, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=5e-4, help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. `all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
        # parser.add_argument('--init_channels', type=int, default=16)
        # parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
        # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
        parser.add_argument('-e', '--evaluate', default='evaluate', type=str, metavar='PATH',
                            help='path to checkpoint (default: none)')
        # parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        # parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
        #                     help='weight decay for alpha')
        parser.add_argument('--lat_lamda', type=float, default=0, help='Lamda for alpha latency loss constraint')
        parser.add_argument('--pretrained_NAS_path', help='Pretained NAS model path')
        parser.add_argument('--all_poly_avgpl', default=True, help='Pretained NAS model path')
        parser.add_argument('--pretrained_path', help='Pretained model path')
        parser.add_argument('--checkpoint_path', help='Checkpoint path')
        parser.add_argument('--act_type', default='GateAct', help='Non-linear activation type')
        parser.add_argument('--pool_type', default='GatePool', help='Pooling layer type')
        parser.add_argument('--arch', default='vgg16_gated_bn', help='Model architecture type')
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        # self.data_path = './data/'
        # self.path = os.path.join(f'searchs_{self.arch}', self.name + str("_{:.0e}".format(self.lat_lamda) + '_Finetune'))
        self.path = os.path.join('aaa')
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)
        if self == "cifar10":
            self.num_classes = 10
        elif self == "cifar100":
            self.num_classes = 100

class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True, help='CIFAR10 / MNIST / FashionMNIST')
        parser.add_argument('--batch_size', type=int, default=96, help='batch size')
        parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=600, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=36)
        parser.add_argument('--layers', type=int, default=20, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
        parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')

        parser.add_argument('--genotype', required=True, help='Cell genotype')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('augments', self.name)
        self.genotype = gt.from_str(self.genotype)
        self.gpus = parse_gpus(self.gpus)


class ImageNetConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Train ImageNet config")
        parser.add_argument('--name', default = "imagenet")
        parser.add_argument('--data', metavar='DIR', default='/data/imagenet/',
                            help='path to dataset (default: imagenet)')
        parser.add_argument('--arch', default='resnet18', help='Model architecture type')

        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('-p', '--print-freq', default=100, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        #                     help='evaluate model on validation set')        
        parser.add_argument('-e', '--evaluate', default='evaluate', type=str, metavar='PATH',
                            help='path to checkpoint (default: none)')
        parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--seed', default=0, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. `all` indicates use all gpus.')

        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        
        parser.add_argument('--decay', type=int, default=30, help='Lamda for alpha latency loss constraint')
        parser.add_argument('--pretrained_NAS_path', help='Pretained NAS model path')
        parser.add_argument('--pretrained_path', help='Pretained model path')
        parser.add_argument('--all_poly_avgpl', default=True, help='Pretained NAS model path')
        parser.add_argument('--lat_lamda', type=float, default=0, help='Lamda for alpha latency loss constraint')
        parser.add_argument('--act_type', default='GateAct', help='Non-linear activation type')
        parser.add_argument('--pool_type', default='GatePool', help='Pooling layer type')
        parser.add_argument('--ext', default='', type=str,
                            help='self-defined extension for saved location')
        parser.add_argument('--cvg_epoch', type=float, default = 10, help='converge epoch for activation layer')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join(f'searchs_{self.arch}_{self.name}', 
                                    f'{self.name}_{self.lat_lamda}_Finetune_lr{self.lr}_dcy{self.decay}{args.ext}')
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)

class ImageNetTestConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Train ImageNet config")
        parser.add_argument('--name', default = "imagenet")
        parser.add_argument('--data', metavar='DIR', default='/data/imagenet/',
                            help='path to dataset (default: imagenet)')
        parser.add_argument('--arch', default='resnet18', help='Model architecture type')

        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=100, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                'batch size of all GPUs on the current node when '
                                'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('-p', '--print-freq', default=100, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        #                     help='evaluate model on validation set')        
        parser.add_argument('-e', '--evaluate', default='evaluate', type=str, metavar='PATH',
                            help='path to checkpoint (default: none)')
        parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        parser.add_argument('--seed', default=0, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. `all` indicates use all gpus.')

        parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        
        parser.add_argument('--decay', type=int, default=30, help='Lamda for alpha latency loss constraint')
        parser.add_argument('--pretrained_NAS_path', help='Pretained NAS model path')
        parser.add_argument('--pretrained_path', help='Pretained model path')
        parser.add_argument('--all_poly_avgpl', default=True, help='Pretained NAS model path')
        parser.add_argument('--lat_lamda', type=float, default=0, help='Lamda for alpha latency loss constraint')
        parser.add_argument('--act_type', default='GateAct', help='Non-linear activation type')
        parser.add_argument('--pool_type', default='GatePool', help='Pooling layer type')
        parser.add_argument('--ext', default='', type=str,
                            help='self-defined extension for saved location')
        parser.add_argument('--cvg_epoch', type=float, default = 10, help='converge epoch for activation layer')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        args.resume = args.resume.split(':')[-1]
        super().__init__(**vars(args))

        self.path = os.path.join(f'searchs_test', 
                                    f'{args.ext}_')
        print("Path:", self.path)
        print("Evaluate:", self.evaluate)
        self.plot_path = os.path.join(self.path, 'plots')
        print("Resume path:", args.resume)
        self.gpus = parse_gpus(self.gpus)
        # exit()



class ImageNetEvalConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Train ImageNet config")
        parser.add_argument('--name', default = "imagenet")
        parser.add_argument('--data', metavar='DIR', default='/data/imagenet/',
                            help='path to dataset (default: imagenet)')
        parser.add_argument('--workers', type=int, default=4, 
                            help='# of workers')
        parser.add_argument('--arch', default='resnet18', 
                            help='Model architecture type')
        parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N',
                            help='batch-size')     
        parser.add_argument('-e', '--evaluate', default='evaluate', type=str, metavar='PATH',
                            help='path to checkpoint (default: none)')
        parser.add_argument('--seed', default=0, type=int,
                            help='seed for initializing training. ')
        parser.add_argument('--gpus', default='0', 
                            help='gpu device ids separated by comma. `all` indicates use all gpus.')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join(f'eval_{self.name}', 
                                    f'{self.evaluate}')
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)