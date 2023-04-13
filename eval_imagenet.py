##############################################################################
### Evaluation code for imagenet under mobilenet_v2, resnet18 and resnet50 ###
##############################################################################

import os
import random
import time
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.search_resnet_imagenet_tmp as resnet
import models.search_mbnet_imagenet_tmp as mbnet
from config import ImageNetEvalConfig
from tensorboardX import SummaryWriter
import utils


def main():
    args = ImageNetEvalConfig()
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    writer = SummaryWriter(log_dir=os.path.join(args.path, "tb"))
    writer.add_text('config', args.as_markdown(), 0)

    logger = utils.get_logger(os.path.join(args.path, "{}.log".format(args.name)))
    args.print_params(logger.info)

    main_worker(args, logger, writer)


def main_worker(args, logger, writer):
    logger.info("Logger is set - training start")
    torch.cuda.set_device(args.gpus[0])

    criterion = nn.CrossEntropyLoss().cuda()

    if 'resnet' in args.arch:
        model = resnet.__dict__[args.arch](criterion, "GateAct", "GatePool")
    elif args.arch == "mobilenet_v2":
        model = mbnet.__dict__["mobilenet_v2"](criterion, "GateAct")

    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model_path = args.evaluate
    logger.info("\n=> loading NAS search alpha from '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location="cpu")
    logger.info("Previous Recorded Top@1 : '{}'".format(checkpoint['best_prec1']))
    model.load_state_dict(checkpoint['state_dict'])  
    model.change_gate(Gate_status = False)

    ##### Change scaling of the resnet50 network to 0.1
    if args.arch == "resnet50":
        model.change_scaling(weight_scale = 0.1)

    model.cuda()

    if args.evaluate:
            
        # for name, W in model.named_parameters():
        #     print(name, W.shape)

        # # --------------------------------
        # ### check model detail
        # from torchinfo import summary
        # summary(model, input_size=(1, 3, 224, 224), depth = 10, col_width = 15,
        #         col_names = ["input_size", "output_size", "num_params", "kernel_size"])

        # from prettytable import PrettyTable
        # def count_parameters(model):
        #     table = PrettyTable(["Modules", "Parameters"])
        #     total_params = 0
        #     for name, parameter in model.named_parameters():
        #         param = parameter.numel()
        #         table.add_row([name, param])
        #         total_params+=param
        #     print(table)
        #     print(f"Total Trainable Params: {total_params}")
        #     return total_params
        # count_parameters(model)

        # inp = torch.rand(1, 1, 1250, 1).to(device)
        # count_ops(net, inp)
        # # --------------------------------
        
        validate(val_loader, model, 0, args, logger)
        return

# ----------------------------------------------------------------------------------

def validate(val_loader, model, epoch, args, logger):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    args.print_freq = 50
    args.epochs = 1
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = model.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, args.epochs, i, len(val_loader)-1, losses=losses,
                        top1=top1, top5=top5))
        progress.display_summary()
    
    logger.info("Valid: Final Prec@1 {:.4%}, Prec@5 {:.4%}".format(top1.avg, top5.avg))


    return top1.avg, top5.avg


# ----------------------------------------------
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


######################################################

if __name__ == '__main__':
    main()
