import argparse
from functools import total_ordering
from glob import glob
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet import *
from PIL import ImageFile
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
logdir = '../../tensorlogs/scalars/' + 'RESNET50_IMGNTT_CBAM_epch80_'+ datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(logdir)

ttime_val = 0.
ttime_train = 0.
largerthan50 = 0.
ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', default=50, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
parser.add_argument('--net-type', type=str, choices=["ImageNet", "CIFAR10", "CIFAR100","Tiny-ImageNet"], default=None)
best_prec1 = 0

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')

def main():
    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # create model
    if args.arch == "resnet":
        model = ResidualNet( args.net_type, args.depth, 10, args.att_type)
        # 3rd argument has to be class number
        # model = ResidualNet( 'ImageNet', args.depth, 1000, args.att_type )
        # assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], 
        # "network type should be ImageNet or CIFAR10 / CIFAR100"

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    #model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()
    print ("model is in line 85")
    # print (model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # import pdb
    # pdb.set_trace()
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.RandomResizedCrop(112),
                # transforms.Resize(224),
                # transforms.CenterCrop(112),
                transforms.ToTensor(),
                normalize,
                ])),
            batch_size=args.batch_size, shuffle=False,
           num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.Resize(224),
            transforms.RandomResizedCrop(112),
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.prefix)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    totaltime_test = 0.
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # print(input[0][0][0])
        # measure data loading time
        data_time.update(time.time() - end)
        # target = target.cuda(async=True)
        # target = target.cuda(non_blocking=True)
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input).to("cpu")
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        # print(batch_time)
        end = time.time()
        global ttime_train
        ttime_train += batch_time.val/60
        global largerthan50
        if largerthan50 !=0 and top5.avg > 50:
            largerthan50 = ttime_train + ttime_val
        if i % args.print_freq == 0:
            print('_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'totaltime(min) {totaltime:.3f}\t'
                  'over50(min) {time50:.3f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, totaltime=ttime_train, time50 =largerthan50))
                   
            writer.add_scalar('training loss/iteration', losses.val, epoch * len(train_loader) + i)
            writer.add_scalar('training loss/avg', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('acc_per_iteration/top1Precval', top1.val, epoch * len(train_loader) + i)
            writer.add_scalar('acc_per_iteration/top5Precval', top5.val, epoch * len(train_loader) + i)
            writer.add_scalar('avg_acc/top1Precval', top1.avg, epoch * len(train_loader) + i)
            writer.add_scalar('avg_acc/top5Precval', top5.avg, epoch * len(train_loader) + i)
            writer.add_scalar('time(min)/iteration', ttime_train, epoch * len(train_loader) + i)

def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    totaltime_val = 0.
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        target = target.cuda(non_blocking=True)
        # oneinput = input[0][0][0].view(-1)
        # print(type(oneinput[0]))
        # print(target[0])
        # original code of this part
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        
        # with torch.no_grad():
            # input_var = input.clone().detach().requires_grad_(True)
            # torch.tensor(input,requires_grad=True)
        # input_var = torch.autograd.Variable(input, volatile = True)
        # target_var = torch.autograd.Variable(target, volatile = True)
        
        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        # losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        global ttime_val
        ttime_val += batch_time.val /60
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'totaltime(min) {totaltime:.3f}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5, totaltime = ttime_val))
            writer.add_scalar('evaluation loss/iteration', losses.val, epoch * len(val_loader) + i)
            writer.add_scalar('avg evaluation loss', losses.avg, epoch * len(val_loader) + i)
            writer.add_scalar('val: top1Precval', top1.val, epoch * len(val_loader) + i)
            writer.add_scalar('val: top5Precval', top5.val, epoch * len(val_loader) + i)
            writer.add_scalar('val: top1Precval avg', top1.avg, epoch * len(val_loader) + i)
            writer.add_scalar('val: top5Precval avg', top5.avg, epoch * len(val_loader) + i)
            writer.add_scalar('time(min)/iteration', ttime_val, epoch * len(val_loader) + i)
            
        
    
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, prefix):
    filename='./checkpoints/%s_checkpoint.pth.tar'%prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar'%prefix)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # print(correct[1])
    # print(len(correct[0]))
    for k in topk:
        # print(correct[:k])
        correct_k = torch.reshape(correct[:k],(-1,)).float().sum(0, keepdim=True)
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
writer.close()