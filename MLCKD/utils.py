# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
import random
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    return losses.avg, top1.avg, top5.avg

def train_kl(module_list, optimizer, criterion, train_loader, device, args):
    for module in module_list:
        module.train()
    module_list[-1].eval()
    #学生模型
    model_s = module_list[0]
    #教师模型
    model_t = module_list[-1]
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_1 = AverageMeter()
    loss_2 = AverageMeter()
    loss_kd_avg = AverageMeter()
    loss_kl_avg = AverageMeter()
    #获取计算loss的方式
    criterion_ce, criterion_kl, criterion_kd = criterion
    cls_t = model_t.get_feat_modules()[-1]
    cls_s = model_s.get_feat_modules()[-1]
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(device), targets.cuda(device)
        with torch.no_grad():
            feat_t, output_t = model_t(inputs, is_feat=True)

        feat_s, output_s = model_s(inputs, is_feat=True)
        loss_kd, logits1, logits2, logits3 = criterion_kd(feat_s, feat_t, cls_t, cls_s)
        loss_ce = criterion_ce(output_s, targets)
        loss1 = criterion_kl(output_s, logits1, 3)
        loss2 = criterion_kl(logits1, output_t, 4) + criterion_kl(logits2, output_t, 4)
        loss_kl = criterion_kl(output_s, output_t, 4)
        loss = loss_ce + args.beta * (  loss1 + args.alpha * loss2 +  loss_kl + args.gamma * loss_kd)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output_s, targets, topk=(1, 5))
        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        loss_1.update(loss1, batch_size)
        loss_2.update(loss2, batch_size)
        loss_kd_avg.update(loss_kd, batch_size)
        loss_kl_avg.update(loss_kl, batch_size)

    return losses.avg, top1.avg, top5.avg, loss_1.avg, loss_2.avg, loss_kd_avg.avg, loss_kl_avg.avg


def test(model, test_loader, device):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(device), targets.cuda(device)
            feat_s, outputs = model(inputs, is_feat=True)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            batch_size = targets.size(0)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)

    return top1.avg, top5.avg


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
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr


def str2bool(s):
    if s not in {'F', 'T'}:
        raise ValueError('Not a valid boolean string')
    return s == 'T'

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True