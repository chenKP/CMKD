# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

import argparse
from utils import *
from dataset import *
from distill import *
import models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from time import time
import tensorboard_logger as tb_logger


def main():
    parser = argparse.ArgumentParser()
    # data                                                      
    parser.add_argument('--data_dir', default='/home/datasets/')
    parser.add_argument('--data', default='CIFAR100')
    parser.add_argument('--trained_dir', default='')
    parser.add_argument('--folder', default='/home/chenkp/test/distill-project/save', type=str,
                        help='save student model path')
     
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--schedule', default=[150, 180, 210], type=int, nargs='+')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--temperature', default=4, type=float)

    parser.add_argument('--model_t', default='resnet32x4', type=str)
    parser.add_argument('--model', default='resnet8x4', type=str)#
    parser.add_argument('--trail', default=1, type=int)
    
    parser.add_argument('--beta', default=1, type=float, help='weight for total')
    parser.add_argument('--alpha', default=0.5, type=float, help='weight for layer logits KD')
    parser.add_argument('--gamma', default=15, type=float, help='weight for feature MSELoss')
    parser.add_argument('--gpu_id', default='0', type=str)

    args = parser.parse_args()
    source_lr = args.lr
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_id = int(args.gpu_id)
    lucky_number = 114514 #114514
    seed_torch(lucky_number)
    logger = tb_logger.Logger(logdir='./{}/log_{}_t-{}-s-{}'.format(args.folder,args.trail,args.model_t,args.model), flush_secs=2)
    for param in sorted(vars(args).keys()):
        print('--{0} {1}'.format(param, vars(args)[param]))

    train_loader, test_loader, args.num_classes, args.image_size = create_loader(args.batch_size, args.data_dir, args.data)
    model_t = models.__dict__[args.model_t](num_classes=args.num_classes)
   
    model_t.load_state_dict(torch.load(args.trained_dir)['model'])
    # model_t.load_state_dict(torch.load(args.trained_dir),strict=True)
    
    model_s = models.__dict__[args.model](num_classes=args.num_classes)
    device = 0
    data = torch.randn(2, 3, args.image_size, args.image_size)
    model_t.eval()
    model_s.eval()
    with torch.no_grad():
        feat_t, _ = model_t(data, is_feat=True)
        feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    args.s_shapes = feat_s[-2].size()
    args.t_shapes = feat_t[-2].size()

    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = DistillKL(args.temperature)
    s_n = feat_s[-2].shape[1]
    t_n = feat_t[-2].shape[1]
    s_n1 = feat_s[-3].shape[1]
    s_n2 =  feat_s[-4].shape[1]
    criterion_kd = MLCKD( s_n, t_n, s_n1, s_n2)
    module_list.append(criterion_kd)
    trainable_list.append(criterion_kd)

    criterion = nn.ModuleList([])
    criterion.append(criterion_ce)
    criterion.append(criterion_kl)
    criterion.append(criterion_kd)

    optimizer = optim.SGD(trainable_list.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    module_list.append(model_t)

    module_list.cuda()
    criterion.cuda()
    cudnn.benchmark = True
    best_acc1 = 0
    for epoch in range(1, args.epoch + 1):
        s = time()
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc1, train_acc5, loss_1, loss_2, loss_kd, loss_kl = train_kl(module_list, optimizer, criterion, train_loader, device, args)
        test_acc1, test_acc5 = test(model_s, test_loader, device)
        print('Epoch: {0:>3d} |GPU: {1:} |Train Loss: {2:>2.4f} |Train Top1: {3:.4f} |Train Top5: {4:.4f} |Test Top1: {5:.4f} |Test Top5: {6:.4f}| Time: {7:>5.1f} (s)'
              .format(epoch, gpu_id, train_loss, train_acc1, train_acc5, test_acc1, test_acc5, time() - s))
        logger.log_value('train_acc1', train_acc1, epoch)
        logger.log_value('train_acc5', train_acc5, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_acc1', test_acc1, epoch)
        logger.log_value('test_acc5', test_acc5, epoch)
        logger.log_value('loss1', loss_1, epoch)
        logger.log_value('loss2', loss_2, epoch)
        logger.log_value('loss_kd', loss_kd, epoch)
        logger.log_value('loss_kl', loss_kl, epoch)
        if best_acc1 < test_acc1:
            best_acc1 = test_acc1
            torch.save(model_s.state_dict(), './{}/log_{}_t-{}-s-{}/best.pth'.format(args.folder, args.trail,args.model_t,args.model))
            print('save best model...')
            test_merics = {
                'epoch': epoch,
                'best_acc1': float(test_acc1),
                'best_acc5': float(test_acc5),
                'train_acc1': float(train_acc1),
                'train_acc5': float(train_acc5),
                'batch-size':args.batch_size,
                'teacher':args.model_t,
                'student':args.model,
                'lr':source_lr,
                'aphl:':args.alpha,
                'beta:':args.beta,
                'gamma:':args.gamma}
            save_dict_to_json(test_merics, os.path.join('./{}/log_{}_t-{}-s-{}/best.pth'.format(args.folder, args.trail,args.model_t,args.model)))
            

if __name__ == '__main__':
    main()
