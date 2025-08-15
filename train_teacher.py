from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import math
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models import model_dict
from transformers import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR, LambdaLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


from dataset.cifar100 import get_cifar100_dataloaders
from dataset.speechcommands import get_speechcommands_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.tiny_imagenet import get_tinyimagenet_dataloaders
from dataset.squad_loader import get_squad_dataloaders


from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate
from helper.loops_qa import train_squad, validate_squad  # add this at the top
import builtins

def suppress_print_if_not_rank0():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            builtins.print = lambda *args, **kwargs: None

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--resize_input', action='store_true', help='resize CIFAR images to 224x224 (e.g., for pretrained ConvNeXt)')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed from torch.distributed.launch or torchrun')

    # Overfitting control
    parser.add_argument('--cutmix', action='store_true', help='Use CutMix augmentation')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--strong_aug', action='store_true', help='Use stronger data augmentation (RandAugment)')
    parser.add_argument('--early_stopping', type=int, default=0, help='Enable early stopping after N epochs of no improvement')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Clip gradients to this value')
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone at start (head-only training)')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='linear warmup epochs before LR scheduler')



    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased', help='Tokenizer name from HuggingFace')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'], help='optimizer type')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine'],
                        help='type of learning rate scheduler to use')




    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet101', 'resnet152', 'regnety4gf', 'regnety16gf', 'regnety32gf',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_40_4', 'wrn_28_10', 'wrn_28_1', 'wrn_28_4', 'wrn_20_4', 'wrn_40_10', 'wrn_10_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'roberta', 'distilbert', 'tinybert', 'mobilebert', 'albert', 'bertbase', 'bertsmall',
                                 'vit_tiny', 'vit_small', 'vit_base', 'vit_medium', 'vit_distilled', 'vit_pretrained', 'vit_pretrainedimagetiny',
                                 'MobileNetV1','MobileNetV2', 'ShuffleV1', 'MobileNetV3.1', 'ShuffleV2', 'wide', 'ResNet50','MobileNetV3','ViT_B_16',
				 'resnext50_32x4d','resnext101_32x8d','pyramidnet272', 'pyramidnet110',
                                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                                 'vitb16','efficientnet_v2m','densenet121','densenet169','convnexttiny','convnextlarge','convnextbase','convnextsmall','convnexttiny4small','convnexttiny4large','convnexttiny4base'])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'speechcommands', 'imagenet', 'squad', 'tinyimagenet'], help='dataset')


    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)
 
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def main():
    start_time = time.time()
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    suppress_print_if_not_rank0()
    best_acc = 0
    no_improvement_counter = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data, train_sampler = get_cifar100_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            is_instance=True,
            resize_input=opt.resize_input,
            strong_aug=opt.strong_aug,
            ddp=True,
            local_rank=local_rank
        )
        n_cls = 100
    elif opt.dataset == 'speechcommands':
        from dataset.speechcommands import get_speechcommands_dataloaders
        train_loader, val_loader, labels = get_speechcommands_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers
        )
        n_cls = len(labels)
        train_sampler = None
    elif opt.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloader(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers
        )
        n_cls = 1000
        train_sampler = None
    elif opt.dataset == 'squad':
        from dataset.squad_loader import get_squad_dataloaders
        train_loader, val_loader = get_squad_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers
        )
        n_cls = None
        train_sampler = None
    elif opt.dataset == 'tinyimagenet':
        train_loader, val_loader, n_data, train_sampler = get_tinyimagenet_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            is_instance=True,
            strong_aug=opt.strong_aug,
            ddp=True,
            resize_input=opt.resize_input,
            local_rank=local_rank
        )
        n_cls = 200
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls) if n_cls is not None else model_dict[opt.model]()
    device = torch.device(f"cuda:{local_rank}")
    model = model_dict[opt.model](num_classes=n_cls) if n_cls is not None else model_dict[opt.model]()
    model = model.to(device)  # ? move the entire model to the correct GPU
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    module_list = nn.ModuleList([model])

    # loss
    if opt.dataset == 'squad':
        criterion = None
    else:
        if opt.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(local_rank)

    # optimizer
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(),
                          lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    # scheduler
    def warmup_cosine_schedule(epoch):
        if epoch < opt.warmup_epochs:
            return float(epoch + 1) / float(opt.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * (epoch - opt.warmup_epochs) / (opt.epochs - opt.warmup_epochs)))

    if opt.lr_scheduler == 'cosine':
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    elif opt.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=opt.lr_decay_epochs[0], gamma=opt.lr_decay_rate)
    else:
        scheduler = None

    # optional: freeze backbone if requested
    if opt.freeze_backbone:
        for name, param in model.module.named_parameters():
            if 'head' not in name and 'classifier' not in name:
                param.requires_grad = False

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    for epoch in range(1, opt.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print("==> training...")
        time1 = time.time()

        if opt.dataset == 'squad':
            train_acc, train_loss = train_squad(epoch, train_loader, model, criterion, optimizer, opt)
        else:
            train_acc, train_loss = train(
                epoch, train_loader, model, criterion, optimizer, opt,
            )

        if scheduler:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)

        time2 = time.time()
        print(f'epoch {epoch}, total time {time2 - time1:.2f}s')

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        if opt.dataset == 'squad':
            test_acc, test_acc_top5, test_loss = validate_squad(val_loader, model, criterion, opt)
        else:
            test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            no_improvement_counter = 0
            state = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f'{opt.model}_best.pth')
            print('saving the best model!')
            torch.save(state, save_file)
        else:
            no_improvement_counter += 1

        if opt.early_stopping > 0 and no_improvement_counter >= opt.early_stopping:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            torch.save(state, save_file)

    print('best accuracy:', best_acc)

    # final save
    state = {
        'opt': opt,
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, f'{opt.model}_last.pth')
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
