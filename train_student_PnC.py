"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time
import csv
from datetime import datetime

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.tiny_imagenet import get_tinyimagenet_dataloaders
from dataset.speechcommands import get_speechcommands_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.squad_loader import get_squad_dataloaders


from helper.util import adjust_learning_rate, update_ema_teacher_safe

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.loops_qa import train_squad_distill as train_squad, validate_squad

from helper.pretrain import init

def move_to_device(model):
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def suppress_print_if_not_rank0():
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank != 0:
            import builtins as __builtin__
            def no_op(*args, **kwargs): pass
            __builtin__.print = no_op

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')
    
    # Overfitting control
    parser.add_argument('--cutmix', action='store_true', help='Use CutMix augmentation')
    parser.add_argument('--mixup', action='store_true', help='Use MixUp augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (if model supports it)')
    parser.add_argument('--strong_aug', action='store_true', help='Use stronger data augmentation (RandAugment)')
    parser.add_argument('--early_stopping', type=int, default=0, help='Enable early stopping after N epochs of no improvement')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Clip gradients to this value')
    parser.add_argument('--resize_input', action='store_true', help='Resize CIFAR-100 inputs to 224x224')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='number of warmup epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased', help='HuggingFace tokenizer name')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw', 'adamwsf'], help='optimizer type')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine'],
                    help='type of learning rate scheduler to use')
    #ema patient and consistent flags
    parser.add_argument('--ema_teacher', action='store_true',
                    help='Use EMA (temporal) teacher built from the student')
    parser.add_argument('--ema_tau', type=float, default=0.999,
                    help='EMA decay for teacher update')
    parser.add_argument('--ema_from_pretrained', action='store_true',
                    help='Start EMA teacher from a pre-trained teacher (Patient & Consistent style)')
    parser.add_argument('--ema_warmup_epochs', type=int, default=0,
                    help='Delay EMA updates for N epochs')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'speechcommands', 'imagenet', 'squad', 'tinyimagenet'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet101', 'resnet152', 'regnety4gf', 'regnety16gf', 'regnety32gf',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'wrn_40_4', 'wrn_28_10', 'wrn_28_1', 'wrn_28_4', 'wrn_20_4', 'wrn_40_10', 'wrn_10_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'roberta', 'distilbert', 'tinybert', 'mobilebert', 'albert', 'bertbase', 'bertsmall',
                                 'vit_tiny', 'vit_small', 'vit_base', 'vit_medium', 'vit_distilled',
                                 'MobileNetV1','MobileNetV2', 'ShuffleV1', 'MobileNetV3.1', 'ShuffleV2', 'wide', 'ResNet50','MobileNetV3','ViT_B_16',
				 'resnext50_32x4d','resnext101_32x8d','pyramidnet272', 'pyramidnet110',
                                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                                 'vitb16','efficientnet_v2m','densenet121','densenet169','convnexttiny','convnexttiny4large','convnexttiny4small','convnexttiny4base','convnextsmall'])


    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    opt.tb_folder = os.path.join(opt.tb_path, f"{opt.model_name}_{timestamp}")
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder,exist_ok=True)

    opt.save_folder = os.path.join(opt.model_path, f"{opt.model_name}_{timestamp}")
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder,exist_ok=True)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    elif segments[0] == 'vit':
        return segments[0] + '_' + segments[1]
    elif segments[0] == 'efficientnet':
        return segments[0] + '_' + segments[1]
    else:
        return segments[0]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    if n_cls != None:
    	model = model_dict[model_t](num_classes=n_cls)
    else:
        model = model_dict[model_t]()
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model

def log_training_info(technique_used, teacher_name, teacher_accuracy, student_name, student_accuracy, training_time, output_dir):
    # Directory path
    output_file = os.path.join(output_dir, "training_info.csv")

    # Check if the CSV file exists
    file_exists = os.path.isfile(output_file)

    # Writing to CSV
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header if file is new
        if not file_exists:
            writer.writerow(['Technique', 'Teacher Name', 'Teacher Accuracy', 'Student Name', 'Student Accuracy', 'Training Time'])

        # Write the extracted data
        writer.writerow([technique_used, teacher_name, teacher_accuracy, student_name, student_accuracy, training_time])

    print(f"Data successfully written to {output_file}")

def write_to_csv(file_path, data):
    """Write results to a CSV file."""
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file is new
        if not file_exists:
            writer.writerow(['Teacher Model', 'Student Model', 'Teacher Accuracy', 'Student Accuracy', 'Training Time'])
        writer.writerow(data)

def main():
    start_time = time.time()
    dist.init_process_group(backend='nccl')  # Use 'gloo' for CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    suppress_print_if_not_rank0()
    best_acc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data, train_sampler = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
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
        train_loader, val_loader, labels = get_speechcommands_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = len(labels)
    elif opt.dataset == 'imagenet':
        train_loader, val_loader = get_imagenet_dataloader(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers
        ) 
        n_cls = 1000
    elif opt.dataset == 'squad':
        from dataset.squad_loader import get_squad_dataloaders
        train_loader, val_loader = get_squad_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = None  # SQuAD does not use classification head

    elif opt.dataset == 'tinyimagenet':
        train_loader, val_loader, n_data, train_sampler = get_tinyimagenet_dataloaders(
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            is_instance=True,
            resize_input=opt.resize_input,
            strong_aug=opt.strong_aug,
            ddp=True,
            local_rank=local_rank
        )
        n_cls = 200


    else:
        raise NotImplementedError(opt.dataset)

    # model
    # Student (DDP)
    model_s = model_dict[opt.model_s](num_classes=n_cls).to(local_rank) if n_cls is not None else model_dict[opt.model_s]().to(local_rank)
    model_s = DDP(model_s, device_ids=[local_rank], find_unused_parameters=True)

    # Teacher: load PRETRAINED model; will be EMA-updated toward the student during training
    assert opt.path_t is not None, "--path_t is required for pretrained EMA teacher"
    model_t = load_teacher(opt.path_t, n_cls).to(local_rank)   # do NOT wrap teacher in DDP
    for p in model_t.parameters():
        p.requires_grad_(False)  # teacher updated only via EMA (no gradients)
    model_t.eval()
    print('==> Using PRETRAINED EMA teacher (patient & consistent style)')


    # Model preparation
    module_list = nn.ModuleList([model_s])
    trainable_list = nn.ModuleList([model_s])
    
    # Loss setup
    if opt.label_smoothing > 0:
        criterion_cls = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
    else:
        criterion_cls = nn.CrossEntropyLoss()

    criterion_div = DistillKL(opt.kd_T)

    # Distillation-specific setup
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.extend([criterion_kd.embed_s, criterion_kd.embed_t])
        trainable_list.extend([criterion_kd.embed_s, criterion_kd.embed_t])
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.extend([embed_s, embed_t])
        trainable_list.extend([embed_s, embed_t])
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList([VIDLoss(s, t, t) for s, t in zip(s_n, t_n)])
        trainable_list.append(criterion_kd)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])
    # optimizer
    if opt.optimizer == 'sgd':
        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adamw':
        optimizer = AdamW(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          weight_decay=opt.weight_decay)
    
    elif opt.optimizer == 'adamwsf':
        from schedulefree import AdamWScheduleFree
        warmup_steps = opt.warmup_epochs * len(train_loader)
        optimizer = AdamWScheduleFree(
            trainable_list.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
            warmup_steps=warmup_steps
        )
    if hasattr(optimizer, "train"):
        optimizer.train()
    if opt.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
    else:
        scheduler = None

    if torch.cuda.is_available():
        module_list = module_list.to(local_rank)
        trainable_list = trainable_list.to(local_rank)
        criterion_list = criterion_list.to(local_rank)
        cudnn.benchmark = True

    # Validate teacher model accuracy
    if opt.dataset == 'squad':
        teacher_acc, _, _ = validate_squad(val_loader, model_t, None, opt)
    else:
        teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)

    print('Teacher Accuracy: {:.2f}%'.format(teacher_acc))

    # Training loop
    for epoch in range(1, opt.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        print(f"==> Training epoch {epoch}...")
        start_epoch_time = time.time()
   
        if opt.optimizer != 'adamwsf' and epoch <= opt.warmup_epochs:
            warmup_lr = opt.learning_rate * epoch / opt.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"Warm-up epoch {epoch}: lr set to {warmup_lr:.6f}")

        if opt.dataset == 'squad':
            train_acc, train_loss = train_squad(epoch, train_loader, module_list + [model_t], criterion_list, optimizer, opt)
            val_acc, _, _ = validate_squad(val_loader, model_s, None, opt)
        else:
            train_acc, train_loss = train(epoch, train_loader, module_list + [model_t], criterion_list, optimizer, opt)
            val_acc, _, _ = validate(val_loader, model_s, criterion_cls, opt)
        if opt.optimizer == 'adamwsf':
            pass
        elif scheduler:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)

        epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s with validation accuracy: {val_acc:.2f}%")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(opt.save_folder, f"{opt.model_s}_best.pth")
            torch.save({'model': model_s.state_dict(), 'accuracy': best_acc}, save_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

    # Training completed
    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.2f}s with best student accuracy: {best_acc:.2f}%")

    # Save results to CSV
    results_file = 'student_output.csv'
    write_to_csv(results_file, [
        opt.model_t,           # Teacher model
        opt.model_s,           # Student model
        opt.distill,           # Distill Method
        f"{teacher_acc:.2f}",  # Teacher accuracy
        f"{best_acc:.2f}",     # Student accuracy
        f"{total_training_time:.2f}",  # Training time
        opt.dataset
    ])
    if opt.optimizer == 'adamwsf' and hasattr(optimizer, 'eval'):
        optimizer.eval()
    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()