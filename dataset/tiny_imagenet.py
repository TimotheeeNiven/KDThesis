from __future__ import print_function

import os
import socket
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.autoaugment import RandAugment
from torch.utils.data.distributed import DistributedSampler


def get_data_folder():
    hostname = socket.gethostname()
    if hostname.startswith("visiongpu"):
        data_folder = "/data/vision/phillipi/rep-learn/datasets"
    elif hostname.startswith("yonglong-home"):
        data_folder = "/home/yonglong/Data/data"
    else:
        data_folder = "./data/"
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    return data_folder


class TinyImageNetInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_tinyimagenet_dataloaders(batch_size=128, num_workers=8, is_instance=False,
                                 strong_aug=False, resize_input=False, ddp=False, local_rank=0):
    import torch.distributed as dist

    data_folder = os.path.join(get_data_folder(), "imagenet_tiny/tiny-imagenet-200")

    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2770, 0.2691, 0.2821)

    if resize_input:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            RandAugment() if strong_aug else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=4),
            RandAugment() if strong_aug else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    train_dir = os.path.join(data_folder, "train")
    val_dir = os.path.join(data_folder, "val", "images")

    if is_instance:
        train_set = TinyImageNetInstance(root=train_dir, transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.ImageFolder(root=train_dir, transform=train_transform)
        n_data = None

    test_set = datasets.ImageFolder(root=val_dir, transform=test_transform)

    # Samplers
    if ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
        test_sampler = DistributedSampler(test_set, shuffle=False, rank=local_rank)
        shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=int(batch_size / 2),
        shuffle=False,
        sampler=test_sampler,
        num_workers=int(num_workers / 2),
        pin_memory=True
    )

    if is_instance:
        return train_loader, test_loader, n_data, train_sampler
    else:
        n_data = len(train_set)
        return train_loader, test_loader, n_data, train_sampler
