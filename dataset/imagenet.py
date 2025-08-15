from __future__ import print_function
import os
import socket
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_data_folder():
    """
    Return server-dependent path to store the data.
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets/imagenet'
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data/imagenet'
    else:
        data_folder = '/users/rniven1/GitHubRepos/RepDistiller/data/imagenet'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder

def safe_loader(path):
    """Safely load an image, skipping corrupted ones."""
    try:
        img = Image.open(path).convert('RGB')
        return img
    except (OSError, Image.DecompressionBombError) as e:
        print(f"Warning: Skipping corrupted image {path}. Error: {e}")
        return None

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = safe_loader(path)
        if sample is None:
            return self.__getitem__((index + 1) % len(self.samples))  # Recursively try the next image
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def get_imagenet_dataloader(dataset='imagenet', batch_size=128, num_workers=4):
    """
    Data Loader for ImageNet with directory-based class folders.
    Assumes structure like:
      - /imagenet/train/n01440764/*.JPEG
      - /imagenet/val/n01440764/*.JPEG
    """
    if dataset != 'imagenet':
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    data_folder = get_data_folder()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(data_folder, 'train')
    val_dir = os.path.join(data_folder, 'val')

    train_set = SafeImageFolder(train_dir, transform=train_transform)
    val_set = SafeImageFolder(val_dir, transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers // 2,
                            pin_memory=True)

    print(f"Loaded {len(train_set)} training images across {len(train_set.classes)} classes.")
    print(f"Loaded {len(val_set)} validation images across {len(val_set.classes)} classes.")

    return train_loader, val_loader
