from util import My_CIFAR10
import torchvision.transforms as transforms
import torch
from util import SampleImageFolder
import numpy as np
from cifar10.data_loader import load_partition_data_cifar10


def load_cifar10_noniid(split_num, alpha):
    assert split_num > 1
    batch_size = 32

    train_data_num, test_data_num, train_data_global, test_data_global, \
    data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = \
        load_partition_data_cifar10('cifar10', './data/cifar10', 'hetero',
                                    alpha, split_num, batch_size)

    presam_loaders = [train_data_local_dict[i] for i in range(split_num)]

    return presam_loaders, train_data_global, test_data_global


def load_cifar10(split_num=1):
    batch_size = 32
    train_set = My_CIFAR10('datasets/cifar10', train=True, transform=transforms.Compose([
        transforms.Resize(35),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),

    ]), target_transform=None, download=True)

    if split_num > 1:
        subsets = np.array_split(range(len(train_set)), split_num)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=2,
                                                   shuffle=True
                                                   )

        train_datasets = torch.utils.data.random_split(train_set, [len(subset) for subset in subsets])
        presam_loader = [torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=True)
                         for ds in train_datasets]

    else:

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=2,
            # sampler=WeightedRandomSampler(np.array([1 for _ in range(len(train_set))]),
            #                             len(train_set), replacement=True),
            shuffle=True
        )

        presam_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            num_workers=2,
            shuffle=False,
        )

    test_set = My_CIFAR10('datasets/cifar10', train=False, transform=transforms.Compose([
        transforms.Resize(33),
        transforms.RandomCrop(32),
        transforms.ToTensor(),

    ]), target_transform=None, download=True)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True,
    )

    return train_loader, presam_loader, test_loader
