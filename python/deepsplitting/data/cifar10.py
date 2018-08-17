import torch
import torchvision

from deepsplitting.data.misc import get_sampler
from .misc import To64fTensor


def load_CIFAR10(training_samples=-1, test_samples=-1,
                 normalize_transform=torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 folder='datasets', target_transform=None):
    """
    Load CIFAR10 data set. Data is always sampled in the same order and full batch.
    Default normalization is to the range [-1, 1].
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                normalize_transform])

    if target_transform is not None:
        target_transform = torchvision.transforms.Lambda(target_transform)

    trainset = torchvision.datasets.CIFAR10(root=folder, train=True, download=True, transform=transform,
                                            target_transform=target_transform)
    testset = torchvision.datasets.CIFAR10(root=folder, train=False, download=True, transform=transform,
                                           target_transform=target_transform)

    training_sampler, training_batch_size = get_sampler(training_samples, trainset)
    test_sampler, test_batch_size = get_sampler(test_samples, testset)

    trainloader = torch.utils.data.DataLoader(trainset, sampler=training_sampler, batch_size=training_batch_size,
                                              shuffle=False,
                                              num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=test_batch_size,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes, training_batch_size, test_batch_size


def load_CIFAR10_batched(training_batch_size, test_batch_size,
                         normalize_transform=torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         folder='datasets', target_transform=None):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                normalize_transform])

    if target_transform is not None:
        target_transform = torchvision.transforms.Lambda(target_transform)

    trainset = torchvision.datasets.CIFAR10(root=folder, train=True, download=True, transform=transform,
                                            target_transform=target_transform)
    testset = torchvision.datasets.CIFAR10(root=folder, train=False, download=True, transform=transform,
                                           target_transform=target_transform)

    # training_sampler, training_batch_size = get_sampler(10, trainset)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=training_batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes, training_batch_size, test_batch_size
