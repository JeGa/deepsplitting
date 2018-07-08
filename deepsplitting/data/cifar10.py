import torch
import torchvision

from deepsplitting.data.misc import get_sampler


def load_CIFAR10(training_samples=-1, test_samples=-1,
                 normalize_transform=torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 folder='../data'):
    """
    Load CIFAR10 data set. Data is always sampled in the same order and full batch.
    Default normalization is to the range [-1, 1].
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                normalize_transform])

    trainset = torchvision.datasets.CIFAR10(root=folder, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=folder, train=False, download=True, transform=transform)

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
