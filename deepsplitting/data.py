import torch
import torchvision


def load_CIFAR10(training_samples=-1, test_samples=-1,
                 normalize_transform=torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 folder='../data'):
    """
    Load CIFAR10 data set. Data is always sampled in the same order and full batch.

    :param training_samples: Number of training samples in training data set.
    :param test_samples: Number of test samples in test data set.
    :param folder: Folder to save data if not already downloaded.
    :param normalize_transform: A transforms.Normalize object used to normalize the inputs.
        Default normalization is to the range [-1, 1].
    :return: trainloader, testloader, classes
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


def load_MNIST_vectorized(training_samples=-1, test_samples=-1, folder='../data', ttransform=None):
    def flatten(img):
        return torch.reshape(img, (-1,))

    input_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Lambda(flatten)])

    if ttransform is not None:
        target_transform = torchvision.transforms.Lambda(ttransform)
    else:
        target_transform = None

    trainset = torchvision.datasets.MNIST(root=folder, train=True, download=True,
                                          transform=input_transform, target_transform=target_transform)
    testset = torchvision.datasets.MNIST(root=folder, train=False, download=True,
                                         transform=input_transform, target_transform=target_transform)

    training_sampler, training_batch_size = get_sampler(training_samples, trainset)
    test_sampler, test_batch_size = get_sampler(test_samples, testset)

    trainloader = torch.utils.data.DataLoader(trainset, sampler=training_sampler, batch_size=training_batch_size,
                                              shuffle=False,
                                              num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=test_batch_size,
                                             shuffle=False,
                                             num_workers=2)

    return trainloader, testloader, training_batch_size, test_batch_size


def get_sampler(N, dataset):
    if N == -1:
        sampler = torch.utils.data.sampler.SequentialSampler()
        batch_size = len(dataset)
    else:
        sampler = SubsetSampler(N)
        batch_size = N

    return sampler, batch_size


class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, N):
        self.N = N

    def __iter__(self):
        return iter(range(self.N))

    def __len__(self):
        return self.N
