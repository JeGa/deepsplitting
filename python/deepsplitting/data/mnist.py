import torch
import torchvision

from deepsplitting.data.misc import get_sampler


def load_MNIST_vectorized(training_samples=-1, test_samples=-1, folder='data', target_transform=None):
    def flatten(img):
        return torch.reshape(img, (-1,))

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Lambda(flatten)])

    if target_transform is not None:
        target_transform = torchvision.transforms.Lambda(target_transform)

    return mnist_loader(training_samples, test_samples, folder, transform, target_transform)


def load_MNIST(training_samples=-1, test_samples=-1, folder='data', target_transform=None,
               normalize_transform=torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
               fullbatch=True, training_batch_size=None, test_batch_size=None):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                normalize_transform])

    if target_transform is not None:
        target_transform = torchvision.transforms.Lambda(target_transform)

    return mnist_loader(training_samples, test_samples, folder, transform, target_transform,
                        fullbatch, training_batch_size, test_batch_size)


def mnist_loader(training_samples, test_samples, folder, transform, target_transform,
                 fullbatch, training_batch_size, test_batch_size):
    trainset = torchvision.datasets.MNIST(root=folder, train=True, download=True,
                                          transform=transform, target_transform=target_transform)

    testset = torchvision.datasets.MNIST(root=folder, train=False, download=True,
                                         transform=transform, target_transform=target_transform)

    training_sampler, training_batch_size_full = get_sampler(training_samples, trainset)
    test_sampler, test_batch_size_full = get_sampler(test_samples, testset)

    if fullbatch:
        training_batch_size = training_batch_size_full
        test_batch_size = test_batch_size_full
    elif training_batch_size is None or test_batch_size is None:
        raise ValueError("Specify training and test batch size.")

    trainloader = torch.utils.data.DataLoader(trainset, sampler=training_sampler, batch_size=training_batch_size,
                                              shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2)

    classes = 10

    return trainloader, testloader, training_batch_size, test_batch_size, classes
