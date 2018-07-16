import torch
import torchvision

from deepsplitting.data.misc import get_sampler


def load_MNIST_vectorized(training_samples=-1, test_samples=-1, folder='data', target_transform=None):
    def flatten(img):
        return torch.reshape(img, (-1,))

    input_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Lambda(flatten)])

    if target_transform is not None:
        target_transform = torchvision.transforms.Lambda(target_transform)

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
