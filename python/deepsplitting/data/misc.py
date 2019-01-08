import torch
import torchvision


def get_sampler(N, dataset):
    """
    If
        N == -1: Sample full batch.
        N != -1: Sample full batch but only a subset of the data.

    The data is always in the same order. When using this function, the sampling has to be done manually.

    :return: Sampler and batch size.
    """
    if N == -1:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        batch_size = len(dataset)
    else:
        sampler = SequentialSubsetSampler(N)
        batch_size = N

    return sampler, batch_size


class SequentialSubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, N):
        self.N = N

    def __iter__(self):
        return iter(range(self.N))

    def __len__(self):
        return self.N


class To64fImageTensor:
    def __call__(self, x):
        tensor = torchvision.transforms.ToTensor()(x)
        return tensor.double()


class To64fTensor:
    def __call__(self, x):
        return torch.from_numpy(x).double()
