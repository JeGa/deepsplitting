import torch
import torchvision


def get_sampler(N, dataset):
    if N == -1:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        batch_size = len(dataset)
    else:
        sampler = SequentialSubsetSampler(N)
        # TODO: This shuffles.
        #sampler = torch.utils.data.sampler.SubsetRandomSampler(list(range(N)))
        batch_size = N

    return sampler, batch_size


class SequentialSubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, N):
        self.N = N

    def __iter__(self):
        return iter(range(self.N))

    def __len__(self):
        return self.N


class To64fTensor:
    def __call__(self, x):
        tensor = torchvision.transforms.ToTensor()(x)
        return tensor.double()
