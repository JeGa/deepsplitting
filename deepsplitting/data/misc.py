import torch


def get_sampler(N, dataset):
    if N == -1:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
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
