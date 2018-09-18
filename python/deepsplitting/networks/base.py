import torch


class BaseNetwork(torch.nn.Module):
    def loss(self, inputs, labels):
        return self.criterion(self(inputs), labels)
