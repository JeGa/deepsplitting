import torch.nn


class WeightedMSELoss(torch.nn.MSELoss):
    def __init__(self, C=0.5, size_average=False):
        super(WeightedMSELoss, self).__init__(size_average=size_average)

        self.C = C

    def forward(self, input, target):
        return self.C * super().forward(input, target)
