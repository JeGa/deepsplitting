from .base import BaseOptimizer
from .base import Hyperparams


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams=Hyperparams(lr=0.0005)):
        super(Optimizer, self).__init__(net, hyperparams)

    def step(self, inputs, labels):
        loss = self.net.loss(inputs, labels)
        loss.backward()

        for p in self.net.parameters():
            if p.grad is None:
                continue

            p.data.add_(-self.hyperparams.lr, p.grad.data)

        return loss.item(), self.net.loss(inputs, labels).item()
