from collections import namedtuple
from .Base import BaseOptimizer

hyperparams = namedtuple('hyperparams', 'lr')


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams=hyperparams(lr=0.01)):
        super(Optimizer, self).__init__(net, hyperparams)

    def step(self, eval):
        for p in self.net.parameters():
            if p.grad is None:
                continue

            p.data.add_(-self.hyperparams.lr, p.grad.data)
