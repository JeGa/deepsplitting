import logging

from deepsplitting.optimizer.base import BaseOptimizer
import deepsplitting.utils.global_config as global_config
import deepsplitting.utils.testrun as testrun
import deepsplitting.utils.global_progressbar as gp


class Optimizer(BaseOptimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer, self).__init__(net, hyperparams)

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer, self).init_parameters(initializer, parameters)

    def step(self, inputs, labels):
        loss = self.net.loss(inputs, labels)
        loss.backward()

        for p in self.net.parameters():
            if p.grad is None:
                continue

            p.data.add_(-self.hyperparams.lr, p.grad.data)

        return loss.item(), self.net.loss(inputs, labels).item()


class Optimizer_batched(BaseOptimizer):
    """
    This version does the batching itself. It assumes as input the full batch samples.
    """

    def __init__(self, net, N, hyperparams):
        super(Optimizer_batched, self).__init__(net, hyperparams)

        self.iteration = 1

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer_batched, self).init_parameters(initializer, parameters)

        self.iteration = 1

    def step(self, inputs, labels):
        batch_size = global_config.cfg.training_batch_size

        indices, _, max_steps = self.fullbatch_subindex_init(batch_size, 0, inputs.size(0))

        loss = []
        correctly_classified = []

        loss.append(self.loss_chunked(inputs, labels))

        for i, index in enumerate(self.batches(indices, batch_size), 1):
            self.gd_step(inputs[index], labels[index])

            current_loss = self.loss_chunked(inputs, labels)

            logging.info("{} (Batch step [{}/{}]): Data loss = {:.6f}".format(
                type(self).__module__, i, max_steps, current_loss))

            correct = testrun.test_at_interval(self.net, self.iteration - 1, inputs, labels,
                                               global_config.cfg.classes)
            if correct is not None:
                correctly_classified.append(correct)

            loss.append(current_loss)

            self.iteration += 1

            gp.bar.next_batch(dict(dataloss=current_loss))

        return loss, correctly_classified

    def gd_step(self, inputs, labels):
        loss = self.net.loss(inputs, labels)

        self.zero_grad()
        loss.backward()

        for p in self.net.parameters():
            if p.grad is None:
                continue

            lr = self.hyperparams.lr
            if self.hyperparams.vanstep:
                lr = (1 / (self.iteration + (1 / self.hyperparams.lr))) + self.hyperparams.min_stepsize

            p.data.add_(-lr, p.grad.data)

        return loss.item(), self.net.loss(inputs, labels).item()
