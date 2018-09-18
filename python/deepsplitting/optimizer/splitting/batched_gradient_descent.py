import torch

from deepsplitting.optimizer.splitting.batched_primal2 import Optimizer_batched
import deepsplitting.utils.global_config as global_config


class Optimizer(Optimizer_batched):
    def __init__(self, net, N, hyperparams):
        super(Optimizer, self).__init__(net, N, hyperparams)

    def current_stepsize(self):
        stepsize = self.hyperparams.stepsize
        if self.hyperparams.vanstep:
            stepsize = 1 / (self.iteration + (1 / self.hyperparams.stepsize))

        return stepsize

    def gd_step(self, index, inputs):
        y_batch = self.forward(inputs[index], requires_grad=True)
        J1 = self.jacobian_torch(y_batch).detach()

        B, _ = self.linear_system_B(index, y_batch, J1)

        stepsize = self.current_stepsize()

        return self.vec_to_params_update(self.hyperparams.rho * stepsize * B, from_numpy=False), B

    def gd_step2(self, index, inputs, labels):
        y_batch = self.forward(inputs[index], requires_grad=True)
        J1 = self.jacobian_torch(y_batch).detach()

        B = J1.t().matmul(torch.reshape(-y_batch + labels[index], (-1, 1)))

        stepsize = self.current_stepsize()

        return self.vec_to_params_update(self.hyperparams.rho * stepsize * B, from_numpy=False), B

    def gd_step3(self, index, inputs, labels):
        loss = self.net.loss(inputs[index], labels[index])

        self.zero_grad()
        loss.backward()

        B = torch.empty(self.numparams(), 1,
                        dtype=global_config.cfg.datatype,
                        device=global_config.cfg.device)

        start_index = 0

        for p in self.net.parameters():
            if p.grad is None:
                continue

            with torch.no_grad():
                psize = p.numel()
                B[start_index:start_index + psize] = torch.reshape(p.grad, (-1, 1))

                start_index = start_index + psize

        B = -B

        stepsize = self.current_stepsize()

        return self.vec_to_params_update(self.hyperparams.rho * stepsize * B, from_numpy=False), B

    def gd_step4(self, index, inputs, labels):
        loss = self.net.loss(inputs[index], labels[index])

        self.zero_grad()
        loss.backward()

        for p in self.net.parameters():
            if p.grad is None:
                continue

            lr = self.current_stepsize()
            p.data.add_(-lr, p.grad.data)

    def primal2(self, inputs, labels, index, _):
        new_params1, B1 = self.gd_step(index, inputs)

        # TODO: Should all be the same when rho=1.
        # new_params2, B2 = self.gd_step2(index, inputs, labels)
        # new_params3, B3 = self.gd_step3(index, inputs, labels)
        # new_params4, B4 = self.gd_step4(index, inputs, labels)

        self.restore_params(new_params1)
