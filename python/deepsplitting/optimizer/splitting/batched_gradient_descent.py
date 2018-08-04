from deepsplitting.optimizer.splitting.batched_primal2 import Optimizer_batched


class Optimizer(Optimizer_batched):
    def __init__(self, net, N, hyperparams):
        super(Optimizer, self).__init__(net, N, hyperparams)

    def gd_step(self, index, subindex, inputs):
        y_batch = self.forward(inputs[index], requires_grad=True)
        J1 = self.jacobian_torch(y_batch).detach()

        B, _ = self.linear_system_B(index, y_batch, J1)

        return self.vec_to_params_update(self.hyperparams.rho * self.hyperparams.stepsize * B, from_numpy=False)

    def primal2(self, inputs, labels, index, subindex):
        new_params = self.gd_step(index, subindex, inputs)
        self.restore_params(new_params)
