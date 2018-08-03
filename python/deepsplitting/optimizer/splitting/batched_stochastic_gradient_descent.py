def step_batched(self, inputs, labels):
    """
    This function does the batching. The given inputs and labels should be sampled in full batch and be always
    in the same order.
    """
    L_data_current, Lagrangian_current = self.eval(inputs, labels)

    indices = torch.randperm(self.N)
    batch_size = global_config.cfg.training_batch_size

    loss_batchstep = list()
    L_batchstep = list()
    for index in self.batches(indices, batch_size):
        self.primal2_cg(inputs, index)
        self.primal1(inputs, labels)
        self.dual(inputs)

        L_data_batch, Lagrangian_batch = self.eval(inputs, labels)
        loss_batchstep.append(L_data_batch.item())
        L_batchstep.append(Lagrangian_batch.item())

        logging.info("{}: Batch step: Loss = {:.6f}, Lagrangian = {:.6f}".format(
            type(self).__module__, L_data_batch, Lagrangian_batch))

    L_data_new, Lagrangian_new = self.eval(inputs, labels)

    if Lagrangian_new > Lagrangian_current:
        self.hyperparams.rho = self.hyperparams.rho + self.hyperparams.rho_add

    return L_data_current.item(), L_data_new.item(), \
           Lagrangian_current.item(), Lagrangian_new.item(), \
           loss_batchstep, L_batchstep


def step_fullbatch(self, inputs, labels):
    L_data_current, Lagrangian_current = self.eval(inputs, labels)

    self.primal2_cg(inputs)

    self.primal1(inputs, labels)

    self.dual(inputs)

    L_data_new, Lagrangian_new = self.eval(inputs, labels)

    if Lagrangian_new > Lagrangian_current:
        self.hyperparams.rho = self.hyperparams.rho + self.hyperparams.rho_add

    return L_data_current.item(), L_data_new.item()


def primal2_levmarq_batched_Mfac(self, inputs, labels, index, subindex):
    i = 0
    max_iter = 1

    while True:
        i += 1

        L, _, _, _, y = self.augmented_lagrangian(inputs, labels)

        y_batch = y[index]
        J1 = self.jacobian_torch(y_batch)

        # J2 =
        J2 = self.jacobian_torch(y[index[subindex]])

        while True:
            A = (J2.t().mm(J2) + self.hyperparams.M * torch.eye(J2.size(1), device=global_config.cfg.device))
            B = J1.t().mm(torch.reshape(y_batch, (-1, 1)))

            step, _ = torch.gesv(B, A)
            new_params = self.vec_to_params_update(step, from_numpy=False)

            L_new, _, _, _, y = self.augmented_lagrangian(inputs, labels, new_params)

            logging.debug("levmarq_step: L={:.2f}, L_new={:.2f}, M={}".format(L, L_new, self.hyperparams.M))

            if L < L_new:
                self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
            else:
                self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                self.restore_params(new_params)
                break

        if i == max_iter:
            break


def primal2_levmarq_batched_armijo(self, inputs, labels):
    raise NotImplementedError()


def primal2_levmarq_batched_vanishing(self, inputs, labels):
    raise NotImplementedError()


def primal2_levmarq(self, inputs, labels):
    i = 0
    max_iter = 1

    while True:
        i += 1

        L, _, _, _, y = self.augmented_lagrangian(inputs, labels)
        J = self.jacobian(y)

        while True:
            params = self.levmarq_step(J, y)

            L_new, _, _, _, _ = self.augmented_lagrangian(inputs, labels, params)

            logging.debug("levmarq_step: L={:.2f}, L_new={:.2f}, M={}".format(L, L_new, self.hyperparams.M))

            if L < L_new:
                self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
            else:
                self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                self.restore_params(params)
                break

        if i == max_iter:
            break


def primal2_cg(self, inputs, index=None):
    i = 0
    max_iter = 1

    while True:
        i += 1

        if index is None:
            y = self.net(inputs)
        else:
            y = self.net(inputs[index])

        J = self.jacobian_torch(y)

        params = self.cg_step(J, y, index)
        self.restore_params(params)

        if i == max_iter:
            break


def cg_step(self, J, y, index):
    j = 0
    max_iter = 8

    if index is None:
        R = self.v.detach() - self.lam.detach() / self.hyperparams.rho - y
    else:
        R = self.v.detach()[index] - self.lam.detach()[index] / self.hyperparams.rho - y

    R = torch.reshape(R, (-1, 1))

    # Solve for x: (J.T*J)*x - R.T*J = 0

    b = J.t().matmul(R)
    x = torch.zeros(J.size(1), 1, device=global_config.cfg.device)

    r = b - J.t().matmul(J.matmul(x))
    p = r

    while True:
        j += 1

        alpha = (r.t().matmul(r)) / (p.t().matmul(J.t().matmul(J.matmul(p))))

        x = x + alpha * p
        r_new = r - alpha * J.t().matmul(J.matmul(p))

        beta = (r_new.t().matmul(r_new)) / (r.t().matmul(r))
        p = r_new + beta * p

        r = r_new

        if j == max_iter:
            return self.vec_to_params_update(x, from_numpy=False)


def levmarq_step(self, J, y):
    r = self.v - self.lam.detach() / self.hyperparams.rho - y
    r = torch.reshape(r, (-1, 1)).data.numpy()

    A = J.T.dot(J) + self.hyperparams.M * scipy.sparse.eye(J.shape[1])
    B = J.T.dot(r)

    s = np.linalg.solve(A, B)
    s = np.float32(s)

    param_list = self.vec_to_params_update(s)

    return param_list
