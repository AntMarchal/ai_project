import torch


class SGD():
    def __init__(self, params, lr=1e-1, momentum=0.5):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.old_update = []
        for p, grad in self.params:
            self.old_update.append(torch.zeros_like(grad))

    def step(self):
        for i, (p, grad) in enumerate(self.params):
            update = self.momentum * self.old_update[i] + self.lr * grad
            p -= update
            self.old_update[i] = update.detach().clone()

