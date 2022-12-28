

class SGD():
    def __init__(self, params, lr=1e-1):
        self.params = params
        self.lr = lr

    def step(self):
        for p, grad in self.params:
            p -= self.lr * grad

