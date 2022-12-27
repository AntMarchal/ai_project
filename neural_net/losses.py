from module import Module

class MSELoss(Module):
    def forward(self, output, target):
        return (output - target).pow(2)

    def backward(self, output, target):
        return 2 * (output - target)

    def params(self):
        return[]



