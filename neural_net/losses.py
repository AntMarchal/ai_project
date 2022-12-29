from module import Module
import torch

class MSELoss(Module):
    def forward(self, output, target):
        return (output - target).pow(2)

    def backward(self, output, target):
        return 2 * (output - target)

    def params(self):
        return[]



class CrossEntropyLoss(Module):
    def forward(self, output, target):
        # softmax
        self.probs = output.exp().div(output.exp().sum(dim=1)) #todo: check dim
        return -self.probs[target].log()

    def backward(self, output, target):
        # The grad is given by p-1 for the class of the target and p for the rest see derivation ??? in report
        grad = self.probs.detach().clone()
        grad[target] -= 1
        return grad

    def params(self):
        return[]
