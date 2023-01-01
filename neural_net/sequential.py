

from module import Module


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gradwrtoutput):
        for layer in self.layers[::-1]:
            gradwrtoutput = layer.backward(gradwrtoutput)

    def params(self):
        return [item for layer in self.layers for item in layer.params()]

    def zero_grad(self):
        for _, grad in self.params():
            grad.zero_()





