from module import Module


class ReLU(Module):
    def forward(self, input):
        self.input = input
        return input * (input > 0)

    def backward(self, gradwrtoutput):
        drelu = 1 * (self.input > 0)
        return gradwrtoutput * drelu

    def params(self):
        return []


class Tanh(Module):
    def forward(self, input):
        self.input = input
        return input.tanh()

    def backward(self, gradwrtoutput):
        dtanh = 1 - self.input.tanh().pow(2)
        return gradwrtoutput * dtanh

    def params(self):
        return []


class Sigmoid(Module):
    def forward(self, input):
        self.input = input
        return input.sigmoid()

    def backward(self, gradwrtoutput):
        dsigmoid = self.input.sigmoid() * (1 - self.input.sigmoid())
        return gradwrtoutput * dsigmoid

    def params(self):
        return []
