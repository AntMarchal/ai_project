from module import Module
from torch import empty
from math import sqrt


class Linear(Module):
    def __init__(self, n_in: int, n_out: int):
        self.n_in = n_in
        self.n_out = n_out
        # The var of a uniform distribution between a and b is given by (a - b)**2 / 12
        # by choosing a = -b = sqrt(6.0 / float(n_in + n_out)) we obtain var(w) = 2 / (n_in + n_out) which corresponds
        # to the Xavier initializationof the weights
        self.w = empty(n_out, n_in).uniform_(
            -sqrt(6.0 / float(n_in + n_out)), sqrt(6.0 / float(n_in + n_out))
        )
        self.gradwrt_w = empty(n_out, n_in)

        self.b = empty(1, n_out).uniform_(
            -sqrt(6.0 / float(n_in + n_out)), sqrt(6.0 / float(n_in + n_out))
        )
        self.gradwrt_b = empty(1, n_out)

    def forward(self, input):
        self.input = input
        return self.input.mm(self.w.t()) + self.b

    def backward(self, gradwrtoutput):
        gradwrtinput = gradwrtoutput.mm(self.w)
        self.gradwrt_w.add_(gradwrtoutput.t().mm(self.input) / self.input.size(0))
        self.gradwrt_b.add_(gradwrtoutput.mean(axis=0))
        return gradwrtinput

    def params(self):
        return [(self.w, self.gradwrt_w), (self.b, self.gradwrt_b)]
