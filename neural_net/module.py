from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def forward(self, input):
        return NotImplementedError

    @abstractmethod
    def backward(self, gradwrtoutput):
        return NotImplementedError

    @abstractmethod
    def params(self):
        return []
