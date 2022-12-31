from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch
import numpy as np

from layers import Linear
from sequential import Sequential
from activation_fcts import ReLU, Sigmoid
from losses import CrossEntropyLoss
from optimizers import SGD

#  set gradient calculation to off
torch.set_grad_enabled(False)


class NeuralNet():
    def __init__(self, n_features, n_label, n_epochs=25, lr=1e-1):
        self.n_epochs = n_epochs
        self.model = Sequential(
                                Linear(n_features, 25),
                                ReLU(),
                                Linear(25, 25),
                                ReLU(),
                                Linear(25, 25),
                                ReLU(),
                                Linear(25, n_label),
                               )

        self.optimizer = SGD(self.model.params(), lr=lr)
        self.criterion = CrossEntropyLoss()


    def fit(self, X_train, y_train):
        # convert to torch tensor
        self.X_train = torch.tensor(X_train.values, dtype=torch.float32)
        # one hot encoding of the target
        self.encoder = OneHotEncoder(sparse=False)
        self.encoder.fit(y_train.values.reshape(-1,1))
        self.y_train = self.encoder.transform(y_train.values.reshape(-1,1))
        self.y_train = torch.tensor(self.y_train)

        # Normalization (X_train, X_test will be normalised by the same mu and std)
        self.mu, self.std = self.X_train.mean(0), self.X_train.std(0)
        self.X_train.sub_(self.mu).div_(self.std)

        for e in tqdm(range(self.n_epochs)):
            epoch_loss = 0
            for i in range(self.X_train.size(0)):
                output = self.model.forward(self.X_train.narrow(0, i, 1).view(-1, 1))
                epoch_loss += self.criterion.forward(output, self.y_train.narrow(0, i, 1)).item()
                self.model.zero_grad()
                gradwrtoutput = self.criterion.backward(output, self.y_train.narrow(0, i, 1))
                self.model.backward(gradwrtoutput)
                self.optimizer.step()


            print(
                f'Epoch #{e}, epoch_loss={epoch_loss}')

    def predict(self, X_test):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        X_test.sub_(self.mu).div_(self.std)

        class_preds = []
        for i in range(X_test.size(0)):
            output = self.model.forward(X_test.narrow(0, i, 1).view(-1, 1))
            probs = output.exp().div(output.exp().sum())
            one_hot_class_pred = np.zeros_like(probs.view(1,-1))
            one_hot_class_pred[:, probs.argmax().item()] = 1
            class_preds.append(self.encoder.inverse_transform(one_hot_class_pred).item())

        return class_preds



