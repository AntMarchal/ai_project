from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import torch
import numpy as np
import math

from layers import Linear
from sequential import Sequential
from activation_fcts import ReLU, Sigmoid
from losses import CrossEntropyLoss
from optimizers import SGD

#  set gradient calculation to off
torch.set_grad_enabled(False)


class NeuralNet():
    def __init__(self, n_features, n_label, n_epochs=25, lr=1e-1, momentum=0.5, batch_size=100, n_hidden=25):
        self.n_epochs = n_epochs
        self.model = Sequential(
                                Linear(n_features, n_hidden),
                                ReLU(),
                                Linear(n_hidden, n_hidden),
                                ReLU(),
                                Linear(n_hidden, n_hidden),
                                ReLU(),
                                Linear(n_hidden, n_label),
                               )

        self.optimizer = SGD(self.model.params(), lr=lr, momentum=momentum)
        self.criterion = CrossEntropyLoss()
        self.batch_size = batch_size


    def fit(self, X_train, y_train):
        # convert to torch tensor
        self.X_train = torch.tensor(X_train.values, dtype=torch.float32)
        # one hot encoding of the target
        self.encoder = OneHotEncoder(sparse=False)
        self.encoder.fit(y_train.values.reshape(-1,1))
        self.y_train = self.encoder.transform(y_train.values.reshape(-1,1))
        self.y_train = torch.tensor(self.y_train)

        # Normalization if not one-hot encoded data (X_train, X_test will be normalised by the same mu and std)
        if (self.X_train.unique() != torch.tensor([0., 1.])).all():
            self.mu, self.std = self.X_train.mean(0), self.X_train.std(0)
            self.X_train.sub_(self.mu).div_(self.std)

        for e in range(self.n_epochs):
            epoch_losses = []
            # shuffle the train set between each epoch
            rnd_perm = torch.randperm(self.X_train.size(0))
            self.X_train = self.X_train[rnd_perm]
            self.y_train = self.y_train[rnd_perm]
            for b in range(0, self.X_train.size(0), self.batch_size):
                X_train_batch = self.X_train[b:b+self.batch_size] if (b+self.batch_size <= self.X_train.size(0)) else self.X_train[b:]
                y_train_batch = self.y_train[b:b+self.batch_size] if (b+self.batch_size <= self.y_train.size(0)) else self.y_train[b:]
                output = self.model.forward(X_train_batch)
                epoch_loss = self.criterion.forward(output, y_train_batch).item()
                if math.isnan(epoch_loss):
                    epoch_loss
                epoch_losses.append(epoch_loss)
                self.model.zero_grad()
                gradwrtoutput = self.criterion.backward(output, y_train_batch)
                self.model.backward(gradwrtoutput)
                self.optimizer.step()


            # print(
            #     f'Epoch #{e}, avg_epoch_loss={sum(epoch_losses) / len(epoch_losses)}')

    def predict(self, X_test):
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        # Normalization
        if (self.X_train.unique() != torch.tensor([0., 1.])).all():
            X_test.sub_(self.mu).div_(self.std)

        output = self.model.forward(X_test)
        probs = output.exp().div(output.exp().sum(axis=1, keepdim=True))
        one_hot_class_pred = np.zeros_like(probs)
        one_hot_class_pred[range(probs.shape[0]), probs.argmax(axis=1, keepdim=False)] = 1
        return self.encoder.inverse_transform(one_hot_class_pred)




