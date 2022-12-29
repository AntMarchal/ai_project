import torch
from tqdm import tqdm


from layers import Linear
from sequential import Sequential
from activation_fcts import ReLU, Sigmoid
from losses import MSELoss
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
        self.criterion = MSELoss()


    def fit(self, X_train, y_train):
        # Normalization (X_train, X_test will be normalised by the same mu and std)
        self.mu, self.std = X_train.mean(0), X_train.std(0)
        X_train.sub_(self.mu).div_(self.std)

        for e in tqdm(range(self.n_epochs)):
            epoch_loss = 0
            for i in range(X_train.size(0)):
                output = self.model.forward(X_train.narrow(0, i, 1).view(-1, 1))
                epoch_loss += self.criterion.forward(output, y_train.narrow(0, i, 1)).item()
                self.model.zero_grad()
                gradwrtoutput = self.criterion.backward(output, y_train.narrow(0, i, 1))
                self.model.backward(gradwrtoutput)
                self.optimizer.step()

            output = (self.model.forward(X_train.narrow(0, i, 1).view(-1, 1)).item() > 0.5) * 1

            # train_accuracy = compute_accuracy(model, input_train, target_train)
            # test_accuracy = compute_accuracy(model, input_test, target_test)
            # train_accuracies.append(train_accuracy)
            # test_accuracies.append(test_accuracy)
            # losses.append(epoch_loss)
            # print(
            #     f'Epoch #{e}, epoch_loss={epoch_loss}, train_accuracy={train_accuracy}, test_accuracy={test_accuracy}')

    def predict(self, X_test):
        X_test.sub_(self.mu).div_(self.std)



