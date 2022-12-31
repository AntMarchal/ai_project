import numpy as np
import torch
from data_loading import train_test_wine_data

if __name__ == '__main__':
    seed = 42
    X_train, X_test, y_train, y_test = train_test_wine_data(
        test_size=0.25, shuffle=True, random_state=seed
    )
    a = np.ones((2979, 11))
    torch.tensor(a)
    b = torch.tensor(X_train.values)
    b.mean()