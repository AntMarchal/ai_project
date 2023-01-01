import pandas as pd
from sklearn.metrics import classification_report


from data_loading import train_test_mushroom_data, train_test_wine_data
from neural_net.model import NeuralNet





if __name__ == '__main__':
    seed = 42
    X_train, X_test, y_train, y_test = train_test_wine_data(
        test_size=0.25, shuffle=True, random_state=seed
    )
    n_features = X_train.shape[1]
    n_label = len(y_train.unique())
    neural_net = NeuralNet(n_features, n_label, n_epochs=25, lr=1e-1)
    neural_net.fit(X_train, y_train)
    y_pred = neural_net.predict(X_test)
    print(classification_report(y_test, y_pred))
    classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']





