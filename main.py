import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import hamming
import joblib
import os


from data_loading import train_test_mushroom_data, train_test_wine_data
from neural_net.model import NeuralNet


def distance(u, v):
    return hamming(u,v)

if __name__ == "__main__":
    seed = 42
    data_loading_dict = {'wine': train_test_wine_data, 'mushroom': train_test_mushroom_data}
    for data_name, loading_fct in data_loading_dict.items():
        X_train, X_test, y_train, y_test = loading_fct(
            test_size=0.25, shuffle=True, random_state=seed
        )
        for model_name in ['neural_net', 'benchmark']:
            filename = f'{model_name}/hyperparam_tuning/{data_name}/study.pkl'
            study = joblib.load(filename)
            params = study.best_params
            if model_name == 'neural_net':
                model = NeuralNet(n_features=X_train.shape[1], n_label=len(y_train.unique()), **params)
            elif model_name == 'benchmark':
                model = KNeighborsClassifier(weights="distance", metric=hamming, **params) if data_name == 'mushroom' else KNeighborsClassifier(weights="distance", **params)

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
            saving_dir = f'results/{model_name}/{data_name}'
            os.makedirs(saving_dir, exist_ok=True)
            # pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T.to_csv(f'{saving_dir}/test.csv')
            # pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)).T.to_csv(f'{saving_dir}/train.csv')





