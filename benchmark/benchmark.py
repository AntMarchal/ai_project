import sklearn
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import hamming
from sklearn.metrics import classification_report
import optuna
from sklearn.model_selection import StratifiedShuffleSplit
import plotly
import joblib


from data_loading import train_test_mushroom_data, train_test_wine_data

def distance(u, v):
    return hamming(u,v)


def study_visualization(study, dir):

    plotly.offline.plot(optuna.visualization.plot_optimization_history(study),
                        filename=f'{dir}/plot_optimization_history.html')
    plotly.offline.plot(optuna.visualization.plot_slice(study),
                        filename=f'{dir}/plot_slice.html')
    plotly.offline.plot(optuna.visualization.plot_param_importances(study),
                        filename=f'{dir}/plot_param_importances.html')

if __name__ == "__main__":

    ################################### mushrooms ###################################

    seed = 42
    # X_train, X_test, y_train, y_test = train_test_mushroom_data(
    #     test_size=0.25, shuffle=True, random_state=seed
    # )
    X_train, X_test, y_train, y_test = train_test_mushroom_data(
        test_size=0.25, shuffle=True, random_state=seed
    )

    knn = KNeighborsClassifier(
        n_neighbors=3, weights="distance", metric=hamming
    )
    param = {"n_neighbors": optuna.distributions.IntUniformDistribution(1, 30)}
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    optuna_search = optuna.integration.OptunaSearchCV(knn, param, cv=sss, n_trials=100,
                                                      random_state=seed, scoring='f1_macro')
    optuna_search.fit(X_train, y_train)
    dir = 'benchmark/hyperparam_tuning/mushroom'
    study = optuna_search.study_
    joblib.dump(optuna_search.study_, f'{dir}/study.pkl')
    study_visualization(study, dir)

    y_pred_train = optuna_search.predict(X_train)
    y_pred_test = optuna_search.predict(X_test)

    print(classification_report(y_test, y_pred_test))

