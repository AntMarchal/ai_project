from data_loading import train_test_mushroom_data, train_test_wine_data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import optuna
import os
import plotly
import joblib


from neural_net.model import NeuralNet

class Tuner():
    def __init__(self, Model, X_train, y_train, splitter, metric='f1-score'):
        self.X_train = X_train
        self.y_train = y_train
        self.splitter = splitter
        self.Model = Model
        self.n_features = X_train.shape[1]
        self.n_label = len(y_train.unique())
        self.metric = metric

    def cross_validation(self, params):
        scores = []
        for train_index, test_index in self.splitter.split(self.X_train, self.y_train):
            X_train = self.X_train.iloc[train_index]
            y_train = self.y_train.iloc[train_index]
            X_test = self.X_train.iloc[test_index]
            y_test = self.y_train.iloc[test_index]
            neural_net = self.Model(self.n_features, self.n_label, **params)
            neural_net.fit(X_train, y_train)
            y_pred = neural_net.predict(X_test)
            scores.append(classification_report(y_test, y_pred, output_dict=True)['weighted avg'][self.metric])

        score_ = sum(scores) / len(scores) #TODO: maybe take into consideration the std
        return score_

    def objective(self, trial):
        params = {
            'n_epochs': trial.suggest_int('n_epochs', 25, 60, 5),
            'lr': trial.suggest_float('lr', 1e-3, 1),
            'batch_size': trial.suggest_int('batch_size', 25, 120, 5),
            'momentum': trial.suggest_float('momentum', 0, 1),
            'n_hidden': trial.suggest_int('n_hidden', 50, 200, 5),
                }
        cv_score = self.cross_validation(params)
        return cv_score

    def tune(self, trials_num=200):
        compt = 0
        best_params = {}
        self.study = optuna.create_study(direction='maximize')
        for i in range(trials_num // os.cpu_count()):  # n_jobs deprecated, use all the cpu automatically
            self.study.optimize(self.objective, n_trials=os.cpu_count(), n_jobs=-1
                                )
            if self.study.best_params == best_params:
                compt += 1
            else:
                compt = 0
                best_params = self.study.best_params.copy()
            if compt * os.cpu_count() > 100:
                print('Stop the parameter tuning after {} trials'.format(i * os.cpu_count()))
                break
        joblib.dump(self.study, 'neural_net/hyperparam_tuning/study.pkl')
        return self.study.best_params

    def study_visualization(self):
        if self.study is None:
            raise Exception('The model must be trained first!')
        plotly.offline.plot(optuna.visualization.plot_optimization_history(self.study),
                            filename='neural_net/hyperparam_tuning/plot_optimization_history.html')
        plotly.offline.plot(optuna.visualization.plot_slice(self.study),
                            filename='neural_net/hyperparam_tuning/plot_slice.html')
        plotly.offline.plot(optuna.visualization.plot_param_importances(self.study),
                            filename='neural_net/hyperparam_tuning/plot_param_importances.html')





if __name__ == '__main__':
    seed = 42
    X_train, X_test, y_train, y_test = train_test_wine_data(
        test_size=0.25, shuffle=True, random_state=seed
    )
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    tuner = Tuner(NeuralNet, X_train, y_train, splitter=sss, metric='f1-score')
    tuner.tune()
    print(tuner.study.best_params)
    tuner.study_visualization()

