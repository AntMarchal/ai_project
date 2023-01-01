from data_loading import train_test_mushroom_data, train_test_wine_data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import optuna
import os


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
            'n_epochs': trial.suggest_int('n_epochs', 5, 20, 5),
            'lr': trial.suggest_loguniform('lr', 1e-3, 1e2),
                }
        cv_score = self.cross_validation(params)
        return cv_score

    def tune(self, trials_num=100):
        compt = 0
        best_params = {}
        self.study = optuna.create_study(direction='maximize')
        for i in range(trials_num // os.cpu_count()):  # n_jobs deprecated, use all the cpu automatically
            self.study.optimize(self.objective, n_trials=os.cpu_count(),
                                )
            if self.study.best_params == best_params:
                compt += 1
            else:
                compt = 0
                best_param = self.study.best_params.copy()
            if compt * os.cpu_count() > 20:
                print('Stop the parameter tuning after {} trials'.format(i * os.cpu_count()))
                break
        return self.study.best_params



if __name__ == '__main__':
    seed = 42
    X_train, X_test, y_train, y_test = train_test_wine_data(
        test_size=0.25, shuffle=True, random_state=seed
    )
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    tuner = Tuner(NeuralNet, X_train, y_train, splitter=sss, metric='f1-score')
    tuner.tune()

