import sklearn
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import hamming
from sklearn.metrics import classification_report

from data_loading import train_test_mushroom_data, train_test_wine_data

def distance(u, v):
    return hamming(u,v)


if __name__ == "__main__":

    seed = 42
    # X_train, X_test, y_train, y_test = train_test_mushroom_data(
    #     test_size=0.25, shuffle=True, random_state=seed
    # )
    X_train, X_test, y_train, y_test = train_test_wine_data(
        test_size=0.25, shuffle=True, random_state=seed
    )

    knn = KNeighborsClassifier(
        n_neighbors=3, weights="distance", metric=hamming
    )
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))

