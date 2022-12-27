import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def train_test_mushroom_data(
    test_size=0.25, shuffle=True, random_state=None
):
    columns = [
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]
    data = pd.read_csv(
        "data/agaricus-lepiota.data", names=columns, index_col=None
    ).reset_index()
    y = (data["index"] == "e") * 1
    X = data.drop(columns="index")
    # the data contains '?' for stalk-root feature that will be encoded as a 4th category
    one_hot_encoder = OneHotEncoder(sparse=False).fit(X)
    X_one_hot = one_hot_encoder.transform(X)
    X_one_hot = pd.DataFrame(X_one_hot, columns=one_hot_encoder.get_feature_names_out())
    X_train, X_test, y_train, y_test = train_test_split(
        X_one_hot,
        y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def train_test_wine_data(
    test_size=0.25, shuffle=True, random_state=None
):
    data = pd.read_csv("data/winequality-white.csv", sep=";")
    y = data["quality"]
    X = data.drop(columns="quality")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train_test_wine_data()
