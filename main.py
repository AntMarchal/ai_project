import pandas as pd
import matplotlib.pyplot as plt





if __name__ == '__main__':
    columns = ['cap-shape',
                'cap-surface', 
                'cap-color',
                'bruises',
                'odor',
                'gill',
                'gill-spacing',
                'gill-size',
                'gill-color',
                'stalk-shape',
                'stalk-root',
                'stalk-surface-above-ring',
                'stalk-surface-below-ring',
                'stalk-color-above-ring',
                'stalk-color-below-ring',
                'veil-type',
                'veil-color',
                'ring-number',
                'ring-type',
                'spore-print-color',
                'population',
                'habitat']

    pd.read_csv('data/agaricus-lepiota.data')

