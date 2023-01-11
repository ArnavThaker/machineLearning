import sys
import scipy
import numpy as np
import matplotlib as plt
import pandas as pd
import sklearn

class IrisFlowers():
    def __init__(self):
        pass

    def read_data(self):
        names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
        data = pd.read_csv('./iris.csv', names=names)
        return data

    def observe_data(self, dataset):
        print("Dataset dimensions are: ", dataset.shape)
        print("First 20 rows of data: \n", dataset.head(20))
        print("General statistics of dataset: \n", dataset.describe())
        print("Grouping by class (species): \n", dataset.groupby("species").size())

if __name__ == '__main__':
    iris = IrisFlowers()
    iris_dataframe = iris.read_data()
    iris.observe_data(iris_dataframe)


