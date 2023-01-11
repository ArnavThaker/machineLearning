import sys
import scipy
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
import pandas as pd
from pandas.plotting import scatter_matrix
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

    def univariate_visualizations(self, dataset):
        dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
        pyplot.show()
        dataset.hist()
        pyplot.show()

    def multivariate_visualizations(self, dataset):
        scatter_matrix(dataset)
        pyplot.show()


if __name__ == '__main__':
    iris = IrisFlowers()
    iris_dataframe = iris.read_data()
    iris.observe_data(iris_dataframe)
    iris.univariate_visualizations(iris_dataframe)
    iris.multivariate_visualizations(iris_dataframe)


