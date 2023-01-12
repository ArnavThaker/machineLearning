import sys
import scipy
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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

    def validation_data(self, dataset):
        data = dataset.values
        x = data[:, 0:4]
        y = data[:, 4]
        models = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=1)
        models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
        models.append(("LDA", LinearDiscriminantAnalysis()))
        models.append(("KNN", KNeighborsClassifier()))
        models.append(("CART", DecisionTreeClassifier()))
        models.append(("NB", GaussianNB()))
        models.append(("SVM", SVC(gamma="auto")))
        for name, model in models:
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
            cv_results = cross_val_score(model, X=x_train, y=y_train, cv=kfold, scoring="accuracy")
            print("Accuracy for {} is {:.2f}% with standard deviation {}\n".format(
                name, cv_results.mean().item() * 100, cv_results.std() * 100))




if __name__ == '__main__':
    iris = IrisFlowers()
    iris_dataframe = iris.read_data()
    iris.observe_data(iris_dataframe)
    #iris.univariate_visualizations(iris_dataframe)
    #iris.multivariate_visualizations(iris_dataframe)
    iris.validation_data(iris_dataframe)


