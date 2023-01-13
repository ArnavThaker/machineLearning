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


def read_data():
    """
    Function that reads the iris.csv (from local directory) and returns a Pandas DataFrame object

    :return: none
    """
    names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    data = pd.read_csv('./iris.csv', names=names)
    return data


def observe_data(dataset):
    """
    Function that examines properties of the Iris Flowers dataset, including shape of the DataFrame,
    descriptive statistics, and a peek at the DataFrame.

    :param dataset: A Pandas DataFrame object of the Iris Flowers dataset
    :return: none
    """
    print("Dataset dimensions are: ", dataset.shape)
    print("First 20 rows of data: \n", dataset.head(20))
    print("General statistics of dataset: \n", dataset.describe())
    print("Grouping by class (species): \n", dataset.groupby("species").size())


def univariate_visualizations(dataset):
    """
    Function that produces a boxplot and histogram to visualize how the variables in the Iris Flowers
    dataset are related to each other

    :param dataset: A Pandas DataFrame object of the Iris Flowers dataset
    :return: none
    """
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()
    dataset.hist()
    pyplot.show()


def multivariate_visualizations(dataset):
    """
    Function that produces a scatter matrix to visualize the multivariate relationship between
    different variables in the Iris Flowers dataset

    :param dataset: A Pandas DataFrame object of the Iris Flowers dataset
    :return: none
    """
    scatter_matrix(dataset)
    pyplot.show()


def model(dataset):
    """
    Function that creates a 20-80 test-train split across the data's variables (sepal length, sepal width
    petal length, petal width) and trains and tests over a 10 fold stratified cross validation for 6
    different models, to evaluate what the best model is. The models tested are Logistic Regression (LR),
    Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Classification and Regression Trees
    (CART), Gaussian Naive Bayes (NB), and Support Vector Machine (SVM). SVM appeared to have the best
    accuracy.

    :param dataset: A Pandas DataFrame object of the Iris Flowers dataset
    :return: a 6 length array of tuples containing the results of each model's run and the name of
    the model. The results are in the format of an ndarray, while the name is a string
    """
    data = dataset.values
    x = data[:, 0:4]
    y = data[:, 4]
    models = []
    out = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=1)
    models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
    models.append(("LDA", LinearDiscriminantAnalysis()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier()))
    models.append(("NB", GaussianNB()))
    models.append(("SVM", SVC(gamma="auto")))
    for name, algorithm in models:
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        cv_results = cross_val_score(algorithm, X=x_train, y=y_train, cv=kfold, scoring="accuracy")
        print("Accuracy for {} is {:.2f}% with standard deviation {:.2f}%\n".format(
            name, cv_results.mean().item() * 100, cv_results.std() * 100))
        out.append((cv_results, name))
    return out


def model_visualization(models):
    """
    Function that produces a visualization of each model's accuracy as a boxplot, to visually evaluate
    the most effective model

    :param models: an array of tuples where the tuples contain the results of each model's run and the
    models' names
    :return: none
    """
    model_results = []
    names = []
    for result, name in models:
        model_results.append(result)
        names.append(name)
    pyplot.boxplot(x=model_results, labels=names)
    pyplot.title("Algorithms for Iris Flowers")
    pyplot.show()


if __name__ == '__main__':
    iris_data = read_data()
    observe_data(iris_data)
    univariate_visualizations(iris_data)
    multivariate_visualizations(iris_data)
    results = model(iris_data)
    model_visualization(results)


