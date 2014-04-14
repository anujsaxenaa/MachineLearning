__author__ = 'Anuj'

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV


def csv_read(filename, label_name, delimiter):
    """
    Reads in the following information:
    > File name for the data
    > Label name
    > Delimiter
    Performs the following:
    > reads in a csv file containing the dataset using pandas
    > using the label, partitions the data into a feature set (x) and the target label (y)
    > returns x & y
    """
    y = pd.read_csv(filename, sep=delimiter)
    X = y.drop(label_name, 1)
    y = y[[label_name]]
    return X, y[label_name]


def splitting(X, y, testProp):
    """
    Reads in the following information:
    > Feature set => x
    > Target labels => y
    > Proportion of the test set from the data (between 0-1)
    Performs the following:
    > implements stratified sampling (shuffled) to split the data into training and test sets
    > returns the splitted data
    """
    train_indeces = []
    test_indeces = []
    sss = cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=testProp)
    for train_index, test_index in sss:
        train_indeces.append(train_index)
        test_indeces.append(test_index)
    X_train, X_test = X.ix[train_indeces[0]], X.ix[test_indeces[0]]
    y_train, y_test = y.ix[train_indeces[0]], y.ix[test_indeces[0]]
    return (X_train, y_train), (X_test, y_test)


def grid_search(trainingset, testset):
    X_train = trainingset[0]
    y_train = trainingset[1]
    X_test = testset[0]
    y_test = testset[1]
    pipeline1 = Pipeline((
        ('clf', RandomForestClassifier()),
    ))

    pipeline2 = Pipeline((

        ('clf', KNeighborsClassifier()),
    ))

    pipeline3 = Pipeline((
        ('clf', SVC()),
    ))

    pipeline4 = Pipeline((
        ('clf', MultinomialNB()),
    ))

    parameters1 = {
        'clf__n_estimators': np.arange(800, 1000, 100),
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [10, 5],
        'clf__max_features': ['log2', None],
    }
    parameters2 = {
        'clf__n_neighbors': [3, 7, 10],
        'clf__weights': ['uniform', 'distance'],
    }
    parameters3 = {
        'clf__C': [0.01, 0.1, 1.0],
        'clf__kernel': ['rbf'],
        'clf__gamma': [0.01, 0.1, 1.0],
    }
    parameters4 = {
        'clf__alpha': [0.01, 0.1, 1.0],
    }
    pars = [parameters1, parameters2, parameters3, parameters4]
    pips = [pipeline1, pipeline2, pipeline3, pipeline4]
    scores = {}
    for i in range(len(pars)):
        #cv = cross_validation.StratifiedKFold(y=y_train, n_folds=3)
        print "starting Gridsearch"
        gs = GridSearchCV(pips[i], pars[i], verbose=2, refit=True, n_jobs=-1)
        gs = gs.fit(X_train, y_train)
        gs_pred = gs.score(X_test, y_test)
        print gs_pred
        print "done gridearch"
        print "SCORE", gs.best_score_
        print "Best Params", gs.best_params_
        scores[gs] = (gs_pred, gs.best_params_)
    return "Scores", scores


def main():
    reading_csv = csv_read('traintitanic.csv', 'Survived', ',')
    split = splitting(reading_csv[0], reading_csv[1], 0.15)
    griding = grid_search(split[0], split[1])
    print griding
main()
