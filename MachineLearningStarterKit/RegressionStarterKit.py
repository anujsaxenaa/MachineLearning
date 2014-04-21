import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
import numpy as np
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
    X = y.drop([label_name], axis=1)
    y = y[[label_name]]
    return X, y[label_name]


def splitting(X, y, testProp, isItTimeSeries):
    """
    Reads in the following information:
    > Feature set => x
    > Target labels => y
    > Proportion of the test set from the data (between 0-1)
    Performs the following:
    > Checks if the data is a time series
    > If 'Yes', then it creates the holdout set from the bottom of the dataset depending on the given test proportion
    > If 'No' or otherwise, it creates a holdout set using train_test_split functionality in cross validation
    > returns the splitted data
    """
    if isItTimeSeries == 'Yes':
        X_train = X.ix[:int(len(X)*(1-testProp))]
        X_test = X.ix[int(round(len(X)*(1-testProp))):]
        y_train = y.ix[:int(len(y)*(1-testProp))]
        y_test = y.ix[int(round(len(y)*(1-testProp))):]
    else:
        tts = cross_validation.train_test_split(X, y, test_size=testProp)
        X_train, X_test = tts[0], tts[1]
        y_train, y_test = tts[2], tts[3]
    return (X_train, y_train), (X_test, y_test)


def grid_search(trainingset, testset):
    """
    Reads in the following information:
    > The training set (which is a list or tuple containing X_train & y_train
    > The test set (which is a list or tuple containing X_test & y_test)
    Performs the following:
    > Creates new train-test X-y's
    > Defines new pipelines that are just the different algorithms to be implemented
    > Defines new parameters that are the model parameters of the corresponding algorithms
    > Loops over these and gives the best possible score and model that each algorithm can give.
    > Testing other parameter values than the ones below is strongly encouraged.
    """
    X_train = trainingset[0]
    y_train = trainingset[1]
    X_test = testset[0]
    y_test = testset[1]
    pipeline1 = Pipeline((
        ('reg', RandomForestRegressor()),
    ))

    pipeline2 = Pipeline((
        ('reg', LinearRegression()),
    ))
    pipeline3 = Pipeline((
        ('reg', KNeighborsRegressor()),
    ))
    parameters1 = {
        'reg__n_estimators': np.arange(500, 1000, 100),
        'reg__criterion': ['mse'],
        'reg__max_depth': [10, 5],
        'reg__max_features': ['log2', None],
    }
    parameters2 = {
        'reg__fit_intercept': [True, False],
    }

    parameters3 = {
        'reg__n_neighbors': np.arange(3, 15, 2),
        'reg__weights': ['uniform', 'distance'],
    }

    pips = [pipeline1, pipeline2, pipeline3]
    pars = [parameters1, parameters2, parameters3]
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
    """
    This is a wrapper function that runs all the above functions
    """
    reading_csv = csv_read('BodyFatPercentage.csv', 'BODYFAT', ',')
    split = splitting(reading_csv[0], reading_csv[1], 0.2, 'No')
    griding = grid_search(split[0], split[1])
    print griding

main()
