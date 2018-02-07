# -*- coding: utf-8 -*-
from sklearn import svm
from models.util import *
import src.data.make_dataset as mkds


def predict(df=get_processed_data(), results_column='Survived'):
    y = df[results_column]
    X = df.drop(columns=results_column)

    clf = svm.SVC()

    clf.fit(X, y)

    test = get_test_data()

    W = mkds.routine(test.copy())

    predictions = clf.predict(W)

    test['Survived'] = predictions

    test = test[['PassengerId', 'Survived']]

    # TODO decide where is the best path to save
    test.to_csv('svm-predictions.csv', index=False)
