# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from models.util import *
import src.data.make_dataset as mkds


def predict(df=get_processed_data(), results_column='Survived'):
    y = df[results_column]
    X = df.drop(columns=results_column)

    clf = RandomForestClassifier(n_estimators=1000)

    clf.fit(X, y)

    test = get_test_data()

    W = mkds.routine(test.copy())

    predictions = clf.predict(W)

    test['Survived'] = predictions

    test = test[['PassengerId', 'Survived']]

    # TODO decide which is the best path to save
    test.to_csv('rfc-predictions.csv', index=False)
