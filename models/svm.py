# -*- coding: utf-8 -*-
from sklearn import svm
from models.util import *
import src.data.make_dataset as mkds


def predict(df=get_processed_data(), results_column='Survived'):
    """
    Instantiates a SVM model and saves predictions to disk.

    :param df: Input DataFrame (Processed)
    :param results_column: (column name for results)
    """

    # Get Results
    y = df[results_column]

    # Get DataFrame without results
    X = df.drop(columns=results_column)

    # Instantiates RFC
    clf = svm.SVC()

    # Fits model
    clf.fit(X, y)

    # Gets test data
    test = get_test_data()

    # Process test data
    W = mkds.routine(test.copy())

    # Make predictions
    predictions = clf.predict(W)

    # Set 'Survived' column
    test['Survived'] = predictions

    # Creates output DataFrame
    test = test[['PassengerId', 'Survived']]

    # TODO decide where is the best path to save
    # Saves to csv to disk
    test.to_csv('svm-predictions.csv', index=False)
