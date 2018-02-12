# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python


from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.svm import SVC

from models.helper import SklearnHelper
import numpy as np

import xgboost as xgb

from models.util import *

train = get_processed_train_set()
test = get_processed_test_set()

ntrain = train.shape[0]
ntest = test.shape[0]
NFOLDS = 5  # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds=NFOLDS)


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 5000,
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 5000,
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 50,
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 5000,
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
}

rf = SklearnHelper(clf=RandomForestClassifier, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, params=gb_params)
svc = SklearnHelper(clf=SVC, params=svc_params)
knn3 = SklearnHelper(clf=KNeighborsClassifier, params={'n_neighbors': 3})
knn5 = SklearnHelper(clf=KNeighborsClassifier, params={'n_neighbors': 5})
knn8 = SklearnHelper(clf=KNeighborsClassifier, params={'n_neighbors': 8})
knn13 = SklearnHelper(clf=KNeighborsClassifier, params={'n_neighbors': 13})

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values  # Creates an array of the train data
x_test = test.values  # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)  # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)  # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)  # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier
knn3_oof_train, knn3_oof_test = get_oof(knn3, x_train, y_train, x_test)  # KNN3
knn5_oof_train, knn5_oof_test = get_oof(knn5, x_train, y_train, x_test)  # KNN5
knn8_oof_train, knn8_oof_test = get_oof(knn8, x_train, y_train, x_test)  # KNN8
knn13_oof_train, knn13_oof_test = get_oof(knn13, x_train, y_train, x_test)  # KNN13

print("Training is complete")

x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train, knn3_oof_train,
                          knn5_oof_train, knn8_oof_train, knn13_oof_train),
                         axis=1)
x_test = np.concatenate(
    (et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test, knn3_oof_test, knn5_oof_test,
     knn8_oof_test, knn13_oof_test), axis=1)

gbm = xgb.XGBClassifier(n_estimators=25000, nthread=-1).fit(x_train, y_train)

predictions = gbm.predict(x_test)
# Set 'Survived' column
test['Survived'] = predictions

# Creates output DataFrame
test = test[['PassengerId', 'Survived']]

test[['PassengerId', 'Survived']].to_csv('ensemble-predictions.csv', index=False)

train_predictions = gbm.predict(x_train)
survival = pd.Series(y_train == train_predictions)
print(survival.mean())
