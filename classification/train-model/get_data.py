# get data from csv
# conditional preprocessing
# conditionally apply feature selection
# train test split
# shuffle
# return data
#
# Should include pre feature selection metrics?

import sys
import os

dir = os.path.dirname(__file__)
utils_path = os.path.join(dir, '../../utils')
sys.path.append(utils_path)

import read_features_csv as features_csv
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from importlib import import_module

def get_data(features_path, options):
    """Function to fetch and prepare data

    Args:
        features_path: the path to the csv file containing features and targets
        options: the configuration file containing setup options

    Return:
        data: X_train, y_train, X_test, y_test
    """
    targets, data = features_csv.read_features_csv(features_path)

    X = data
    y = targets

    if options.variance_threshold:
        sel = VarianceThreshold(threshold=(options.variance_threshold * (1 - options.variance_threshold)))

        X = sel.fit_transform(X)

    if options.feature_percentile:
        # TODO load module on command
        # module = import_module(options.percentile['mode'])

        if options.feature_percentile['mode'] == 'f_classif':
            selector = SelectPercentile(f_classif, percentile=options.feature_percentile['percentage'])

            selector.fit(X, y)

            selector.transform(X)
            print('x')


    X_train, X_test, y_train, y_test = train_test_split(X, y)

    if options.mlp_scaling:
        scaler = StandardScaler()

        # fit only to the training data
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if options.svc_scaling:
        max_abs_scaler = MaxAbsScaler()

        X_train = max_abs_scaler.fit_transform(X_train)
        X_test = max_abs_scaler.transform(X_test)


    return X_train, X_test, y_train, y_test
