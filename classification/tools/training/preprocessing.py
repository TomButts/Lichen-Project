"""
This function handles logic for scaling feature sets during training.

"""

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def scale(data, scaling_type):
    """Scale the training data

    Apply appropriate data scaling according to config options
    """

    scaler = None

    if scaling_type == 'StandardScaler':
        scaler = StandardScaler()

        scaler.fit(data)

        data = scaler.transform(data)

    if scaling_type == 'MaxAbsScaler':
        max_abs_scaler = MaxAbsScaler()

        data = max_abs_scaler.fit_transform(data)

    return data, scaler
