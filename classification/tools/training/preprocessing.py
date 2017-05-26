from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def scale(data, scaling_type):
    """Scale the training data

    Apply appropriate data scaling according to config options

    Args:
        data: feature data to scaler
        scaling_type: which scaler to use
    Returns:
        data: the scaled data
        scaler: optinally returns a fitted standard scaler object
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
