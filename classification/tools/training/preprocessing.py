from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def scale(data, scaling_type):
    """Scale the training data

    Apply appropriate data scaling according to config options
    """
    if scaling_type == 'StandardScaler':
        scaler = StandardScaler()

        scaler.fit(data)

        data = scaler.transform(data)

    if scaling_type == 'MaxAbsScaler':
        max_abs_scaler = MaxAbsScaler()

        data = max_abs_scaler.fit_transform(data)

    # TODO: min max

    return data


def prepare(data, targets):
    """Prepare the data

    shuffle data then split into training and testing batches
    """
    data, targets = shuffle(data, targets, random_state=0)

    return train_test_split(data, targets)
