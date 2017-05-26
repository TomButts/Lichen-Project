from sklearn.feature_selection import VarianceThreshold, SelectPercentile
import sklearn.feature_selection as selection

def fit_selectors(features, targets, options):
    """Fit the feature selectors used in training

    Args:
        features: the feature data
        targets: feature labels
        options: configuration settings dictionary
    Returns:
        variance_threshold_selector: a fitted variance threshold selector object
        percentile_selector: a fitted univariate selector object
    """
    variance_threshold_selector = None
    percentile_selector = None

    if 'variance_threshold' in options:
        variance_threshold_selector = VarianceThreshold(threshold=(
            options['variance_threshold'] * (1 - options['variance_threshold'])))

        features = variance_threshold_selector.fit_transform(features)

    if 'feature_percentile' in options:
        # load the feature fitness algorithm module
        module = getattr(selection, options['feature_percentile']['mode'])

        percentile_selector = SelectPercentile(
            module, percentile=options['feature_percentile']['percentage'])

        percentile_selector.fit(features, targets)

    return variance_threshold_selector, percentile_selector

def transform_features(features, variance_threshold_selector=None, percentile_selector=None):
    """Applies feature selector objects to data

    Args:
        features: the features to be transformed
        variance_threshold_selector: variance threshold selector object
        percentile_selector: univariate selector object
    Returns:
        features: feature selected version of feature set
    """
    if variance_threshold_selector != None:
        features = variance_threshold_selector.transform(features)

    if percentile_selector != None:
        features  = percentile_selector.transform(features)

    return features
