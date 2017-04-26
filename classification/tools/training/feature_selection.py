from sklearn.feature_selection import VarianceThreshold, SelectPercentile
import sklearn.feature_selection as selection

def fit_selectors(features, targets, options):
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
    print(len(features[0]))
    if variance_threshold_selector != None:
        features = variance_threshold_selector.transform(features)

    print(len(features[0]))
    if percentile_selector != None:
        features  = percentile_selector.transform(features)

    return features
