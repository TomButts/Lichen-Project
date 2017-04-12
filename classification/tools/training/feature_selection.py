from sklearn.feature_selection import VarianceThreshold, SelectPercentile
import sklearn.feature_selection as selection


def select_features(features, targets, options):
    if 'variance_threshold' in options:
        sel = VarianceThreshold(threshold=(
            options['variance_threshold'] * (1 - options['variance_threshold'])))

        features = sel.fit_transform(features)

    if 'feature_percentile' in options:
        module = getattr(selection, options['feature_percentile']['mode'])

        selector = SelectPercentile(
            module, percentile=options['feature_percentile']['percentage'])

        selector.fit(features, targets)

        selector.transform(features)

    return features, targets
