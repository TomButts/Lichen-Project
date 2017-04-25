options = dict(
    svc = dict(
        tuned_parameters = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'probability': [True]},
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'probability': [True]}
        ],
        scoring_strategies = ['f1_macro', 'accuracy', 'neg_log_loss'],
        probability = True
    ),
    selectors = dict(
        variance_threshold = .8,
        feature_percentile = dict(
            mode='f_classif',
            percentage=10,
        )
    ),
    scaling = 'MaxAbsScaler',
    # data transform factor
    transform_factor = 2
)
