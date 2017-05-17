options = dict(
    svc = dict(
        tuned_parameters = [
            {'kernel': ['rbf'], 'gamma': [1e-5, 1e-4, 1e-3, 1, 2, 5, 10, 100], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'probability': [True]},
            {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000], 'probability': [True]},
        ],
        scoring_strategies = ['f1_macro', 'accuracy', 'neg_log_loss'],
        probability = True
    ),
    # selectors = dict(
    #     # variance_threshold = .1,
    #     feature_percentile = dict(
    #         mode='f_classif',
    #         percentage=30,
    #     )
    # ),
    scaling = 'MaxAbsScaler',
    transform_factor = 2
)
