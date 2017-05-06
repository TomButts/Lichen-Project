options = dict(
    mlp=dict(
        tuned_parameters=[
            {'solver': ['sgd'], 'alpha': [0.01, 0.1, 1, 10, 100, 1000], 'hidden_layer_sizes': [[100], [50, 50]]},
            {'solver': ['lbfgs'], 'alpha': [0.01, 0.1, 1, 10, 100, 1000], 'hidden_layer_sizes': [[100], [50, 50]]},
            {'solver': ['adam'], 'alpha': [0.01, 0.1, 1, 10, 100, 1000], 'hidden_layer_sizes': [[100], [50, 50]]},
        ],
        scoring_strategies=['f1_macro', 'accuracy', 'neg_log_loss'],
        probability=True
    ),
    selectors=dict(
        variance_threshold=.8,
        feature_percentile=dict(
            mode='f_classif',
            percentage=10,
        )
    ),
    scaling='StandardScaler',
    transform_factor=2
)
