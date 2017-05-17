options = dict(
    mlp = dict(
        tuned_parameters=[
            {'solver': ['sgd'], 'alpha': [0.001, 0.01, 1, 5, 10, 100, 1000], 'hidden_layer_sizes': [[2], [5], [10], [15], [20], [25], [40]]},
            {'solver': ['lbfgs'], 'alpha': [0.001, 0.01, 1, 5, 10, 100, 1000], 'hidden_layer_sizes': [[2], [5], [10], [15], [20], [25], [40]]},
            {'solver': ['adam'], 'alpha': [0.001, 0.01, 1, 5, 10, 100, 1000], 'hidden_layer_sizes': [[2], [5], [10], [15], [20], [25], [40]]},
        ],
        scoring_strategies=['f1_macro', 'accuracy', 'neg_log_loss'],
        probability=True
    ),
    # selectors=dict(
    #     # variance_threshold=.1,
    #     # feature_percentile=dict(
    #     #     mode='f_classif',
    #     #     percentage=5,
    #     # )
    # ),
    scaling='StandardScaler',
    transform_factor=2
)
