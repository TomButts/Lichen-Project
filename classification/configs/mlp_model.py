options = dict(
    mlp = dict(
        hidden_layer_sizes=(110,60),
        max_iter=350,
        probability=True,
    ),
    selectors = dict(
        variance_threshold = .8,
        feature_percentile = dict(
            mode='f_classif',
            percentage=10,
        )
    ),
    scaling = 'StandardScaler',
    transform_factor = 2
)
