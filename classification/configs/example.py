"""Example configuration file

"""

# mlp = dict(
#     hidden_layer_sizes=(20, 60, 30),
#     max_iter=2000,
# )

options = dict(
    svc = dict(
        probability=True,
        kernel="rbf",
        C=2.8,
        gamma=.0073
    ),
    calibration = dict(
        cv = 2,
        method = 'sigmoid'
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
    transform_factor = 0
)
