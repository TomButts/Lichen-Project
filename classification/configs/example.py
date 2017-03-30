"""Example configuration file

"""

# mlp = dict(
#     hidden_layer_sizes=(20, 60, 30),
#     max_iter=2000,
#     probability=False,
# )

options = dict(
    svc = dict(
        probability=True,
        kernel="rbf",
        C=2.8,
        gamma=.0073
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
