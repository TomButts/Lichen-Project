"""Example configuration file

"""

# mlp = dict(
#     hidden_layer_sizes=(20, 60, 30),
#     max_iter=2000,
#     probability=False,
# )
#
# svc = dict(
#     probability=True,
#     kernel="rbf",
#     C=2.8,
#     gamma=.0073
# ),

options = dict(
    mlp = dict(
        hidden_layer_sizes=(20, 60, 30),
        max_iter=2000,
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

    # data transform factor
    transform_factor = 0
)
