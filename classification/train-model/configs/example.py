"""Example configuration file

"""

# mlp = dict(
#     hidden_layer_sizes=(20, 60, 30),
#     max_iter=2000,
# )

options = dict(
    svc = dict(
        probability=False,
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
    scaling = 'MaxAbsScaler'
)

# maybe only hardcoded extra info goes here and leave
# calculable things to the ouput file to print as text
# Unused information
# total_images
# categories
# transformations
# minimum image per cat
