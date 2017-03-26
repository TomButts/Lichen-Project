"""Example configuration file



"""

# mlp = dict(
#     hidden_layer_sizes=(20, 60, 30),
#     max_iter=2000,
# )
#
# mlp = True

mlp=False
mlp_scaling=False



## Example model
svc = dict(
    probability=False,
    kernel="rbf",
    C=2.8,
    gamma=.0073
)

svc_scaling=True

# svc=False
# svc_scaling=False

# Feature selection

# percentage of features to eliminate (0-1)
variance_threshold = .8
# variance_threshold = False

# retain most valuable percentage of features
feature_percentile = dict(
    mode='f_classif',
    percentage=10,
)

# feature_percentile = False

# maybe only hardcoded extra info goes here and leave
# calculable things to the ouput file to print as text
# Unused information
# total_images
# categories
# transformations
# minimum image per cat
