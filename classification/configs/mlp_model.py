"""
Example configuration file

This configuration file runs model selection on a batch of MLPClassifiers
using a specified feature set.
"""

options = dict(
    mlp = dict(
        tuned_parameters=[
            {'solver': ['sgd'], 'alpha': [1, 5]},
            {'solver': ['lbfgs'], 'alpha': [1, 5]},
            {'solver': ['adam'], 'alpha': [1, 5]},
        ],
        scoring_strategies=['f1_macro', 'accuracy', 'neg_log_loss'],
        probability=True
    ),
    selectors=dict(
        # variance_threshold=.1,
        feature_percentile=dict(
            mode='f_classif',
            percentage=1,
        )
    ),
    scaling='StandardScaler',
    training_directory='/Users/tom/Masters-Project/Lichen-Images/Feature Sets/augmented-inception/training-80/training.csv',
    validation_directory='/Users/tom/Masters-Project/Lichen-Images/Feature Sets/augmented-inception/validation-20/validation.csv',
    output_directory='/Users/tom/Masters-Project/Evaluations/',
    folder_name='test_mlp',
    transform_factor=2
)
