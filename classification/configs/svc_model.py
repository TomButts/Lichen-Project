options = dict(
    svc = dict(
        probability=True,
        kernel="rbf",
        C=100,
        gamma='auto'
    ),
    scaling = 'MaxAbsScaler',
    transform_factor = 0
)
