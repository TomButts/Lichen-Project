from sklearn.neural_network import MLPClassifier

def mlp(training_data, training_targets, options):

    mlp = MLPClassifier(hidden_layer_sizes=options['hidden_layer_sizes'], max_iter=options['max_iter'])

    print("Fitting MLP Classifier:\n")
    mlp.fit(training_data, training_targets)

    return mlp
