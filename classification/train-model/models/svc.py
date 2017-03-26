from sklearn.svm import SVC

def svc(training_data, training_targets, options):
    svc = SVC(probability=options['probability'], kernel=options['kernel'], C=options['C'], gamma=options['gamma'])

    svc.fit(training_data, training_targets)

    return svc
