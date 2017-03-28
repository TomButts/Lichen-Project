from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def svc(training_data, training_targets, options):
    svc = SVC(probability=options['probability'], kernel=options['kernel'], C=options['C'], gamma=options['gamma'])

    print("Fitting MLP Classifier:\n")

    svc.fit(training_data, training_targets)

    return svc
