from sklearn.calibration import CalibratedClassifierCV

def calibrate(classifer, data, targets, options):
     calib = CalibratedClassifierCV(classifer, cv=options['cv'], method=options['method'])

     calib.fit(data, targets)

     return calib
