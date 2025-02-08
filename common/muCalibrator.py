import pickle
import numpy as np
from scipy.interpolate import RBFInterpolator

class muCalibrator:
    def __init__(self, calibration_file):
        self.interpolator = self.getInterpolator(calibration_file)

    def getInterpolator(self, filename):
        with open(filename, 'rb') as file:
            calibrationFile = pickle.load(file)
        calibrationValues = calibrationFile["calibration"]
        mu = calibrationFile["mu"]
        jes = calibrationFile["nu_jes"]
        tes = calibrationFile["nu_tes"]
        met = calibrationFile["nu_met"]
        calibrationParameterPoints = np.column_stack((mu, jes, tes, met))
        return RBFInterpolator(calibrationParameterPoints, calibrationValues)

    def getMu(self, mu, nu_jes, nu_tes, nu_met):
        point = np.array([[mu, nu_jes, nu_tes, nu_met]])
        muCorrection = self.interpolator(point)[0]
        return mu+muCorrection

    def getCorrection(self, mu, nu_jes, nu_tes, nu_met):
        point = np.array([[mu, nu_jes, nu_tes, nu_met]])
        muCorrection = self.interpolator(point)[0]
        return muCorrection
