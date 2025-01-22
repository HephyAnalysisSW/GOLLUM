import logging
logger = logging.getLogger(__name__)

class intervalFinder:
    def __init__(self, muValues, qValues, threshold=1.0):
        self.muValues = muValues
        self.qValues = qValues
        self.threshold = threshold

    def __findCrossings(self, x, y, threshold):
        crossings = []
        for i in range(len(y) - 1):
            # find the two y values, where one is above and one below y = crossing
            if (y[i] - threshold) * (y[i+1] - threshold) < 0:
                # do linear interpolation to find the x value
                x_cross = x[i] + (threshold - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i])
                crossings.append(x_cross)
        return crossings

    def getInterval(self):
        crossings = self.__findCrossings(self.muValues, self.qValues, self.threshold)
        if len(crossings) != 2:
            logger.error("[ERROR] Found", len(crossings), "point(s) where q = 1 in the mu range, should be 2! Please increase scan range.")
        return crossings
