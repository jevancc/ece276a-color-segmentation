import abc
import pickle
import cv2
import numpy as np
import detector


class Classifier(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def predict(self, X):
        return NotImplemented


class MLClassifier(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, X):
        return NotImplemented

    @abc.abstractmethod
    def predict(self, X):
        return NotImplemented

    def save(self, path):
        with open(filename, 'w') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))


class SimpleHSVRedClassifier(Classifier):

    def __init__(self, lower_red=170, upper_red=10):
        self._lower_red = lower_red
        self._upper_red = upper_red

    def predict(self, X):
        mask = cv2.inRange(X, np.array([self._lower_red, 30, 30]), np.array([180, 255, 255])) \
            + cv2.inRange(X, np.array([0, 30, 30]), np.array([self._upper_red, 255, 255]))
        return mask


class SSBBoxDeterministic(Classifier):

    def __init__(self, image_area):
        self._image_area = image_area

    def predict(self, region):
        return self._predict_region(region)

    def _predict_region(self, region):
        if region.area >= 100:

            minr, minc, maxr, maxc = region.bbox
            lr = maxr - minr
            lc = maxc - minc

            if max(lc, lr) - min(lc, lr) > 1.2 * min(lc, lr):
                return False

            for contour in detector.Contour.find(region.padded_image(10)):
                if contour.area > self._image_area * 0.001 and contour.area > region.bbox_area * 0.5:
                    if not (region.image[region.image != 0].sum() >
                            contour.area * 0.6):
                        return False

                    approx_polygon = contour.approx_polygon()
                    n_edges = approx_polygon.n_edges

                    if 7 <= n_edges < 16:
                        degs = approx_polygon.angles_deg * 2
                        continuous_degs_sums = [
                            sum(degs[i:i + 4]) % 360 for i in range(n_edges)
                        ]
                        n_valid_partial_octagon = sum([
                            160 < v < 200 or 160 < (v + 180) % 360 < 200
                            for v in continuous_degs_sums
                        ])
                        if (n_valid_partial_octagon >= 2 and n_edges <= 12) \
                            or (n_valid_partial_octagon >= 3 and n_edges <= 16):
                            return True
        return False
