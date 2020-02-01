import math
import cv2
import numpy as np
from functools import wraps
from skimage.measure import label, regionprops, approximate_polygon
from skimage.color import label2rgb


def _cached(f):

    @wraps(f)
    def wrapper(obj):
        if not hasattr(obj, '_cache'):
            obj._cache = {}

        cache = obj._cache
        prop = f.__name__

        if not ((prop in cache) and obj._cache_active):
            cache[prop] = f(obj)

        return cache[prop]

    return wrapper


def _pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


class Region:

    def find(image, original_image=None):
        image_label = label(image)
        return Region.find_label(image_label,
                                 image=image,
                                 original_image=original_image)

    def find_label(label, image=None, original_image=None):
        regions = regionprops(label)
        return [Region(r, original_image) for r in regions]

    def __init__(self, data, original_image=None):
        self._data = data
        self._original_image = original_image

    def padded_image(self, pad_width=10):
        return np.pad(self.image, pad_width, _pad_with, padder=0)

    @property
    @_cached
    def original_image(self):
        minr, minc, maxr, maxc = self.bbox
        return self._original_image[minr:maxr, minc:maxc, :]

    def padded_original_image(self, pad_width=10):
        minr, minc, maxr, maxc = self.bbox
        minr = max(minr - pad_width, 0)
        minc = max(minc - pad_width, 0)
        maxr = min(maxr + pad_width, self._original_image.shape[0])
        maxc = min(maxc + pad_width, self._original_image.shape[1])
        return self._original_image[minr:maxr, minc:maxc, :]

    @property
    @_cached
    def pyplot_coordinates(self):
        minr, minc, maxr, maxc = self.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        return (bx, by)

    def __getattr__(self, name):
        return getattr(self._data, name)

    def __dir__(self):
        return dir(self._data) + ['find', 'find_labeled', 'padded_imag']

class Contour:

    def find(image):
        contours = cv2.findContours(image.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        return [Contour(c) for c in contours]

    def __init__(self, contour, cache_active=True):
        self._data = contour
        self._ploygon = None
        self._cache_active = cache_active

    @property
    @_cached
    def area(self):
        return cv2.contourArea(self._data)

    @property
    @_cached
    def perimeter(self):
        return cv2.arcLength(self._data, True)

    @property
    @_cached
    def is_convex(self):
        return cv2.isContourConvex(self._data)

    @_cached
    def approx_polygon(self, epsilon=None):
        if epsilon is None:
            epsilon = 0.02 * self.perimeter

        approx_curve = cv2.approxPolyDP(self._data, epsilon, True)
        return Ploygon(approx_curve)

    @_cached
    def approx_polygon_skimage_raw(self, tolerance=0.01):
        contour = self._data[:, 0, :]
        appr = approximate_polygon(contour, tolerance=tolerance)
        return appr

    @property
    def numpy(self):
        return self._data.copy()

    @property
    @_cached
    def pyplot_coordinates(self):
        coordinates = self.numpy[:, 0, :]
        x, y = coordinates[:, 0], coordinates[:, 1]
        return (x, y)


def _angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype(float), (p2 - p1).astype(float)
    return np.dot(d1, d2.T) / np.sqrt(np.dot(d1, d1.T) * np.dot(d2, d2.T))


class Ploygon:

    def __init__(self, curve, cache_active=True):
        self._curve = curve
        self._cache_active = cache_active

    @property
    @_cached
    def angles_cos(self):
        L = self.n_edges
        curve = self._curve
        return [
            _angle_cos(curve[i], curve[(i + 1) % L], curve[(i + 2) % L])
            for i in range(L)
        ]

    @property
    @_cached
    def angles_rad(self):
        return [math.acos(val) for val in self.angles_cos]

    @property
    @_cached
    def angles_deg(self):
        return [val / math.pi * 360.0 for val in self.angles_rad]

    @property
    def n_edges(self):
        return len(self._curve)

    @property
    @_cached
    def shape_name(self):
        # initialize the shape name and approximate the contour
        curve = self._curve

        if len(curve) == 3:
            return 'triangle'
        elif len(curve) == 4:
            (x, y, w, h) = cv2.boundingRect(curve)
            ar = w / float(h)
            return 'square' if ar >= 0.95 and ar <= 1.05 else 'rectangle'
        elif len(curve) == 5:
            return 'pentagon'
        elif len(curve) == 6:
            return 'hexagon'
        elif len(curve) == 7:
            return 'heptagon'
        elif len(curve) == 8:
            return 'octagon'
        elif len(curve) == 9:
            return 'decagon'
        else:
            return 'circle'

    @property
    def numpy(self):
        return self._curve.copy()

    @property
    @_cached
    def pyplot_coordinates(self):
        coordinates = self.numpy[:, 0, :]
        x, y = coordinates[:, 0], coordinates[:, 1]
        return (x, y)
