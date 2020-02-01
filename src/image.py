import cv2
import numpy as np
from enum import Enum, auto
from copy import deepcopy


def build_histogram_equalizer(channel, vmax):
    nvals = channel.size
    vcnt = {v: cnt for v, cnt in zip(*np.unique(channel, return_counts=True))}
    cumulative_sum = np.zeros((vmax + 1))
    for i in range(vmax + 1):
        cumulative_sum[i] = cumulative_sum[i - 1] + vcnt.get(i, 0)

    cumulative_sum = np.round(cumulative_sum / nvals * vmax)

    def f(val):
        return cumulative_sum[val]

    return f


class Image:
    COLOR_SPACES = ['RGB', 'BGR', 'YCrCb', 'HSV', 'GRAY']

    def load(path):
        return Image(cv2.imread(path), 'BGR')

    def __init__(self, data, colorspace='BGR'):
        self._assert_valid_colorspace(colorspace)
        self._data = data
        self._colorspace = colorspace

    def _assert_valid_colorspace(self, colorspace):
        assert colorspace in Image.COLOR_SPACES, f'{colorspace} is not a valid cv2 colorspace'

    @property
    def area(self):
        return self.nr * self.nc

    @property
    def nr(self):
        return self._data.shape[0]

    @property
    def nc(self):
        return self._data.shape[1]

    def change_colorspace(self, to):
        self._assert_valid_colorspace(to)
        if self._colorspace == to:
            return self.copy()

        data = self._data
        if f'COLOR_{self._colorspace}2{to}' in cv2.__dict__:
            data = cv2.cvtColor(data, cv2.__dict__[f'COLOR_{self._colorspace}2{to}'])
        else:
            data = cv2.cvtColor(data, cv2.__dict__[f'COLOR_{self._colorspace}2RGB'])
            data = cv2.cvtColor(data, cv2.__dict__[f'COLOR_RGB2{to}'])
        return Image(data, to)

    @property
    def hsv(self):
        return self.change_colorspace('HSV')

    @property
    def rgb(self):
        return self.change_colorspace('RGB')

    @property
    def ycrcb(self):
        return self.change_colorspace('YCrCb')

    @property
    def bgr(self):
        return self.change_colorspace('BGR')

    @property
    def gray(self):
        return self.change_colorspace('GRAY')

    def histogram_equalize(self, vmin=0, vmax=255, channel_id=None):

        if channel_id is None:
            equalizer = build_histogram_equalizer(self._data[:, :], vmax)
            newimg = equalizer(self._data[:, :]).astype(np.uint8)
        else:
            equalizer = build_histogram_equalizer(self._data[:, :, channel_id],
                                                  vmax)
            newimg = self._data.copy()
            newimg[:, :, channel_id] = equalizer(
                newimg[:, :, channel_id]).astype(np.uint8)

        return Image(newimg, self._colorspace)

    def mulclip(self, factor, vmin=0, vmax=255, channel_id=None):
        if channel_id is None:
            newimg = np.clip((self._data.astype(float) * factor),
                             a_min=vmin,
                             a_max=vmax).astype(np.uint8)
        else:
            newimg = self._data.copy()
            newimg[:, :, channel_id] = np.clip(
                (newimg[:, :, channel_id].astype(float) * factor),
                a_min=vmin,
                a_max=vmax).astype(np.uint8)

        return Image(newimg, self._colorspace)

    def addclip(self, offset, vmin=0, vmax=255, channel_id=None):
        if channel_id is None:
            newimg = np.clip((self._data.astype(float) + offset),
                             a_min=vmin,
                             a_max=vmax).astype(np.uint8)
        else:
            newimg = self._data.copy()
            newimg[:, :, channel_id] = np.clip(
                (newimg[:, :, channel_id].astype(float) + offset),
                a_min=vmin,
                a_max=vmax).astype(np.uint8)

        return Image(newimg, self._colorspace)

    def numpy(self):
        return self._data.copy()

    @property
    def data(self):
        return self.numpy()

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone

    def __deepcopy__(self):
        return Image(self._data.copy(), self._colorspace)

    def copy(self):
        return self.__deepcopy__()

    def __getattr__(self, name):
        return getattr(self._data, name)
