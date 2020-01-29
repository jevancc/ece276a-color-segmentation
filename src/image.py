import cv2
import numpy as np


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
