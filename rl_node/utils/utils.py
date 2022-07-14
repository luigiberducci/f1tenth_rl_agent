import numpy as np


def linear_scaling(v, from_range, to_range, clip=True):
    if clip:
        v = np.clip(v, from_range[0], from_range[1], dtype=np.float32)  # clip it
    new_v = (v - from_range[0]) / (from_range[1] - from_range[0])  # norm in 0,1
    new_v = to_range[0] + (to_range[1] - to_range[0]) * new_v  # map it to target range
    return new_v