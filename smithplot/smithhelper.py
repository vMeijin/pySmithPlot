# -*- coding: utf-8 -*-
# last edit: 11.04.2018

# Changed by FG for python 3.10
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable


import numpy as np

INF = 1e9
EPSILON = 1e-7
TWO_PI = 2 * np.pi


def xy_to_z(*xy):
    if len(xy) == 1:
        z = xy[0]
        if isinstance(z, Iterable):
            z = np.array(z)
            if len(z.shape) == 2:
                z = z[0] + 1j * z[1]
            elif len(z.shape) > 2:
                raise ValueError("Something went wrong!")
    elif len(xy) == 2:
        x, y = xy
        if isinstance(x, Iterable):
            if isinstance(y, Iterable) and len(x) == len(y):
                z = np.array(x) + 1j * np.array(y)
            else:
                raise ValueError("x and y vectors dont match in type and/or size")
        else:
            z = x + 1j * y
    else:
        raise ValueError("Arguments are not valid - specify either complex number/vector z or real and imaginary number/vector x, y")

    return z


def z_to_xy(z):
    return z.real, z.imag


def moebius_z(*args, norm):
    z = xy_to_z(*args)
    return 1 - 2 * norm / (z + norm)


def moebius_inv_z(*args, norm):
    z = xy_to_z(*args)
    return norm * (1 + z) / (1 - z)


def ang_to_c(ang, radius=1):
    return radius * (np.cos(ang) + np.sin(ang) * 1j)


def lambda_to_rad(lmb):
    return lmb * 4 * np.pi


def rad_to_lambda(rad):
    return rad * 0.25 / np.pi
