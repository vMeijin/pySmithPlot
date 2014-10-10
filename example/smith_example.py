#!/usr/bin/env python

from __future__ import division, unicode_literals

import sys
from multiprocessing import pool
sys.path.append("..")

import matplotlib
#matplotlib.use("GTKCairo")

import matplotlib.pyplot as pp
import numpy as np
import os
import shutil
import smithplot
import time
from smithplot.smithaxes import update_scParams
from matplotlib.transforms import Affine2D
from multiprocessing.pool import Pool
from types import FunctionType
from utils import parseCSV

# sample data
data = parseCSV("data/s11", startRow=1, steps=10)
s11 = data[:, 1] + data[:, 2] * 1j

data = parseCSV("data/s22", startRow=1, steps=10)
s22 = data[:, 1] + data[:, 2] * 1j

line = np.array([0.4 + 0.7j, 0.4 + 1.8j, 2 + 1j, 2])

def plot_example(ss=True, poly=True, circ=True, rescale=1, **kwargs):
    if ss:
        pp.plot(rescale * s11, rescale * s22, markevery=5, **kwargs)
    if poly:
        if not "path_interpolation" in kwargs:
            kwargs["path_interpolation"] = 0
        pp.plot(rescale * line, **kwargs)
    if circ:
        pp.gca().plot_vswr_circle(0.3 - 1j, real=1, solution2=True, direction="ccw", **kwargs)

# default params
update_scParams({"init.updaterc": True,
                "plot.hacklines": False,
                "plot.rotatemarker": False,
                "grid.major.fancy": False,
                "grid.minor.fancy": False,
                "grid.minor.xauto": 3,
                "grid.minor.yauto": 3,
                "axes.norm": None,
                "axes.ylabel.correction": (-2, 0)})

FT = [False, True]

build_all = True
build_path = "./build/"
        

def make_grids_on():
    fig = pp.figure(figsize=(24, 16))
    fig.set_tight_layout(True)

    i = 0
    for major_fancy in FT:
        for minor in FT:
            for minor_fancy in FT:
                if minor or not minor_fancy:
                    i += 1
                    pp.subplot(2, 3, i, projection="smith",
                                grid_major_fancy=major_fancy,
                                grid_minor_enable=minor,
                                grid_minor_fancy=minor_fancy)

                    plot_example()
                    major_str = "fancy" if major_fancy else "standard"
                    minor_str = "off" if not minor else "fancy" if minor_fancy else "standard"

                    pp.title("Major: %s - Minor: %s" % (major_str, minor_str))

    pp.savefig(build_path + "ex_grid.pdf", format="pdf")

def make_fancy_grids():
    fig = pp.figure(figsize=(24, 16))
    fig.set_tight_layout(True)

    i = 0
    for threshold in [(50, 50), (100, 50), (125, 100)]:
        i += 1
        pp.subplot(2, 3, i, projection="smith",
                   grid_major_fancy=True,
                   grid_minor_enable=False,
                   grid_major_fancy_threshold=threshold)
        plot_example()
        pp.title("Major Threshold=(%d, %d)" % threshold)


    pp.gca().scParams["grid.major.edgecolor"] = "b"

    for threshold in [15, 30, 60]:
        i += 1
        pp.subplot(2, 3, i, projection="smith",
                   grid_major_fancy=True,
                   grid_minor_fancy=True,
                   grid_minor_fancy_threshold=threshold)
        plot_example()
        pp.title("Minor Threshold=%d" % threshold)

    pp.savefig(build_path + "ex_fancy_threshold.pdf", format="pdf")

def make_grid_locators():
    fig = pp.figure(figsize=(24, 16))
    fig.set_tight_layout(True)

    i = 0
    for num in [5, 8, 14, 20]:
        i += 1
        pp.subplot(2, 4, i, projection="smith",
                   grid_major_xmaxn=num,
                   grid_major_fancy=True,
                   grid_minor_fancy=True)
        plot_example()
        pp.title("Max real steps: %d" % num)

    for num in [6, 14, 25, 50]:
        i += 1
        pp.subplot(2, 4, i, projection="smith",
                   grid_major_ymaxn=num,
                   grid_major_fancy=True,
                   grid_minor_fancy=True)
        plot_example()
        pp.title("Max imaginary steps: %d" % num)

    pp.savefig(build_path + "ex_grid_maxn.pdf", format="pdf")

def make_scale():
    fig = pp.figure(figsize=(24, 16))
    fig.set_tight_layout(True)

    i = 0
    for precision in [2, 3]:
        for scale in [1, 50, 200]:
            i += 1
            norm = None if scale != 1 else 50
            pp.subplot(2, 3, i, projection="smith",
                       grid_locator_precision=precision,
                       axes_scale=scale,
                       axes_norm=norm)
            if scale > 1:
                plot_example(circ=False, rescale=50)
            else:
                plot_example()
            norm_str = "50" if norm is not None else "off"
            pp.title("Scale: %d Ohm - Precision: %d - Norm: %s" % (scale, precision, norm_str))

    pp.savefig(build_path + "ex_scale.pdf", format="pdf")

def make_markers():
    VStartMarker = np.array([[0, 0], [0.5, 0.5], [0, -0.5], [-0.5, 0.5], [0, 0]])
    XEndMarker = np.array([[0, 0], [0.5, 0.5], [0.25, 0], [0.5, -0.5], [0, 0], [-0.5, -0.5], [-0.25, 0], [-0.5, 0.5], [0, 0]])

    fig = pp.figure(figsize=(24, 16))
    fig.set_tight_layout(True)

    i = 0
    for hackline, startmarker, endmarker, rotate_marker in [[False, None, None, False],
                                                            [True, "s", "^", False],
                                                            [True, "s", None , False],
                                                            [True, VStartMarker, XEndMarker, False],
                                                            [True, "s", "^", True],
                                                            [True, None, "^", False]]:
        i += 1
        pp.subplot(2, 3, i, projection="smith",
                   plot_hacklines=hackline,
                   plot_rotatemarker=rotate_marker)
        pp.gca().update_scParams(plot_startmarker=startmarker,
                                 plot_endmarker=endmarker,)
        plot_example(markersize=15)
        f = lambda x: "custom" if isinstance(x, np.ndarray) \
                        else "on" if x is True \
                        else "off" if x is False \
                        else 'None' if x is None else "'%s'" % x
        pp.title("HackLines: %s - StartMarker: %s\nEndMarker: %s - Rotate: %s" % tuple(map(f, [hackline, startmarker, endmarker, rotate_marker])))

    pp.savefig(build_path + "ex_marker.pdf", format="pdf")

def make_circle():
    fig = pp.figure(figsize=(24, 32))
    fig.set_tight_layout(True)

    i = 0
    for sol in FT:
        for d in ['clockwise', 'counterclockwise']:
            for x, y, rot in [[1, None, None],
                              [None, 0, None],
                              [None, None, 0.125]]:
                i += 1
                pp.subplot(4, 3, i, projection="smith",
                           plot_hacklines=True)

                for point in [0.5 + 0.5j, 0.7 + 0.5j, 0.7 - 1.2j, 0.2 - 0.5j]:
                    pp.gca().plot_vswr_circle(point,
                                              real=x,
                                              imag=y,
                                              lambda_rotation=rot,
                                              solution2=sol,
                                              direction=d)
                    f = lambda a, b, c: 'real=%d' % a if a is not None \
                                        else 'imag=%d' % b if b is not None \
                                        else 'lambda=%.3f' % c
                    g = lambda x: 'on' if x else 'off'
                    pp.title("Destination: %s - Solution 2: %s\ndirection: %s" % (f(x, y, rot), g(sol), d))

    pp.savefig(build_path + "ex_circle.pdf", format="pdf")

def make_interpolation():
    fig = pp.figure(figsize=(16, 16))
    fig.set_tight_layout(True)

    i = 0
    for interp in [1, 2, 5, 0]:
        i += 1
        pp.subplot(2, 2, i, projection="smith",
                   plot_hacklines=True)
        plot_example(ss=False, circ=False, path_interpolation=interp)
        pp.title("Path interpolation: %d" % interp)

    pp.savefig(build_path + "ex_interp.pdf", format="pdf")

#TODO: find bug when drawing minor_fancy_dividers
#def make_misc():
    #fig = pp.figure(figsize=(16, 16))
    #fig.set_tight_layout(True)

    #pp.subplot(2, 2, 1, projection="smith", plot_hacklines=True)
    #plot_example()
    #pp.legend(["S11", "S22", "Polyline", "Z \u2192 0.125l/\u03BB"])
    #pp.title("Legend")

    #divs = [2, 5]
    #pp.subplot(2, 2, 2, projection="smith",
               #grid_minor_fancy=True,
               #grid_minor_fancy_dividers=divs)
    #plot_example()
    #pp.title("Minor fancy dividers=%s" % divs)

    #pp.subplot(2, 2, 3, projection="smith", axes_radius=0.25)
    #plot_example()
    #pp.title("Axes radius: 0.25")

    #pp.subplot(2, 2, 4, projection="smith",
               #symbol_infinity="inf",
               #symbol_infinity_correction=0)
    #plot_example()
    #pp.title("Infinity symbol: 'inf'")

    ##pp.savefig(build_path + "ex_misc.pdf", format="pdf")



if __name__ == '__main__':
    if build_all:
    # clear and create path
        if os.path.exists(build_path):
            shutil.rmtree(build_path)
            time.sleep(0.5)
        os.makedirs(build_path)

        p = Pool(pool.cpu_count())
        r = []
        for key, func in locals().copy().iteritems():
            if isinstance(func, FunctionType) and "make_" in key:
                r += [p.apply_async(func, {})]

        for proc in r:
            proc.get()
    else:
#         make_grids_on()
#         make_fancy_grids()
#         make_grid_locators()
#         make_scale()
#         make_markers()
#         make_circle()
#         make_interpolation()
	  make_misc()
#         pp.show()

print "finish"
