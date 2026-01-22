#!/usr/bin/env python3

import os
import shutil
import sys
import time
from multiprocessing.pool import Pool
from types import FunctionType

import numpy as np
from matplotlib import rcParams, pyplot as pp


def get_app_path() -> str:
    """Gets Appliction path

    Returns:
        str: Application path
    """
    # determine if application is a script file or frozen exe
    application_path = ""
    if getattr(sys, "frozen", False):
        application_path = os.path.dirname(sys.executable)
    elif __file__:
        application_path = os.path.dirname(__file__)
    return application_path

sys.path.append("..")
from smithplot.smithaxes import SmithAxes
from smithplot import smithhelper

rcParams.update({"legend.numpoints": 3,
                 "axes.axisbelow": True})

# sample data
steps = 40
data = np.loadtxt(os.path.join(get_app_path(),"data/s11.csv"), delimiter=",", skiprows=1)[::steps]
sp_data = data[:, 1] + data[:, 2] * 1j

data = np.loadtxt(os.path.join(get_app_path(),"data/s22.csv"), delimiter=",", skiprows=1)[::steps]
z_data = 50 * (data[:, 1] + data[:, 2] * 1j)

# default params
SmithAxes.update_scParams({"plot.marker.hack": False,
                           "plot.marker.rotate": False,
                           "grid.minor.enable": False,
                           "grid.minor.fancy": False})

FT = [False, True]
figsize = 6
ExportFormats = ["pdf", "png"]


def plot_example(testbench, title, scale=50, **kwargs):
    print("Testbench '%s' : %s" % (testbench, title.replace("\n", "")))
    kwargs.setdefault("markevery", 1)
    pp.plot(smithhelper.moebius_inv_z(sp_data, norm=50), datatype="Z", **kwargs)
    pp.plot(z_data, datatype="Z", **kwargs)
    pp.plot(100, datatype="Z", **kwargs)
    pp.plot(25 + 25j, datatype="Z", **kwargs)
    pp.title(title)


def savefig(testbench):
    for ext in ExportFormats:
        pp.savefig("%s/sample_%s.%s" % (build_path, testbench.lower().replace(" ", "_"), ext), format=ext)


def tb_grid_styles():
    tb = "Grid Styles"
    fig = pp.figure(figsize=(3 * figsize, 2 * figsize))
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

                    major_str = "fancy" if major_fancy else "standard"
                    minor_str = "off" if not minor else "fancy" if minor_fancy else "standard"

                    plot_example(tb, "Major: %s - Minor: %s" % (major_str, minor_str))

    savefig(tb)


def tb_fancy_grids():
    tb = "Fancy Grid"
    fig = pp.figure(figsize=(3 * figsize, 2 * figsize))
    fig.set_tight_layout(True)

    i = 0
    for threshold in [(50, 50), (100, 50), (125, 100)]:
        i += 1
        pp.subplot(2, 3, i, projection="smith",
                   grid_major_fancy_threshold=threshold)
        plot_example(tb, "Major Threshold=(%d, %d)" % threshold)

    for threshold in [15, 30, 60]:
        i += 1
        pp.subplot(2, 3, i, projection="smith",
                   grid_minor_fancy=True,
                   grid_minor_enable=True,
                   grid_minor_fancy_threshold=threshold)
        plot_example(tb, "Minor Threshold=%d" % threshold)

    savefig(tb)


def tb_grid_locators():
    tb = "Grid Locators"
    fig = pp.figure(figsize=(4 * figsize, 2 * figsize))
    fig.set_tight_layout(True)

    i = 0
    for num in [5, 8, 14, 20]:
        i += 1
        pp.subplot(2, 4, i, projection="smith",
                   grid_major_xmaxn=num)
        plot_example(tb, "Max real steps: %d" % num)

    for num in [6, 14, 25, 50]:
        i += 1
        pp.subplot(2, 4, i, projection="smith",
                   grid_major_ymaxn=num)
        plot_example(tb, "Max imaginary steps: %d" % num)

    savefig(tb)


def tb_normalize():
    tb = "Normalize"
    fig = pp.figure(figsize=(3 * figsize, 2 * figsize))
    fig.set_tight_layout(True)

    i = 0
    for normalize in FT:
        for impedance in [10, 50, 200]:
            i += 1
            pp.subplot(2, 3, i, projection="smith",
                       axes_impedance=impedance,
                       axes_normalize=normalize)
            plot_example(tb, "Impedance: %d Ω — Normalize: %s" % (impedance, normalize))

    savefig(tb)


def tb_markers():
    tb = "Markers"
    VStartMarker = np.array([[0, 0], [0.5, 0.5], [0, -0.5], [-0.5, 0.5], [0, 0]])
    XEndMarker = np.array([[0, 0], [0.5, 0.5], [0.25, 0], [0.5, -0.5], [0, 0], [-0.5, -0.5], [-0.25, 0], [-0.5, 0.5], [0, 0]])

    fig = pp.figure(figsize=(4 * figsize, 2 * figsize))
    fig.set_tight_layout(True)

    i = 0
    for hackline, startmarker, endmarker, rotate_marker in [[False, None, None, False],
                                                            [True, "s", "^", False],
                                                            [True, "s", 'None', False],
                                                            [True, VStartMarker, XEndMarker, False],
                                                            [True, "s", "^", True],
                                                            [True, 'None', "^", False]]:
        i += 1
        ax = pp.subplot(2, 3, i, projection="smith",
                        plot_marker_hack=hackline,
                        plot_marker_rotate=rotate_marker)
        SmithAxes.update_scParams(instance=ax, plot_marker_start=startmarker,
                                  plot_marker_end=endmarker)

        def ptype(x):
            if isinstance(x, np.ndarray):
                return "custom"
            elif x is True:
                return "on"
            elif x is False:
                return "off"
            elif x is None:
                return None
            else:
                return "'%s'" % x

        plot_example(tb, "HackLines: %s - StartMarker: %s\nEndMarker: %s - Rotate: %s" % tuple(map(ptype, [hackline, startmarker, endmarker, rotate_marker])), markersize=10)

    savefig(tb)


def tb_interpolation():
    tb = "Interpolation"
    fig = pp.figure(figsize=(3 * figsize, 2 * figsize))
    fig.set_tight_layout(True)

    i = 0
    for interpolation, equipoints in [[False, False],
                                      [10, False],
                                      [False, 10],
                                      [False, 50]]:
        i += 1
        pp.subplot(2, 2, i, projection="smith")
        plot_example(tb, "Interpolation: %s — Equipoints: %s" % ("False" if interpolation is False else interpolation,
                                                                 "False" if equipoints is False else equipoints), interpolate=interpolation, equipoints=equipoints)

    savefig(tb)


def tb_misc():
    tb = "Miscellaneous"
    fig = pp.figure(figsize=(3 * figsize, 2 * figsize))
    fig.set_tight_layout(True)

    pp.subplot(2, 3, 1, projection="smith",
               plot_marker_hack=True)
    plot_example(tb, "Legend")
    pp.legend(["S11", "S22", "Polyline", "Z \u2192 0.125l/\u03BB"])

    divs = [1, 3, 7]
    pp.subplot(2, 3, 2, projection="smith",
               grid_minor_enable=True,
               grid_minor_fancy=True,
               grid_minor_fancy_dividers=divs)
    plot_example(tb, "Minor fancy dividers=%s" % divs)

    pp.subplot(2, 3, 3, projection="smith",
               axes_radius=0.3)
    plot_example(tb, "Axes radius: 0.25")

    pp.subplot(2, 3, 4, projection="smith",
               symbol_infinity="Inf",
               symbol_infinity_correction=0,
               symbol_ohm="Ohm")
    plot_example(tb, "Infinity symbol: 'Inf' — Ohm symbol: Ohm")

    pp.subplot(2, 3, 5, projection="smith",
               grid_locator_precision=4)
    plot_example(tb, "Grid Locator Precision: 4")

    pp.subplot(2, 3, 6, projection="smith",
               axes_xlabel_rotation=0)
    plot_example(tb, "Axes X Label Rotation: 0")

    savefig(tb)


build_all = True
build_path = os.path.join(get_app_path(),"build")

if __name__ == '__main__':
    # clear and create path
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
        time.sleep(0.5)
    os.makedirs(build_path)

    if build_all:
        print("Start parallel testbenches...")
        p = Pool()
        r = []
        for key, func in locals().copy().items():
            if isinstance(func, FunctionType) and "tb_" in key:
                r += [p.apply_async(func, {})]

        for proc in r:
            proc.get()
    else:
        pass
        # tb_grid_styles()
        # tb_fancy_grids()
        # tb_grid_locators()
        # tb_normalize()
        tb_markers()
        # tb_interpolation()
        # tb_misc()
        pp.show()

    print("build finished")
    input("Press Enter to exit")
