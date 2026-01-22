#!/usr/bin/env python3

import sys
import os

import numpy as np
from matplotlib import rcParams, pyplot as pp

rcParams.update({"legend.numpoints": 3})

sys.path.append("..")
from smithplot import SmithAxes
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

# sample data
data = np.loadtxt(os.path.join(get_app_path(),"data/s11.csv"), delimiter=",", skiprows=1)[::100]
val1 = data[:, 1] + data[:, 2] * 1j

data = np.loadtxt(os.path.join(get_app_path(),"data/s22.csv"), delimiter=",", skiprows=1)[::100]
val2 = data[:, 1] + data[:, 2] * 1j

# plot data
pp.figure(figsize=(6, 6))

ax = pp.subplot(1, 1, 1, projection='smith')
pp.plot([10, 100], markevery=1)


pp.plot(200 + 100j, datatype=SmithAxes.Z_PARAMETER)
pp.plot(50 * val1, label="default", datatype=SmithAxes.Z_PARAMETER)
pp.plot(50 * val2, markevery=1, label="interpolate=3", interpolate=3, datatype=SmithAxes.Z_PARAMETER)
pp.plot(val1, markevery=1, label="equipoints=22", equipoints=22, datatype=SmithAxes.S_PARAMETER)
pp.plot(val2, markevery=3, label="equipoints=22, \nmarkevery=3", equipoints=22, datatype=SmithAxes.S_PARAMETER)

#leg = 
pp.legend(loc="lower right", fontsize=12)
pp.title("Matplotlib Smith Chart Projection")

pp.savefig(os.path.join(get_app_path(),"build","export.pdf"), format="pdf", bbox_inches="tight")
pp.show()
