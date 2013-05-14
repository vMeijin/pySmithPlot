#!/usr/bin/env python

from utils import parseCSV

import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import smithplot

# sample data 
data = parseCSV("data/s11", startRow=1, steps=10)
val1 = data[:, 1] + data[:, 2] * 1j

data = parseCSV("data/s22", startRow=1, steps=10)
val2 = data[:, 1] + data[:, 2] * 1j

line = np.array([0.7 + 0.2j, 0.7 + 1.8j, 0.3 + 1.8j, 2])

# plot data
plt.figure(figsize=(8, 8))

ax = plt.subplot(1, 1, 1, projection='smith', axes_norm=50)
plt.plot(val1, markevery=5, label="S11")
plt.plot(val2, markevery=5, label="S22")
ax.plot_vswr_circle(0.3 - 0.7j, real=1, solution2=True, label="Re(Z)->1")

plt.plot(line, path_interpolation=0, label="Polyline")

plt.legend(loc="lower right")
plt.title("Matplotlib Smith Chart Projection")

plt.show()
