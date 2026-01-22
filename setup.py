import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name="pysmithplot",
      version="0.2.2",
      packages=["smithplot"],
      description="An extension for Matplotlib providing a projection class to generate high quality Smith Chart plots.",
      long_description=read('README.md'),
      author="Paul Staerke",
      author_email="paul.staerke@gmail.com",
      license="BSD",
      url="https://github.com/vMeijin/pySmithPlot",
      install_requires=["matplotlib >= 1.2.0", "numpy", "scipy"])
