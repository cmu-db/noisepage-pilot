from setuptools import setup  # isort:skip

import numpy
from Cython.Build import cythonize

setup(ext_modules=cythonize("behavior/plans/diff_c.pyx"), include_dirs=[numpy.get_include()])
