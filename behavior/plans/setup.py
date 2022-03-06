from setuptools import Extension, setup  # isort:skip

import numpy
from Cython.Build import cythonize

extension = [
    Extension(
        "diff_c",
        ["behavior/plans/diff_c.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(ext_modules=cythonize(extension, compiler_directives={"language_level": "3"}))
