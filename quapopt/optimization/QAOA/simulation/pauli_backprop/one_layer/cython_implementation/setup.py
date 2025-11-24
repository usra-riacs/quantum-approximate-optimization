# Copyright 2025 USRA
# Authors: Filip B. Maciejewski (fmaciejewski@usra.edu; filip.b.maciejewski@gmail.com)


import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# This was needed to handle relative imports in the cython file
extensions = [
    Extension(
        "cython_p1_qaoa",  # Explicit module name
        ["cython_p1_qaoa.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "cython_grad_p1_qaoa",  # Explicit module name
        ["cython_grad_p1_qaoa.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]
setup(
    ext_modules=cythonize(extensions),
)
