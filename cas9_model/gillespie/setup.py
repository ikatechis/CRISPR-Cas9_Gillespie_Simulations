# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 00:16:22 2019

@author: iason
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("cyGillespie.pyx"),
    include_dirs=[numpy.get_include()]
)