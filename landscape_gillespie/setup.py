# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:16:22 2019

@author: iasonas
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("simulate_cy.pyx"),
    include_dirs=[numpy.get_include()]
)