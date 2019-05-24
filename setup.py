from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
    Extension("cynthonized_functions",
              sources=["cynthonized_functions.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(name="cynthonized_functions",
      ext_modules=cythonize(ext_modules))
