from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import numpy

setup(
    name="ptudr",
    version="0.0.1",
    author="Max Budninskiy",
    description=(
        "Parallel Transport Unfolding: Quasi-Isometric Manifold Learning"
    ),
    url="https://github.com/mbudnins/ptudr",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    ext_modules=cythonize(
        Extension(
            "ptu_dijkstra",
            ["parallel_transport_unfolding/ptu_dijkstra.pyx"],
            include_dirs=[numpy.get_include()]
        )
    )
)
