
# from core import __version__
from distutils.core import setup

setup(
    name='coordrdinate',
    py_modules=['coordinate'],
    version='1.0.0',
    description='A library of managed coordinates for transitions between coordinate systems',
    long_description='A library of managed coordinates for conversion between various coordinate systems, '
                     'supported coordinate axes are: Cartesian2D, Polar, Cartesian3D, Cylindrical, '
                     'Spherical coordinates',
    author='Hurd',
    url='https://github.com/hurd-git/coordrdinate',
    requires=['numpy'],
)

