# Copyright (c) 2025, Carlos Andres del Valle
#
# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions

from setuptools import setup

setup(
    name='JaxDEM',
    url='https://github.com/cdelv/JaxDEM',
    author='Carlos Andres del Valle',
    author_email='carlos.delvalleurberuaga@yale.edu',
    packages=['jaxdem'],
    install_requires=['jax', 'vtk'],
    version='0.1',
    license='BSD-3',
    description='An example of a python package from pre-existing code',
    long_description=open('README.md').read(),
)