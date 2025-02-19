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