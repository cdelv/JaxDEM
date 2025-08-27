# Copyright (c) 2025, Carlos Andres del Valle
#
# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions

from setuptools import setup, find_packages


_rl_deps = ["flax", "optax", "distrax"]
_docs_deps = [
    "sphinx",
    "myst-parser",
    "pydata-sphinx-theme",
    "sphinx-gallery",
    *_rl_deps,
]

setup(
    name="JaxDEM",
    version="0.1",
    url="https://github.com/cdelv/JaxDEM",
    author="Carlos Andres del Valle",
    author_email="carlos.delvalleurberuaga@yale.edu",
    license="BSD-3",
    description="An example of a python package from pre-existing code",
    long_description=open("README.md").read(),
    packages=find_packages(),
    install_requires=[
        "jax>=0.6,<0.7",
        "jaxlib>=0.6,<0.7",
        "vtk",
        "numpy",
    ],
    extras_require={
        # Install these for reinforcement learning support: pip install JaxDEM[rl]
        "rl": _rl_deps,
        # Additional dependencies required to build the documentation
        "docs": _docs_deps,
    },
)
