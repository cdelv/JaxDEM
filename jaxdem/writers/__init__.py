# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining data writers.

This module provides a high-level VTKWriter and CheckpointWriter frontend, a VTKBaseWriter
plugin interface, and concrete writers (e.g., VTKSpheresWriter, VTKDomainWriter)
for exporting JAX-based simulation snapshots to VTK files.
"""

from __future__ import annotations

from pathlib import Path
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from dataclasses import dataclass

from ..factory import Factory

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass(slots=True, frozen=True)
class VTKBaseWriter(Factory, ABC):
    """
    Abstract base class for writers that output simulation data.

    Concrete subclasses implement the `write` method to specify how a given
    snapshot (:class:`jaxdem.State`, :class:`jaxdem.System` pair) is converted into a
    specific file format.

    Example
    -------
    To define a custom VTK writer, inherit from `VTKBaseWriter` and implement its abstract methods:

    >>> @VTKBaseWriter.register("my_custom_vtk_writer")
    >>> @dataclass(slots=True, flozen=True)
    >>> class MyCustomVTKWriter(VTKBaseWriter):
            ...
    """

    @classmethod
    @abstractmethod
    def write(
        cls,
        state: "State",
        system: "System",
        filename: Path,
        binary: bool,
    ):
        """
        Write information from a simulation snapshot to a VTK PolyData file.

        This abstract method is the core interface for all concrete VTK writers.
        Implementations should assume that all jax arrays are converted to numpy arrays
        before write is called.

        Parameters
        ----------
        state : State
            The simulation :class:`jaxdem.State` snapshot to be written.
        system : System
            The simulation :class:`jaxdem.System` configuration.
        filename : Path
            Target path where the VTK file should be saved. The caller
            guarantees that it exists.
        binary : bool
            If `True`, the VTK file is written in binary mode; if `False`,
            it is written in ASCII (human-readable) mode.
        """
        raise NotImplementedError


from .vtkDomainWriter import VTKDomainWriter
from .vtkSpheresWriter import VTKSpheresWriter
from .vtkWriter import VTKWriter
from .checkpointWriter import CheckpointWriter


__all__ = [
    "VTKBaseWriter",
    "VTKWriter",
    "CheckpointWriter",
    "VTKDomainWriter",
    "VTKSpheresWriter",
]
