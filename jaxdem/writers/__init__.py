# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Interface for defining data writers.

This module provides a high-level VTKWriter and CheckpointWriter frontend, a VTKBaseWriter
plugin interface, and concrete writers (e.g., VTKSpheresWriter, VTKDomainWriter)
for exporting JAX-based simulation snapshots to VTK files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..factory import Factory

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass(slots=True)
class VTKBaseWriter(Factory, ABC):
    """Abstract base class for writers that output simulation data.

    Concrete subclasses implement the `write` method to specify how a given
    snapshot (:class:`jaxdem.State`, :class:`jaxdem.System` pair) is converted into a
    specific file format.

    Example:
    --------
    To define a custom VTK writer, inherit from `VTKBaseWriter` and implement its abstract methods:

    >>> @VTKBaseWriter.register("my_custom_vtk_writer")
    >>> @dataclass(slots=True)
    >>> class MyCustomVTKWriter(VTKBaseWriter):
            ...

    """

    @classmethod
    def is_active(cls, state: State, system: System) -> bool:
        """Check whether this writer has data to write for the given state and system."""
        return True

    @staticmethod
    def _write_polydata(
        poly: Any, filename: Path, binary: bool, *, label: str = "VTK writer"
    ) -> None:
        """Shared epilogue: write a ``vtkPolyData`` object to ``filename``.

        Parameters
        ----------
        poly : vtk.vtkPolyData
            The polydata object to write.
        filename : Path
            Target path of the ``.vtp`` file.
        binary : bool
            If True, write compressed binary; otherwise ASCII.
        label : str
            Human-readable writer name used in the error message.
        """
        import vtk  # type: ignore[import-untyped]

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(poly)
        if binary:
            writer.SetDataModeToAppended()
            compressor = vtk.vtkZLibDataCompressor()
            writer.SetCompressor(compressor)
        else:
            writer.SetDataModeToAscii()
        ok = writer.Write()
        if ok != 1:
            raise RuntimeError(f"{label} failed")

    @classmethod
    @abstractmethod
    def write(
        cls,
        state: State,
        system: System,
        filename: Path,
        binary: bool,
    ) -> None:
        """Write information from a simulation snapshot to a VTK PolyData file.

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


from .checkpoints import (
    CheckpointLoader,
    CheckpointModelLoader,
    CheckpointModelWriter,
    CheckpointWriter,
)
from .vtk_deformable_particle_writer import (
    VTKDeformableEdgeAdjacenciesWriter,
    VTKDeformableEdgesWriter,
    VTKDeformableElementsWriter,
)
from .vtk_domain_writer import VTKDomainWriter
from .vtk_facets_writer import VTKFacetsWriter
from .vtk_spheres_writer import VTKFacetSpheresWriter, VTKSpheresWriter
from .vtk_writer import VTKWriter

__all__ = [
    "CheckpointLoader",
    "CheckpointModelLoader",
    "CheckpointModelWriter",
    "CheckpointWriter",
    "VTKBaseWriter",
    "VTKDeformableEdgeAdjacenciesWriter",
    "VTKDeformableEdgesWriter",
    "VTKDeformableElementsWriter",
    "VTKDomainWriter",
    "VTKSpheresWriter",
    "VTKFacetSpheresWriter",
    "VTKFacetsWriter",
    "VTKWriter",
]
