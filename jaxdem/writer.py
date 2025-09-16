# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining data writers.
"""

import jax
import jax.numpy as jnp

import pathlib
import shutil
import concurrent.futures as cf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import List, Optional
import math
import re

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
import xml.etree.ElementTree as ET

from .factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .system import System


def _is_safe_to_clean(path: pathlib.Path) -> bool:
    """
    Return True if and only if it is safe to delete the target directory.

    Cleaning is refused when `path` resolves to:
      - the current working directory,
      - any ancestor of the current working directory, or
      - the filesystem root (or drive root on Windows).
    """
    p = path.resolve()
    cwd = pathlib.Path.cwd().resolve()

    # never nuke CWD, any parent of CWD, or the root/drive
    if p == cwd or p in cwd.parents:
        return False
    if p == pathlib.Path(p.anchor):  # '/' on POSIX, 'C:\\' on Windows
        return False
    return True


class VTKBaseWriter(Factory, ABC):
    """
    Abstract base class for writers that output simulation data.

    Concrete subclasses implement the `write` method to specify how a given
    snapshot (:class:`State`, :class:`System` pair) is converted into a
    specific file format.

    Notes
    -----
    - These writers are registered with the :class:`VTKBaseWriter` factory
      and orchestrated by the :class:`VTKWriter` frontend.
    """

    __slots__ = ()

    @classmethod
    @abstractmethod
    def write(
        cls,
        state: "State",
        system: "System",
        counter: int,
        directory: pathlib.Path,
        binary: bool,
    ) -> int:
        """
        Write a simulation snapshot to a VTK PolyData file.

        This abstract method is the core interface for all concrete VTK writers.
        Implementations should convert the provided (host) array data from
        `state` and `system` into VTK data structures and persist them to disk.

        Parameters
        ----------
        state : State
            The simulation :class:`State` snapshot to be written.
        system : System
            The simulation :class:`System` configuration.
        counter : int
            A global, monotonically increasing integer identifier embedded
            in the file name (e.g., `spheres_00000042.vtp`) to ensure uniqueness.
        directory : pathlib.Path
            Target directory where the VTK file should be saved. The caller
            guarantees that it exists.
        binary : bool
            If `True`, the VTK file is written in binary mode; if `False`,
            it is written in ASCII (human-readable) mode.

        Returns
        -------
        int
            The counter value `counter + 1`.

        Raises
        ------
        NotImplementedError
            This is an abstract method and must be implemented by subclasses.
        """
        raise NotImplementedError


@dataclass(slots=True)
class VTKWriter:
    """
    High-level front end for writing simulation data to VTK files.

    This class orchestrates the conversion of JAX-backed :class:`State` and
    :class:`System` pytrees into VTK files, handling batches, trajectories,
    and dispatch to registered :class:`VTKBaseWriter` subclasses.

    How leading axes are interpreted
    --------------------------------
    Let particle positions have shape ``(..., N, dim)``, where ``N`` is the
    number of particles and ``dim`` is 2 or 3. Define ``L = pos.ndim - 2``,
    i.e., the number of *leading* axes before ``(N, dim)``.

    * ``L == 0`` — single snapshot:
        The input is treated as one frame and is written directly.

    * ``L >= 1`` with ``trajectory=False`` (default in :meth:`save`):
        Axis 0 is interpreted as a **batch** dimension. Any remaining leading
        axes (if present) are treated as a **trajectory** within each batch.
        Each batch is written into its own subdirectory
        ``batch_XXXXXXXX`` under ``self.directory``.

    * ``L >= 1`` with ``trajectory=True``:
        Axis 0 is interpreted as **time**. All leading axes are considered
        part of the trajectory (after an optional `trajectory_axis` swap),
        and frames are written in temporal order. If additional leading axes
        exist (i.e., a trajectory of batches), the batch axes are flattened
        so that each batch gets its own output directory and time series.

    The method performs the necessary axis swaps and reshaping:
    - If ``trajectory`` is True, ``trajectory_axis`` is swapped to axis 0.
    - When ``L >= 2``, batch axes (and, if applicable, non-time leading axes)
      are flattened to produce either ``(T, B, N, dim)`` or ``(B, N, dim)``.
    - Concrete writers receive per-frame NumPy arrays.
    """

    writers: Optional[List[str]] = None
    """
    A list of strings specifying which registered :class:`VTKBaseWriter`
    subclasses should be used for writing. If `None`, all available
    `VTKBaseWriter` subclasses will be used.
    """

    directory: str | pathlib.Path = "frames"
    """
    The base directory where output VTK files will be saved.
    Subdirectories might be created within this path for batched outputs.
    Defaults to "frames".
    """

    binary: bool = True
    """
    If :obj:`True`, VTK files will be written in binary format.
    If :obj:`False`, files will be written in ASCII format.
    Defaults to :obj:`True`.
    """

    clean: bool = True
    """
    If :obj:`True`, the `directory` will be completely emptied before any
    files are written. Defaults to :obj:`True`. This is useful for
    starting a fresh set of output frames.
    """

    _counter: int = 0
    """
    Internal global counter for generating unique file names. Initialized to 0.
    """

    _pool: cf.ThreadPoolExecutor = field(
        default_factory=cf.ThreadPoolExecutor,
        init=False,  # Field is initialized in __post_init__ or factory
        repr=False,  # Exclude from default __repr__
    )
    """
    Internal :class:`concurrent.futures.ThreadPoolExecutor` used for asynchronous
    file writing, allowing I/O operations to run in the background.
    """

    _writer_classes: List = field(default_factory=list)

    def __post_init__(self):
        self.directory = pathlib.Path(self.directory)
        available = list(VTKBaseWriter._registry.keys())
        if self.writers is None:
            self.writers = available
        unknown = [w for w in self.writers if w not in available]
        if unknown:
            raise KeyError(
                f"Unknown VTK writers {unknown}.  " f"Available: {available}"
            )
        # Ensure the directory exists and clean if requested
        if self.clean and self.directory.exists():
            if _is_safe_to_clean(self.directory):
                shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self._writer_classes = [VTKBaseWriter._registry[name] for name in self.writers]

    def save(
        self,
        state: "State",
        system: "System",
        *,
        trajectory: bool = False,
        trajectory_axis: int = 0,
    ):
        """
        Schedule writing of a :class:`State` / :class:`System` pair to VTK files.

        This public entry point interprets leading axes (batch vs. trajectory),
        performs any required axis swapping and flattening, and then writes the
        resulting frames using the registered writers. It also creates per-batch
        ParaView ``.pvd`` collections referencing the generated files.

        Parameters
        ----------
        state : State
            The simulation :class:`State` object to be saved. Its array leaves
            must end with ``(N, dim)``.
        system : System
            The :class:`System` object corresponding to `state`. Leading axes
            must be compatible (or broadcastable) with those of `state`.
        trajectory : bool, optional
            If ``True``, interpret ``trajectory_axis`` as time and write a
            trajectory; if ``False``, interpret the leading axis as batch.
        trajectory_axis : int, optional
            The axis in `state`/`system` to treat as the trajectory (time) axis
            when ``trajectory=True``. This axis is swapped to the front prior
            to writing.
        """
        Ndim = state.pos.ndim

        # Make sure trajectory axis is axis 0
        if trajectory:
            state = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, trajectory_axis, 0), state
            )
            system = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, trajectory_axis, 0), system
            )

        # flatten all batch dimensions
        if Ndim >= 4:
            L = Ndim - 2
            if trajectory:
                state = jax.tree_util.tree_map(
                    lambda x: x.reshape(
                        (
                            x.shape[0],
                            math.prod(x.shape[1:L]),
                        )
                        + x.shape[L:]
                    ),
                    state,
                )
                system = jax.tree_util.tree_map(
                    lambda x: x.reshape(
                        (
                            x.shape[0],
                            math.prod(x.shape[1:L]),
                        )
                        + x.shape[L:]
                    ),
                    system,
                )
            else:
                state = jax.tree_util.tree_map(
                    lambda x: x.reshape((math.prod(x.shape[:L]),) + x.shape[L:]), state
                )
                system = jax.tree_util.tree_map(
                    lambda x: x.reshape((math.prod(x.shape[:L]),) + x.shape[L:]), system
                )

        match state.pos.ndim:
            case 2:
                directory = self.directory / pathlib.Path(f"batch_{0:08d}")
                self.save_frame(state, system, directory)
                self.build_pvd_collections(directory=directory, dt=float(system.dt))
            case 3:
                if trajectory:
                    T, _, _ = state.pos.shape
                    directory = self.directory / pathlib.Path(f"batch_{0:08d}")
                    for i in range(T):
                        st = jax.tree_util.tree_map(lambda x: x[i], state)
                        sys = jax.tree_util.tree_map(lambda x: x[i], system)
                        self.save_frame(st, sys, directory)
                    self.build_pvd_collections(
                        directory=directory, dt=float(system.dt[-1])
                    )
                else:
                    B, _, _ = state.pos.shape
                    for i in range(B):
                        st = jax.tree_util.tree_map(lambda x: x[i], state)
                        sys = jax.tree_util.tree_map(lambda x: x[i], system)
                        directory = self.directory / pathlib.Path(f"batch_{i:08d}")
                        self.save_frame(st, sys, directory)
                    for i in range(B):
                        directory = self.directory / pathlib.Path(f"batch_{i:08d}")
                        self.build_pvd_collections(
                            directory=directory, dt=float(system.dt[i])
                        )
            case 4:
                T, B, _, _ = state.pos.shape
                for i in range(T):
                    for j in range(B):
                        st = jax.tree_util.tree_map(lambda x: x[i, j], state)
                        sys = jax.tree_util.tree_map(lambda x: x[i, j], system)
                        directory = self.directory / pathlib.Path(f"batch_{j:08d}")
                        self.save_frame(st, sys, directory)
                for j in range(B):
                    directory = self.directory / pathlib.Path(f"batch_{j:08d}")
                    self.build_pvd_collections(
                        directory=directory, dt=float(system.dt[-1, j])
                    )

    def save_frame(self, state, system, directory):
        """
        Write a single simulation snapshot to disk using all registered writers.

        This ensures the target directory exists, converts every leaf in the
        ``state`` and ``system`` pytrees to host ``numpy.ndarray`` via
        ``np.asarray``, and then invokes each writer class.

        Parameters
        ----------
        state : State
            Snapshot of the simulation state.
        system : System
            Matching system/configuration object for the snapshot.
        directory : pathlib.Path
            Output directory for this snapshot. It will be created if it does not
            already exist.
        """
        directory.mkdir(parents=True, exist_ok=True)
        state = jax.tree_util.tree_map(lambda x: np.asarray(x), state)
        system = jax.tree_util.tree_map(lambda x: np.asarray(x), system)
        for cls in self._writer_classes:
            cls.write(state, system, system.step_count, directory, self.binary)

    def build_pvd_collections(
        self,
        *,
        directory: pathlib.Path,
        dt: float = 1.0,
        pattern: str = "*.vtp",  # your writers emit .vtp
        time_format: str = ".12g",
    ) -> None:
        """
        Build ParaView ``.pvd`` time-collection files for a single data directory.

        This scans ``directory`` (non-recursive) for files matching ``pattern``
        whose names follow ``<writer>_<frame>.<ext>`` (e.g., ``spheres_00001234.vtp``).
        Files are grouped by the writer prefix and sorted by their numeric frame.
        A PVD file per writer is then written into the *parent* directory with
        the name ``<dirname>_<writer>.pvd``, where ``<dirname>`` is
        ``directory.name``. Each dataset entry references ``<dirname>/<file>``,
        and its timestep is computed as ``dt * frame`` formatted with
        ``time_format``.

        Parameters
        ----------
        directory : pathlib.Path
            The directory that contains the time-varying datasets to index
            (e.g., ``frames/batch_00000000``). The scan is **not recursive**.
        dt : float
            Physical timestep per frame. The timestep written into the PVD is
            ``dt * frame``, where ``frame`` is parsed from the filename's trailing
            digits (e.g., ``..._000042.vtp`` → ``frame=42``).
        pattern : str
            Glob pattern used to select datasets inside ``directory``. Use
            ``"*.vtp"`` for PolyData or ``"*.vtu"`` for UnstructuredGrid outputs.
        time_format : str
            Python format specifier applied to the timestep when writing the XML
            attribute.

        Returns
        -------
        None
        """
        # Match '<writer>_<frame>.<ext>', capture writer and frame
        # e.g. 'spheres_000123.vtp' -> writer='spheres', frame=123
        name_regex = re.compile(r"^(?P<writer>[^_.][^_]*)_(?P<frame>\d+)\.[^.]+$")

        # Collect files by writer prefix
        groups: dict[str, list[tuple[int, str]]] = {}
        for f in directory.glob(pattern):
            m = name_regex.match(f.name)
            if not m:
                continue
            writer = m.group("writer")
            frame = int(m.group("frame"))
            groups.setdefault(writer, []).append((frame, f.name))

        if not groups:
            return

        for w in list(groups.keys()):
            groups[w].sort(key=lambda x: x[0])

        parent = directory.parent
        dname = directory.name

        for writer, items in groups.items():
            vtk_file_element = ET.Element(
                "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
            )
            collection_element = ET.SubElement(vtk_file_element, "Collection")

            for frame, fname in items:
                t = dt * frame
                ET.SubElement(
                    collection_element,
                    "DataSet",
                    timestep=format(t, time_format),
                    file=f"{dname}/{fname}",
                )

            pvd_path = parent / f"{dname}_{writer}.pvd"
            ET.ElementTree(vtk_file_element).write(
                pvd_path, encoding="utf-8", xml_declaration=True
            )


@VTKBaseWriter.register("spheres")
class SpheresWriter(VTKBaseWriter):
    """
    A :class:`VTKBaseWriter` that writes particle centers as VTK points and
    attaches per-particle :class:`State` fields as ``PointData`` attributes.

    For each particle, its position is written as a point. Relevant per-particle
    fields (e.g., ``vel``, ``rad``, ``mass``) are exported as arrays.
    Positions and 2-component vectors are padded to 3D as required by VTK.
    """

    @classmethod
    def write(
        cls,
        state: "State",
        system: "System",
        counter: int,
        directory: pathlib.Path,
        binary: bool,
    ) -> int:
        """
        Write particle data from a single snapshot to a VTK PolyData (``.vtp``) file.

        The file contains points for particle centers and one array per eligible
        per-particle field from :class:`State`.

        Parameters
        ----------
        state : State
            The simulation state snapshot (NumPy-converted).
        system : System
            The simulation system configuration (NumPy-converted).
        counter : int
            The unique integer identifier for this snapshot.
        directory : pathlib.Path
            The target directory for the output file.
        binary : bool
            If `True`, writes in binary mode; `False` for ASCII.

        Returns
        -------
        int
            The incremented counter (``counter + 1``).
        """
        filename = directory / f"spheres_{counter:08d}.vtp"
        pos = state.pos
        n = pos.shape[0]
        if pos.shape[-1] == 2:
            pos = np.pad(pos, (*[(0, 0)] * (pos.ndim - 1), (0, 1)), "constant")

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(pos))
        poly.SetPoints(points)

        for fld in fields(state):
            name = fld.name
            arr = getattr(state, name)
            if name != "pos" and (isinstance(arr, np.ndarray) or arr.shape[0] == n):
                if arr.dtype == np.bool_:
                    arr = arr.astype(np.int8)

                if arr.shape[-1] == 2:
                    arr = np.pad(arr, (*[(0, 0)] * (arr.ndim - 1), (0, 1)), "constant")

                vtk_arr = vtk_np.numpy_to_vtk(arr)
                vtk_arr.SetName(name)
                poly.GetPointData().AddArray(vtk_arr)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(poly)
        writer.SetDataModeToBinary() if binary else writer.SetDataModeToAscii()
        writer.Write()
        return counter + 1


@VTKBaseWriter.register("domain")
class DomainWriter(VTKBaseWriter):
    """
    A :class:`VTKBaseWriter` that writes the simulation domain as a VTK geometric
    primitive.

    The domain is represented as an axis-aligned cuboid (3D) or rectangle (2D),
    using a ``vtkCubeSource``. If input arrays are 2D, they are padded to 3D
    as required by VTK.
    """

    @classmethod
    def write(
        cls,
        state: "State",
        system: "System",
        counter: int,
        directory: pathlib.Path,
        binary: bool,
    ) -> int:
        """
        Write the simulation domain geometry to a VTK PolyData (``.vtp``) file.

        The domain is produced with ``vtkCubeSource`` using
        ``system.domain.box_size`` and ``system.domain.anchor``.

        Parameters
        ----------
        state : State
            The simulation state snapshot (NumPy-converted).
        system : System
            The simulation system configuration (NumPy-converted).
        counter : int
            The unique integer identifier for this snapshot.
        directory : pathlib.Path
            The target directory for the output file.
        binary : bool
            If `True`, writes in binary mode; `False` for ASCII.

        Returns
        -------
        int
            The incremented counter (``counter + 1``).
        """
        filename = directory / f"domain_{counter:08d}.vtp"

        box = system.domain.box_size
        anch = system.domain.anchor
        if box.shape[-1] == 2:
            box = np.pad(box, (*[(0, 0)] * (box.ndim - 1), (0, 1)), "constant")

        if anch.shape[-1] == 2:
            anch = np.pad(anch, (*[(0, 0)] * (anch.ndim - 1), (0, 1)), "constant")

        cube = vtk.vtkCubeSource()
        cube.SetXLength(float(box[0]))
        cube.SetYLength(float(box[1]))
        cube.SetZLength(float(box[2]))
        cube.SetCenter(*(anch + 0.5 * box))
        cube.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(cube.GetOutput())
        writer.SetDataModeToBinary() if binary else writer.SetDataModeToAscii()
        writer.Write()
        return counter + 1
