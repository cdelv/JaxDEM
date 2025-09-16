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
from typing import List, Optional, Tuple, Any
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


def _np_tree(tree: Any) -> Any:
    """
    Converts every JAX array leaf of a PyTree to a host `numpy.ndarray`.
    All other leaves (non-JAX arrays) are returned unchanged.

    Parameters
    ----------
    tree : Any
        A PyTree (e.g., a :class:`State` or :class:`System` object) containing JAX arrays as leaves.

    Returns
    -------
    Any
        A new PyTree with the same structure as the input `tree`, but with all
        JAX array leaves converted to `numpy.ndarray` on the host CPU.
    """

    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, jax.Array) else x, tree
    )


def _pad2d(arr: np.ndarray) -> np.ndarray:
    """
    Pads a 2D array (geometric sense) with a zero Z-component to make it 3D.

    If `arr`'s last axis has a length of 2 (representing X, Y coordinates),
    a zero Z-component is appended to it.

    Parameters
    ----------
    arr : numpy.ndarray
        The input NumPy array. Its last dimension determines if padding occurs.

    Returns
    -------
    numpy.ndarray
        The padded array if its last dimension was 2, otherwise the original array.
        The returned array will have its last dimension as 3.
    """

    if arr.shape[-1] == 2:
        return np.pad(arr, (*[(0, 0)] * (arr.ndim - 1), (0, 1)), "constant")
    return arr


def _is_safe_to_clean(path: pathlib.Path) -> bool:
    """
    Return True iff it's safe to delete the target directory.
    We refuse to clean if `path` resolves to:
      - the current working directory
      - any ancestor of the current working directory
      - the filesystem root (or drive root on Windows)
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
        directory: pathlib.Path | str,
        binary: bool,
    ) -> int:
        """
        Writes a simulation snapshot to a VTK PolyData file.

        This abstract method is the core interface for all concrete VTK writers.
        Implementations should convert the provided JAX-based `state` and `system`
        data into VTK data structures and write them to a file.

        Parameters
        ----------
        state : State
            The simulation :class:`State` snapshot to be written.
        system : System
            The simulation :class:`System` configuration.
        counter : int
            A global, monotonically increasing integer identifier to be embedded
            in the file name (e.g., `spheres_00000042.vtp`). This ensures unique file names.
        directory : pathlib.Path or str
            The target directory where the VTK file should be saved.
        binary : bool
            If `True`, the VTK file should be written in binary mode.
            If `False`, it should be written in ASCII mode (human-readable).

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
    High-level front-end for writing simulation data to VTK files.

    This class orchestrates the process of converting JAX-based `State` and
    `System` PyTrees into VTK files, handling batching, trajectories, and
    dispatching to concrete :class:`VTKBaseWriter` subclasses.

    How leading axes are interpreted
    --------------------------------
    Given `state.pos.shape == (..., N, dim)` where `N` is the particle index
    and `dim` is the spatial dimension (2 or 3). Let `L` be the number of
    remaining *leading* axes (i.e., `L = state.pos.ndim - 2`).

    1.  If `L == 0`:
        The input represents a **single snapshot**. All writers directly process it.

    2.  If `L >= 1` and `trajectory` is :obj:`False` (default behavior of :meth:`save`):
        -   Axis 0 is treated as a **batch** dimension.
        -   Axes 1 through `L-1` are treated as **trajectory** dimensions within each batch.
        This means each slice along axis 0 (`state.pos[b, ...]`) is considered a separate
        batch. Each batch is then processed recursively, with its remaining leading axes
        treated as a trajectory. Separate subdirectories (e.g., `batch_00000000/`) are
        created for each batch.

    3.  If `L >= 1` and `trajectory` is :obj:`True`:
        -   **All** leading axes (from axis 0 up to `L-1`) are treated as **trajectory** dimensions.
        This is suitable for cases like "trajectory of trajectories" (e.g., from Monte Carlo runs)
        or when the primary leading dimension is explicitly time.

    Inside each batch directory (or the main `directory` for non-batched trajectories),
    every trajectory step becomes one or more VTK files per concrete writer
    (e.g., `spheres_00000042.vtp`, `domain_00000042.vtp`).

    Requirements on `system`
    ------------------------
    The `System` object may share leading axes with `state` or be broadcastable
    (e.g., a scalar `dt` for all particles/batches/frames). During the recursive
    processing, every array leaf of `system` that has a length matching the
    current leading axis (`lead`) is sliced together with the corresponding `state` slice.
    This ensures that each individual writer receives matching per-snapshot
    `State` and `System` objects.

    Notes
    -----
    - All I/O operations (file writing) are executed in a single `ThreadPoolExecutor`
      managed by the `VTKWriter` instance, allowing for concurrent file writes.
    - A global counter (`_counter`) is incremented *before* a snapshot is submitted
      to the thread pool. This guarantees unique file names for all output files,
      even when threads finish out of order due to background execution.
    - `VTKWriter` itself is **not** a JAX PyTree (it's a standard Python dataclass)
      and therefore never appears inside `jax.jit` or `jax.vmap` transforms;
      it operates purely on the Python side.
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

    def save(
        self,
        state: "State",
        system: "System",
        *,
        trajectory: bool = False,
        trajectory_axis: int = 0,
    ) -> int:
        """
        Schedules the writing of `state` / `system` to VTK files.

        This is the main public method to trigger saving data. It handles the
        interpretation of leading axes (batch vs. trajectory) and dispatches
        the write jobs to a background thread pool. The method blocks until
        all writing operations are completed and files are on disk.

        Parameters
        ----------
        state : State
            The simulation :class:`State` object to be saved.
        system : System
            The simulation :class:`System` object corresponding to the `state`.
            It should be consistent in leading dimensions with `state`.
        trajectory : bool, optional
            TO DO: EXPLAIN

        Returns
        -------
        int
            The new value of the global counter after all snapshots (including
            all batches and trajectory steps) have been written. This counter
            represents the total number of frames written so far by this writer instance.
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
                self.build_pvd_collections(directory=directory, dt=system.dt)
            case 3:
                if trajectory:
                    T, _, _ = state.pos.shape
                    for i in range(T):
                        st = jax.tree_util.tree_map(lambda x: x[i], state)
                        sys = jax.tree_util.tree_map(lambda x: x[i], system)
                        directory = self.directory / pathlib.Path(f"batch_{0:08d}")
                        self.save_frame(st, sys, directory)
                else:
                    B, _, _ = state.pos.shape
                    for i in range(B):
                        st = jax.tree_util.tree_map(lambda x: x[i], state)
                        sys = jax.tree_util.tree_map(lambda x: x[i], system)
                        directory = self.directory / pathlib.Path(f"batch_{i:08d}")
                        self.save_frame(st, sys, directory)
            case 4:
                T, B, _, _ = state.pos.shape
                for i in range(T):
                    for j in range(B):
                        st = jax.tree_util.tree_map(lambda x: x[i, j], state)
                        sys = jax.tree_util.tree_map(lambda x: x[i, j], system)
                        directory = self.directory / pathlib.Path(f"batch_{j:08d}")
                        self.save_frame(st, sys, directory)

    def save_frame(self, state, system, directory):
        state = _np_tree(state)
        system = _np_tree(system)
        for name in self.writers:
            writer_cls = VTKBaseWriter._registry[name]
            writer_cls.write(state, system, system.step_count, directory, self.binary)

    def build_pvd_collections(
        self,
        *,
        directory,
        dt: float = 1.0,
        pattern: str = "*.vtp",  # your writers emit .vtp
        time_format: str = ".12g",
    ) -> None:
        """
        Build ParaView .pvd collections for a single data directory.

        - Writes one PVD per writer prefix into the *parent* directory:
            <dirname>_<writer>.pvd  (e.g., batch_00000000_spheres.pvd)
        - Optionally also writes a combined PVD into the parent:
            <dirname>.pvd           (e.g., batch_00000000.pvd)
          using group="writer" so multiple datasets appear at each timestep.

        Timestep = float(dt) * frame, with `frame` parsed from trailing digits:
            'spheres_000042.vtp' -> frame = 42
        """
        root = pathlib.Path(directory)
        if not root.exists() or not root.is_dir():
            return

        # robust float extraction if dt is JAX/NumPy scalar
        try:
            dt_val = float(np.asarray(dt))
        except Exception:
            dt_val = float(dt)

        # Match '<writer>_<frame>.<ext>', capture writer and frame
        # e.g. 'spheres_000123.vtp' -> writer='spheres', frame=123
        name_regex = re.compile(r"^(?P<writer>[^_.][^_]*)_(?P<frame>\d+)\.[^.]+$")

        # Collect files by writer prefix
        groups: dict[str, list[tuple[int, str]]] = {}
        for f in sorted(root.glob(pattern)):
            m = name_regex.match(f.name)
            if not m:
                continue
            writer = m.group("writer")
            frame = int(m.group("frame"))
            groups.setdefault(writer, []).append((frame, f.name))

        if not groups:
            return

        # Sort frames within each writer
        for w in list(groups.keys()):
            groups[w].sort(key=lambda x: x[0])

        parent = root.parent
        dname = root.name  # used in PVD filenames and relative file paths

        # ---------- write per-writer PVDs ----------
        for writer, items in groups.items():
            vtk_file_element = ET.Element(
                "VTKFile", type="Collection", version="0.1", byte_order="LittleEndian"
            )
            collection_element = ET.SubElement(vtk_file_element, "Collection")

            for frame, fname in items:
                t = dt_val * frame
                ET.SubElement(
                    collection_element,
                    "DataSet",
                    timestep=format(t, time_format),
                    file=f"{dname}/{fname}",  # relative to the PVD (in the parent)
                )

            pvd_path = parent / f"{dname}_{writer}.pvd"
            ET.ElementTree(vtk_file_element).write(
                pvd_path, encoding="utf-8", xml_declaration=True
            )


@VTKBaseWriter.register("spheres")
class SpheresWriter(VTKBaseWriter):
    """
    A `VTKBaseWriter` implementation that writes particle centers as VTK points
    and attaches per-particle `State` fields as PointData attributes.

    For each particle, its position is treated as a point. Other relevant
    per-particle fields from the :class:`State` object (like `vel`, `rad`, `mass`, etc.)
    are added as attributes to these points in the VTK file.

    Notes
    -----
    - Particle positions are padded to 3D if they are originally 2D, as required by VTK.
    - Only 1D scalar fields (like `rad`, `mass`) and 2D/3D vector fields (like `vel`, `accel`)
      are included as PointData. Higher-rank fields or non-array fields are ignored.
    - Boolean arrays (e.g., `fixed`) are converted to `int8` before being passed to VTK.
    """

    @classmethod
    def write(
        cls,
        state: "State",
        system: "System",  # system is technically not used in SpheresWriter.write, but kept for signature consistency
        counter: int,
        directory: pathlib.Path | str,
        binary: bool,
    ) -> int:
        """
        Writes particle data from a single snapshot to a VTK PolyData (.vtp) file.

        The file will contain a set of points representing particle centers,
        and each particle's relevant properties from the `State` will be saved
        as point attributes.

        Parameters
        ----------
        state : State
            The simulation state snapshot (NumPy-converted, not JAX arrays).
        system : System
            The simulation system configuration (NumPy-converted).
            (Note: Not directly used by `SpheresWriter`, but required by base signature.)
        counter : int
            The unique integer identifier for this snapshot.
        directory : pathlib.Path or str
            The target directory for the output file.
        binary : bool
            If `True`, writes in binary mode; `False` for ASCII.

        Returns
        -------
        int
            The incremented counter (`counter + 1`).
        """
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / f"spheres_{counter:08d}.vtp"

        # state is already _np_tree-converted by _dispatch
        pos = _pad2d(state.pos)  # Ensure positions are 3D for VTK
        n = pos.shape[0]  # Number of particles

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(pos))  # Convert numpy array to VTK data
        poly.SetPoints(points)

        # Add every per-particle dataclass field as PointData
        for fld in fields(state):  # Iterate through dataclass fields of the State
            name = fld.name
            if name == "pos":
                continue  # Position is handled as points, not a separate array

            arr = getattr(state, name)  # Get the numpy array for the field
            # Skip if not a numpy array or if its leading dimension doesn't match particle count
            if not isinstance(arr, np.ndarray) or arr.shape[0] != n:
                continue

            # Handle boolean arrays by converting to int8 (VTK compatible)
            if arr.dtype == np.bool_:
                arr = arr.astype(np.int8)

            # Reshape based on array dimension for VTK
            if arr.ndim == 1:  # Scalar per particle (e.g., radius, mass)
                data = arr.reshape(n, 1)
            elif arr.ndim == 2 and arr.shape[1] in (
                2,
                3,
            ):  # 2D or 3D vector (e.g., velocity, accel)
                data = _pad2d(arr)  # Pad 2D vectors to 3D for VTK compatibility
            else:
                continue  # Skip unsupported shapes (e.g., higher-rank tensors)

            vtk_arr = vtk_np.numpy_to_vtk(data, deep=1)  # Convert to VTK array
            vtk_arr.SetName(name)  # Set the name of the data array in VTK
            poly.GetPointData().AddArray(vtk_arr)  # Add to the PolyData's PointData

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(poly)
        # Set data mode based on binary flag
        writer.SetDataModeToBinary() if binary else writer.SetDataModeToAscii()
        writer.Write()  # Write the file to disk
        return counter + 1


@VTKBaseWriter.register("domain")
class DomainWriter(VTKBaseWriter):
    """
    A `VTKBaseWriter` implementation that writes the simulation domain as a VTK geometric primitive.

    The domain is represented as an axis-aligned cuboid (for 3D simulations) or
    a rectangle (for 2D simulations). Its position is determined by `system.domain.anchor`
    and `system.domain.box_size`.

    Notes
    -----
    - The domain's dimensions from :attr:`system.domain.box_size` are automatically padded
      to 3D if the simulation is 2D, as required by VTK's `vtkCubeSource`.
    - The center of the VTK cube/rectangle is set to `anchor + 0.5 * box_size`.
    """

    @classmethod
    def write(
        cls,
        state: "State",  # state is technically not used in DomainWriter.write, but kept for signature consistency
        system: "System",
        counter: int,
        directory: pathlib.Path | str,
        binary: bool,
    ) -> int:
        """
        Writes the simulation domain geometry to a VTK PolyData (.vtp) file.

        The domain is represented as a `vtkCubeSource`, automatically adjusting
        for 2D or 3D simulation dimensions.

        Parameters
        ----------
        state : State
            The simulation state snapshot (NumPy-converted).
            (Note: Not directly used by `DomainWriter`, but required by base signature.)
        system : System
            The simulation system configuration (NumPy-converted), providing
            `domain.anchor` and `domain.box_size`.
        counter : int
            The unique integer identifier for this snapshot.
        directory : pathlib.Path or str
            The target directory for the output file.
        binary : bool
            If `True`, writes in binary mode; `False` for ASCII.

        Returns
        -------
        int
            The incremented counter (`counter + 1`).
        """
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / f"domain_{counter:08d}.vtp"

        box = _pad2d(system.domain.box_size)  # Ensure box_size is 3D (X, Y, Z)
        anch = _pad2d(system.domain.anchor)  # Ensure anchor is 3D (X, Y, Z)

        cube = vtk.vtkCubeSource()
        cube.SetXLength(box[0])  # Set X dimension
        cube.SetYLength(box[1])  # Set Y dimension
        cube.SetZLength(box[2])  # Set Z dimension
        cube.SetCenter(*(anch + 0.5 * box))  # Set the center of the cube
        cube.Update()  # Compute the output data

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(cube.GetOutput())  # Set the cube's generated data as input
        # Set data mode based on binary flag
        writer.SetDataModeToBinary() if binary else writer.SetDataModeToAscii()
        writer.Write()  # Write the file to disk
        return counter + 1
