# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining data writers.
"""

import jax
import pathlib
import shutil
import concurrent.futures as cf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple, Any

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np

from .factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .system import System


def _np_tree(tree: Any) -> Any:
    """
    Recursively converts every JAX array leaf of a PyTree to a host `numpy.ndarray`.
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


def _slice_along_lead(tree: Any, idx: int, current_dim_size: int) -> Any:
    """
    Returns a copy of a PyTree with arrays sliced along the first leading axis.

    Recursively traverses `tree`. For any JAX array leaf that has `shape[0] == current_dim_size`,
    it is replaced by its slice `x[idx]`. Scalar leaves and arrays that do
    not match the condition (e.g., have a different leading dimension size)
    are left untouched. This is used to extract a single snapshot from a
    batched state or trajectory.

    Parameters
    ----------
    tree : Any
        A PyTree (e.g., :class:`State` or :class:`System`) containing JAX arrays.
    idx : int
        The index along the leading axis to slice.
    current_dim_size : int
        The expected length of the leading axis being sliced. Only arrays with
        `x.shape[0] == current_dim_size` will be sliced.
    """
    return jax.tree_util.tree_map(
        lambda x: (
            x[idx]
            if (
                isinstance(x, jax.Array)
                and x.ndim > 0
                and x.shape[0] == current_dim_size
            )
            else x
        ),
        tree,
    )


def _flatten_leading_dims(tree: Any, num_dims_to_flatten: int) -> Any:
    """
    Flattens the first `num_dims_to_flatten` leading dimensions of each JAX array
    leaf in a PyTree.

    For example, if an array has shape (D1, D2, D3, ..., D_N), and num_dims_to_flatten = 2,
    it will flatten D1 and D2 into a single new leading dimension (D1*D2, D3, ..., D_N).
    """
    if num_dims_to_flatten <= 0:
        return tree

    def flatten_array(arr: jax.Array):
        if not isinstance(arr, jax.Array) or arr.ndim < num_dims_to_flatten:
            return arr  # Cannot flatten if not enough dimensions

        new_shape = (-1,) + arr.shape[num_dims_to_flatten:]
        return arr.reshape(new_shape)

    return jax.tree_util.tree_map(flatten_array, tree)


class VTKBaseWriter(Factory["VTKBaseWriter"], ABC):
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

    This class orchestrates the process of converting `State` and
    `System` PyTrees into VTK files, handling batching, trajectories, and
    dispatching to concrete :class:`VTKBaseWriter` subclasses.

    How leading axes are interpreted for writing
    -------------------------------------------
    JaxDEM's `State` objects support complex data layouts. For the purpose of
    VTK output, `VTKWriter` interprets leading dimensions according to this convention:

    *   **Single snapshot (no leading dimensions):**
        `pos.shape = (N, dim)` for particle properties. Files are written directly
        to the specified `directory`.

    *   **Batched states (one leading dimension):**
        `pos.shape = (B, N, dim)`. The first dimension `B` is interpreted as the
        **batch dimension**. For each batch element, a separate subdirectory
        (e.g., `batch_00000000/`) is created within the main output `directory`.

    *   **Trajectories of a single simulation (one leading dimension, when `trajectory=True`):**
        `pos.shape = (T, N, dim)`. The first dimension `T` is interpreted as the
        **trajectory (time) dimension**. Files for each time step (e.g., `spheres_00000042.vtp`)
        are created directly within the main output `directory`. This mode is activated
        by passing `trajectory=True` to the `save` method when `state.pos.ndim == 3`.

    *   **Trajectories of batched states (multiple leading dimensions):**
        `pos.shape = (B, T_1, T_2, ..., T_k, N, dim)`.
        The **first dimension (`pos.shape[0]`) is interpreted as the batch dimension (`B`)**.
        All subsequent leading dimensions (`T_1, T_2, ..., T_k`) are interpreted as
        **trajectory dimensions**.
        At save time, these trajectory dimensions (`T_1` to `T_k`) are **flattened**
        into a single `T_flat` dimension before being written. For example,
        ` (B, T_1, T_2, N, dim) ` becomes ` (B, T_flat, N, dim) ` internally for writing.
        If trajectory=True, the first leading dimension is also treated as atrajectory and it flattened.

    Requirements on `system`
    ------------------------
    The `System` object's attributes should be either broadcastable to the `state`'s
    leading dimensions (e.g., a scalar `dt` for all batches/frames) or share
    identical leading dimensions with `state` itself. During recursive processing,
    every array leaf of `system` that has a length matching the current leading
    axis is sliced together with the corresponding `state` slice. This ensures
    that each individual writer receives consistent `State` and `System` objects
    for each snapshot.

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
    List of writers to be used when saving data
    """

    directory: str | pathlib.Path = "frames"
    """
    Path to the directory where data should be saved
    """

    binary: bool = True
    """
    Wheter to save data in binary format
    """

    clean: bool = True
    """
    Whether to clear all the contents of directory should be deleted at initialization
    """

    _counter: int = 0
    """
    Internal counter to keep track of the current frame ID
    """

    _pool: cf.ThreadPoolExecutor = field(
        default_factory=cf.ThreadPoolExecutor,
        init=False,
        repr=False,
    )
    """
    Thread pool to be used for saving files in batch.
    """

    def __post_init__(self):
        self.directory = pathlib.Path(self.directory)
        available = list(VTKBaseWriter._registry.keys())
        if self.writers is None:
            self.writers = available
        unknown = [w for w in self.writers if w not in available]
        if unknown:
            raise KeyError(f"Unknown VTK writers {unknown}. Available: {available}")
        # Ensure the directory exists and clean if requested
        if self.directory.exists() and self.clean:
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    #  VTKWriter.save                                                       #
    # --------------------------------------------------------------------- #
    def save(
        self, state: "State", system: "System", *, trajectory: bool = False
    ) -> int:
        """
        Write `state`/`system` to disk.

        Leading-axis convention
        -----------------------
        - snapshot:                (           N,  dim)
        - trajectory only:         (T,         N,  dim)         trajectory=True
        - batch only:              (B,         N,  dim)
        - batch + trajectory dims: (B, T₁ … Tₖ, N,  dim)

        In the last case all T₁…Tₖ are flattened into a single T axis.
        A fresh frame counter (0,1,2,…) is started *inside each batch directory*.
        """

        # ------------------------------------------------------------------ #
        #  Helper – submit the actual writers                                #
        # ------------------------------------------------------------------ #
        def _dispatch(st, sys, frame_id, out_dir):
            st_np, sys_np = _np_tree(st), _np_tree(sys)

            futs = [
                self._pool.submit(
                    VTKBaseWriter._registry[w].write,
                    st_np,
                    sys_np,
                    frame_id,
                    out_dir,
                    self.binary,
                )
                for w in self.writers
            ]
            cf.wait(futs, return_when=cf.ALL_COMPLETED)
            # propagate exceptions
            for f in futs:
                f.result()

        # ------------------------------------------------------------------ #
        #  Case analysis on state.pos.ndim                                   #
        # ------------------------------------------------------------------ #
        ndim = state.pos.ndim
        futures: list[cf.Future] = []

        # ---------- 0) single snapshot ------------------------------------ #
        if ndim == 2:
            futures.append(
                self._pool.submit(_dispatch, state, system, 0, self.directory)
            )

        # ---------- 1) one leading axis ----------------------------------- #
        elif ndim == 3:
            L = state.pos.shape[0]

            if trajectory:
                # trajectory only  (T, N, dim)
                for t in range(L):
                    st = _slice_along_lead(state, t, L)
                    sy = _slice_along_lead(system, t, L)
                    futures.append(
                        self._pool.submit(_dispatch, st, sy, t, self.directory)
                    )

            else:
                # batches only  (B, N, dim)
                for b in range(L):
                    batch_dir = self.directory / f"batch_{b:08d}"
                    batch_dir.mkdir(parents=True, exist_ok=True)

                    st = _slice_along_lead(state, b, L)
                    sy = _slice_along_lead(system, b, L)
                    futures.append(self._pool.submit(_dispatch, st, sy, 0, batch_dir))

        # ---------- 2) ≥ 2 leading axes ----------------------------------- #
        else:
            if trajectory:
                # flatten *all* leading dims
                flat_state = _flatten_leading_dims(state, ndim - 2)  # (T, N, dim)
                flat_system = _flatten_leading_dims(system, ndim - 2)
                T = flat_state.pos.shape[0]

                for t in range(T):
                    st = _slice_along_lead(flat_state, t, T)
                    sy = _slice_along_lead(flat_system, t, T)
                    futures.append(
                        self._pool.submit(_dispatch, st, sy, t, self.directory)
                    )

            else:
                # keep batch dim, flatten the rest
                B = state.pos.shape[0]
                flat_state = _flatten_leading_dims(state, ndim - 3)  # (B, T, N, dim)
                flat_system = _flatten_leading_dims(system, ndim - 3)
                T = flat_state.pos.shape[1]

                for b in range(B):
                    batch_dir = self.directory / f"batch_{b:08d}"
                    batch_dir.mkdir(parents=True, exist_ok=True)

                    st_b = _slice_along_lead(flat_state, b, B)  # (T, …)
                    sy_b = _slice_along_lead(flat_system, b, B)

                    for t in range(T):
                        st = _slice_along_lead(st_b, t, T)
                        sy = _slice_along_lead(sy_b, t, T)
                        futures.append(
                            self._pool.submit(_dispatch, st, sy, t, batch_dir)
                        )

        # ------------------------------------------------------------------ #
        #  Wait & propagate errors                                           #
        # ------------------------------------------------------------------ #
        cf.wait(futures, return_when=cf.ALL_COMPLETED)
        for f in futures:
            f.result()

        return 0  # global counter no longer used for file names


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
        filename = directory / f"spheres_{counter:08d}.vtp"

        # state is already _np_tree-converted by _dispatch
        st = state
        pos = _pad2d(st.pos)  # Ensure positions are 3D for VTK
        n = pos.shape[0]  # Number of particles

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(pos))  # Convert numpy array to VTK data
        poly.SetPoints(points)

        # Add every per-particle dataclass field as PointData
        for fld in fields(st):  # Iterate through dataclass fields of the State
            name = fld.name
            if name == "pos":
                continue  # Position is handled as points, not a separate array

            arr = getattr(st, name)  # Get the numpy array for the field
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
        filename = directory / f"domain_{counter:08d}.vtp"

        sys_np = system  # system is already _np_tree-converted by _dispatch
        box = _pad2d(sys_np.domain.box_size)  # Ensure box_size is 3D (X, Y, Z)
        anch = _pad2d(sys_np.domain.anchor)  # Ensure anchor is 3D (X, Y, Z)

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
