# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import pathlib
import shutil
import concurrent.futures as cf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import List, Optional

import jax
import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

def np_tree(tree):
    """
    Recursively convert every JAX array leaf of *tree* to a host
    `numpy.ndarray`.  All other leaves are returned unchanged.
    """
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)) if isinstance(x, jax.Array) else x,
        tree,
    )

def pad2d(arr: np.ndarray) -> np.ndarray:
    """
    If *arr* is 2-D in the geometric sense (last axis length == 2),
    append a zero z-component so VTK can treat it as 3-D.  Otherwise
    return *arr* unchanged.
    """
    if arr.shape[-1] == 2:
        return np.pad(arr, (*[(0, 0)] * (arr.ndim - 1), (0, 1)), "constant")
    return arr

def _slice_along_lead(tree, idx: int, lead: int):
    """
    Return a copy of *tree* where every JAX array that has `shape[0] ==
    lead` is replaced by its slice `x[idx]`.  Scalars and arrays that do
    not match the condition are left untouched.
    """
    return jax.tree_util.tree_map(
        lambda x: x[idx] if (isinstance(x, jax.Array) and x.ndim > 0 and x.shape[0] == lead) else x,
        tree,
    )

class VTKBaseWriter(Factory["VTKBaseWriter"], ABC):
    """Stateless helper that writes exactly one VTK file."""
    @classmethod
    @abstractmethod
    def write(cls, state: "State", system: "System", counter: int, directory: pathlib.Path | str, binary: bool) -> int:
        """
        Write the given snapshot out as a VTK PolyData file.

        The method must be pure: it is allowed to touch only the file
        system and must *not* modify any input argument in place.

        Parameters
        ----------
        state / system :
            The snapshot to be written.
        counter :
            Global monotonically increasing identifier to be embedded
            in the file name.
        directory :
            Target directory; is guaranteed to exist.
        binary :
            Write the VTK file in binary (`True`) or ASCII (`False`)
            mode.

        Returns
        -------
        int
            The counter value *counter + 1*.  Returning the incremented
            value makes it straightforward to compute a per-writer
            maximum when different writer implementations run in
            parallel.
        """
        raise NotImplementedError


@dataclass(slots=True)
class VTKWriter:
    """
    High-level front-end that turns a (`State`, `System`) pair into one
    or many VTK files using the concrete subclasses registered via
    `@VTKBaseWriter.register`.

    How leading axes are interpreted
    --------------------------------
    Let `pos.shape == (..., N, dim)` where the last two axes are the
    particle index *N* and the spatial dimension *dim ∈ {2, 3}*.
    Denote the number of remaining *leading* axes by `L`.

        L = state.pos.ndim – 2

    1. L == 0                                           → single snapshot
    2. L ≥ 1 and `trajectory is False` (default)
         • axis 0 is treated as **batch**
         • axes 1 … L are treated as **trajectory**
    3. L ≥ 1 and `trajectory is True`
         • **all** leading axes are treated as trajectory

    Inside each batch directory every trajectory step becomes one pair
    of files per concrete writer (e.g. `spheres_00000042.vtp`,
    `domain_00000042.vtp`, …).

    Requirements on *system*
    ------------------------
    `System` may share leading axes with `state` or be broadcastable
    (i.e. scalars).  
    During the recursion every array leaf of `system` that has length
    `lead` along the current leading axis is sliced together with the
    corresponding `state` slice, so each writer receives matching
    per-snapshot values.

    Parameters
    ----------
    writers : list[str] | None, default = None
        Subset of registered writer keys to use.  `None` → all writers.
    directory : str | pathlib.Path, default = "frames"
        Output directory (created if missing).
    binary : bool, default = True
        Write VTK files in binary (smaller, faster) or ASCII format.
    clean : bool, default = True
        If `True`, the contents of *directory* are removed **once**, the
        first time `save()` is called on the instance.

    Notes
    -----
    • All I/O work is executed in a single `ThreadPoolExecutor`
      belonging to the `VTKWriter` instance.  
    • A global counter is incremented *before* a snapshot is submitted,
      guaranteeing unique file names even when threads finish out of
      order.
    • `VTKWriter` itself is **not** a JAX PyTree and therefore never
      appears inside `jit`/`vmap` transforms; it lives purely on the
      Python side.
    """
    writers:   Optional[List[str]]     = None
    directory: str | pathlib.Path      = "frames"
    binary:    bool                    = True
    clean:     bool                    = True

    # internal state (not part of the public interface)
    _counter: int                      = 0
    _pool: cf.ThreadPoolExecutor       = field(
        default_factory=cf.ThreadPoolExecutor,
        init=False,
        repr=False,
    )

    def __post_init__(self):
        self.directory = pathlib.Path(self.directory)
        available = list(VTKBaseWriter._registry.keys())
        if self.writers is None:
            self.writers = available
        unknown = [w for w in self.writers if w not in available]
        if unknown:
            raise KeyError(
                f"Unknown VTK writers {unknown}.  "
                f"Available: {available}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.clean:
            shutil.rmtree(self.directory)
            self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, state: "State", system: "System", *, trajectory: bool = False) -> int:
        """
        Write *state* / *system*.

        The leading axis is interpreted as

        - **time**      if `trajectory=True`, or
        - **batch**     if `trajectory=False`.

        For arrays with more than three spatial axes
        (`state.pos.ndim > 3`) the leading axis is *always* interpreted
        as time, because deeper nesting of batches is rarely meaningful
        whereas “trajectory of trajectories” is common in e.g. Monte
        Carlo simulations.

        All snapshots are scheduled immediately and executed in the
        background thread pool; the method blocks until every writer has
        finished so once `save` returns all files are guaranteed to be
        on disk.

        Returns
        -------
        int
            The new value of the global counter after the snapshots have
            been written.
        """
        futures = self._save_recursive(state, system, self.directory, trajectory)
        cf.wait(futures, return_when=cf.ALL_COMPLETED)
        for f in futures:  # propagate exceptions
            f.result()
        return self._counter

    
    def _save_recursive(self, state: "State", system: "System", directory: pathlib.Path | str, trajectory: bool) -> List[cf.Future]:
        """
        Recursive walker over the leading axes.

        The *first* leading axis is interpreted as
          - batch      if `trajectory` is False
          - trajectory if `trajectory` is True

        Every *additional* leading axis is always treated as trajectory.
        """
        directory = pathlib.Path(directory)
        rank = state.pos.ndim

        # ---------- base case: plain snapshot --------------------------- #
        if rank <= 2:
            return self._dispatch(state, system, directory)

        lead = state.pos.shape[0]            # length of the first leading axis
        is_time_axis = trajectory            # rule described above

        futures: List[cf.Future] = []
        if is_time_axis:                     # -------- trajectory ----------
            for t in range(lead):
                st_t  = _slice_along_lead(state,  t, lead)
                sys_t = _slice_along_lead(system, t, lead)
                futures += self._save_recursive(st_t, sys_t, directory, trajectory=True)
        else:                                # -------- batch ---------------
            for b in range(lead):
                subdir = directory / f"batch_{b:08d}"
                subdir.mkdir(exist_ok=True)
                st_b  = _slice_along_lead(state,  b, lead)
                sys_b = _slice_along_lead(system, b, lead)
                futures += self._save_recursive(st_b, sys_b, subdir, trajectory=True)
        return futures

    # -----------------------------------------------------------------
    def _dispatch(self, state: "State", system: "System", directory: pathlib.Path | str) -> List[cf.Future]:
        """
        Submit one write job per concrete writer and return the list of
        `Future`s.  The global counter is incremented *immediately* so
        every snapshot gets a unique identifier regardless of the order
        in which the threads finish.
        """
        directory = pathlib.Path(directory)
        counter_id = self._counter
        self._counter += 1

        futures: List[cf.Future] = []
        for name in self.writers:
            writer_cls = VTKBaseWriter._registry[name]
            futures.append(
                self._pool.submit(
                    writer_cls.write,
                    state,
                    system,
                    counter=counter_id,
                    directory=directory,
                    binary=self.binary,
                )
            )
        return futures

@VTKBaseWriter.register("spheres")
class SpheresWriter(VTKBaseWriter):
    """
    Write every particle centre as a point and attach each per-particle
    `State` field as PointData.  Scalars become one-component arrays,
    planar vectors are padded to three components, higher-rank fields
    are ignored.
    """
    @classmethod
    def write(cls, state: "State", system: "System", counter: int, directory: pathlib.Path | str, binary: bool) -> int:
        directory = pathlib.Path(directory)
        filename = directory / f"spheres_{counter:08d}.vtp"

        st  = np_tree(state)
        pos = pad2d(st.pos)
        n   = pos.shape[0]

        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(vtk_np.numpy_to_vtk(pos))
        poly.SetPoints(points)

        # add every per-particle dataclass field
        for fld in fields(st):
            name = fld.name
            if name == "pos":
                continue

            arr = getattr(st, name)
            if not isinstance(arr, np.ndarray) or arr.shape[0] != n:
                continue  # skip non-array or non-per-particle fields

            if arr.ndim == 1:                # scalar
                data = arr.reshape(n, 1)
            elif arr.ndim == 2 and arr.shape[1] in (2, 3):
                data = pad2d(arr)            # planar or spatial vector
            else:
                continue                     # unsupported shape

            vtk_arr = vtk_np.numpy_to_vtk(data, deep=1)
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
    Write the simulation domain as an axis-aligned cuboid (3-D) or
    rectangle (2-D) centred at `anchor + box_size / 2`.
    """

    @classmethod
    def write(cls, state: "State", system: "System", counter: int, directory: pathlib.Path | str, binary: bool) -> int:
        directory = pathlib.Path(directory)
        filename = directory / f"domain_{counter:08d}.vtp"

        sys_np = np_tree(system)
        box  = pad2d(sys_np.domain.box_size)
        anch = pad2d(sys_np.domain.anchor)

        cube = vtk.vtkCubeSource()
        cube.SetXLength(box[0])
        cube.SetYLength(box[1])
        cube.SetZLength(box[2])
        cube.SetCenter(*(anch + 0.5 * box))
        cube.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(cube.GetOutput())
        writer.SetDataModeToBinary() if binary else writer.SetDataModeToAscii()
        writer.Write()
        return counter + 1