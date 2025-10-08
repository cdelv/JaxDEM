# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of the high-level VTKWriter frontend.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import math
import os
import tempfile
import threading
from pathlib import Path
import shutil
import concurrent.futures as cf
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Set, Optional, Sequence
from functools import partial

import numpy as np
import xml.etree.ElementTree as ET

from . import VTKBaseWriter

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


def _is_safe_to_clean(path: Path) -> bool:
    """
    Return True if and only if it is safe to delete the target directory.

    Cleaning is refused when `path` resolves to:
      - the current working directory,
      - any ancestor of the current working directory, or
      - the filesystem root (or drive root on Windows).

    Parameters
    ----------
    path : Path
        Directory to test.

    Returns
    -------
    bool
        True if the path is safe to delete; False otherwise.
    """
    p = path.resolve()
    cwd = Path.cwd().resolve()

    # never nuke CWD, any parent of CWD, or the root/drive
    if p == cwd or p in cwd.parents:
        return False
    if p == Path(p.anchor):  # '/' on POSIX, 'C:\\' on Windows
        return False
    return True


@dataclass(slots=True)
class VTKWriter:
    """
    High-level front end for writing simulation data to VTK files.

    This class orchestrates the conversion of JAX-based :class:`jaxdem.State` and
    :class:`jaxdem.System` pytrees into VTK files, handling batches, trajectories,
    and dispatch to registered :class:`jaxdem.VTKBaseWriter` subclasses.

    How leading axes are interpreted
    --------------------------------
    Let particle positions have shape ``(..., N, dim)``, where ``N`` is the
    number of particles and ``dim`` is 2 or 3. Define ``L = state.pos.ndim - 2``,
    i.e., the number of *leading* axes before ``(N, dim)``.

    - ``L == 0`` — single snapshot
        The input is one frame. It is written directly into
        ``frames/batch_00000000/`` (no batching, no trajectory).

    - ``trajectory=False`` (default)
        All leading axes are treated as **batch** axes (not time). If multiple
        batch axes are present, they are **flattened** into a single batch axis:
        ``(B, N, dim)`` with ``B = prod(shape[:L])``. Each batch ``b`` is written
        as a single snapshot under its own subdirectory
        ``frames/batch_XXXXXXXX/``. No trajectory is implied.

        - Example: ``(B, N, dim)`` → B separate directories with one frame each.
        - Example: ``(B1, B2, N, dim)`` → flatten to ``(B1*B2, N, dim)`` and treat as above.

    - ``trajectory=True``
        The axis given by ``trajectory_axis`` is **swapped to the front (axis 0)**
        and interpreted as **time** ``T``. Any remaining leading axes are batch
        axes. If more than one non-time leading axis exists, they are flattened
        into a single batch axis so the data becomes ``(T, B, N, dim)`` with
        ``B = prod(other leading axes)``.

        - If there is only time (``L == 1``): ``(T, N, dim)`` — a single batch
            directory ``frames/batch_00000000/`` contains a time series with ``T``
            frames.
        - If there is time plus batching (``L >= 2``): ``(T, B, N, dim)`` — each
            batch ``b`` gets its own directory ``frames/batch_XXXXXXXX/`` containing
            a time series (``T`` frames) for that batch.

    After these swaps/reshapes, dispatch is:
    - ``(N, dim)`` → single snapshot
    - ``(B, N, dim)`` → batches (no time)
    - ``(T, N, dim)`` → single batch with a trajectory
    - ``(T, B, N, dim)`` → per-batch trajectories

    Concrete writers receive per-frame NumPy arrays; leaves in :class:`System`
    are sliced/broadcast consistently with the current frame/batch.
    """

    writers: List[str] = field(default_factory=list)
    """
    A list of strings specifying which registered :class:`VTKBaseWriter`
    subclasses should be used for writing. If `None`, all available
    `VTKBaseWriter` subclasses will be used.
    """

    directory: Path | str = Path("./frames")
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

    save_every: int = 1
    """
    How often to write; writes on every ``save_every``-th call to :meth:`save`.
    """

    max_queue_size: int = 512
    """
    The maximum number of scheduled writes allowed. ``0`` means unbounded.
    """

    max_workers: Optional[int] = None
    """
    Maximum number of worker threads for the internal thread pool.
    """

    _counter: int = 0
    """
    Internal counter for how many times :meth:`save` has been called.
    Initialized to 0.
    """

    _pool: cf.ThreadPoolExecutor = field(
        default_factory=cf.ThreadPoolExecutor, init=False, repr=False
    )
    """
    Internal :class:`concurrent.futures.ThreadPoolExecutor` used for asynchronous
    file writing, allowing I/O operations to run in the background.
    """

    _writer_classes: List = field(default_factory=list)
    """
    Concrete writer classes corresponding to the names in :attr:`writers`,
    resolved from :class:`VTKBaseWriter`'s registry.
    """

    _manifest: Dict = field(default_factory=dict)
    """
    In-memory manifest of written (or scheduled) frames and metadata.
    Structure:
        {batch: {writer: {frame: {frame, time, epoch}, "_pvd_epoch": int}}}
    Used to prevent stale publishes and to build PVD collections.
    """

    _pending_futures: Set[cf.Future] = field(
        default_factory=set, init=False, repr=False
    )
    """
    Futures representing in-flight write tasks. Drained by :meth:`block_until_ready`
    or when the queue is throttled via :attr:`max_queue_size`.
    """

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """
    Internal lock protecting access to shared state such as :attr:`_manifest`
    and :attr:`_pending_futures`.
    """

    def __post_init__(self):
        """
        Validate configuration, clean/create the output directory,
        resolve writer classes from the registry, and start the thread pool.
        """
        self.save_every = int(self.save_every)
        self.max_queue_size = int(self.max_queue_size)
        if self.max_workers:
            self.max_workers = int(self.max_workers)

        self.directory = Path(self.directory)
        available = list(VTKBaseWriter._registry.keys())
        if not self.writers:
            self.writers = available
        unknown = [w for w in self.writers if w not in available]
        if unknown:
            raise KeyError(
                f"Unknown VTK writers {unknown}.  " f"Available: {available}"
            )
        if self.clean and self.directory.exists():
            if _is_safe_to_clean(self.directory):
                shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self._writer_classes = [VTKBaseWriter._registry[name] for name in self.writers]
        self._pool = cf.ThreadPoolExecutor(max_workers=self.max_workers)

    @partial(jax.named_call, name="VTKWriter.close")
    def close(self):
        """
        Flush all pending tasks and shut down the internal thread pool.
        Safe to call multiple times.
        """
        self.block_until_ready()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)

    def __del__(self):
        """
        Destructor to ensure the thread pool is shut down and pending tasks
        have completed before object is garbage-collected.
        """
        try:
            self.close()
        except Exception:
            pass

    @partial(jax.named_call, name="VTKWriter._publish_vtp_if_latest")
    def _publish_vtp_if_latest(
        self,
        batch: str,
        writer: str,
        frame: int,
        epoch: int,
        final_path: Path,
        tmp_path: Path,
    ) -> bool:
        """
        Publish a completed .vtp write if its epoch matches the latest known
        epoch for the given (batch, writer, frame). Otherwise, discard temp.

        Parameters
        ----------
        batch : str
            Batch directory name (e.g., 'batch_00000003').
        writer : str
            Writer key (e.g., 'spheres').
        frame : int
            Frame number (usually system.step_count).
        epoch : int
            Epoch recorded when the task was scheduled.
        final_path : Path
            Destination file path.
        tmp_path : Path
            Temporary file path to be atomically renamed into place.

        Returns
        -------
        bool
            True if the file was published; False if discarded due to staleness.
        """
        current = self._current_epoch_for_vtp(batch, writer, frame)
        if current != epoch:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            return False
        return self._replace_atomic(final_path, tmp_path)

    @partial(jax.named_call, name="VTKWriter._append_manifest")
    def _append_manifest(self, directory: Path, system) -> None:
        """
        Record (or update) the manifest entry for the current frame/time
        for all writers in the given batch directory. Also updates the
        per-writer PVD epoch when the set of frames changes.

        Parameters
        ----------
        directory : Path
            Target batch directory (e.g., frames/batch_00000000).
        system : System
            System snapshot providing `step_count` and `time`.
        """
        frame = int(system.step_count)
        t = float(system.time)
        bkey = directory.name
        with self._lock:
            per_batch = self._manifest.setdefault(bkey, {})
            for name in self.writers:
                per_writer = per_batch.setdefault(name, {})
                before = set(k for k in per_writer.keys() if isinstance(k, int))
                per_writer[frame] = {
                    "frame": frame,
                    "time": t,
                    "epoch": self._counter,
                }
                after = before | {frame}
                if after != before:
                    per_writer["_pvd_epoch"] = self._counter

    @partial(jax.named_call, name="VTKWriter._append_manifest_batch")
    def _append_manifest_batch(
        self, directory: Path, systems: "Sequence[System]"
    ) -> None:
        """
        Record manifest entries for many frames under one lock.
        Each System must have step_count and time populated.
        """
        bkey = directory.name
        with self._lock:
            per_batch = self._manifest.setdefault(bkey, {})
            for name in self.writers:
                per_writer = per_batch.setdefault(name, {})
                before = {k for k in per_writer.keys() if isinstance(k, int)}
                for sys in systems:
                    f = int(sys.step_count)
                    per_writer[f] = {
                        "frame": f,
                        "time": float(sys.time),
                        "epoch": self._counter,
                    }
                after = {k for k in per_writer.keys() if isinstance(k, int)}
                if after != before:
                    per_writer["_pvd_epoch"] = self._counter

    @partial(jax.named_call, name="VTKWriter._current_epoch_for_vtp")
    def _current_epoch_for_vtp(self, batch: str, writer: str, frame: int) -> int:
        """
        Get the current (latest) epoch recorded for a specific VTP frame.

        Parameters
        ----------
        batch : str
            Batch directory name.
        writer : str
            Writer key.
        frame : int
            Frame number.

        Returns
        -------
        int
            Epoch value, or None if unknown.
        """
        with self._lock:
            return (
                self._manifest.get(batch, {})
                .get(writer, {})
                .get(frame, {})
                .get("epoch", None)
            )

    @partial(jax.named_call, name="VTKWriter._current_epoch_for_pvd")
    def _current_epoch_for_pvd(self, batch: str, writer: str) -> int:
        """
        Get the current (latest) epoch recorded for a writer's PVD collection.

        Parameters
        ----------
        batch : str
            Batch directory name.
        writer : str
            Writer key.

        Returns
        -------
        int
            Epoch value for the PVD, or None if unknown.
        """
        with self._lock:
            return self._manifest.get(batch, {}).get(writer, {}).get("_pvd_epoch", None)

    @staticmethod
    @partial(jax.named_call, name="VTKWriter._replace_atomic")
    def _replace_atomic(final_path: Path, tmp_path: Path) -> bool:
        """
        Atomically replace `final_path` with `tmp_path`.

        On success, returns True. On failure, attempts to delete the temporary
        file and re-raises the exception.

        Parameters
        ----------
        final_path : Path
            Destination path to replace.
        tmp_path : Path
            Temporary file to move into place.

        Returns
        -------
        bool
            True if replaced successfully.

        Raises
        ------
        Exception
            Any exception raised by `os.replace` after temporary cleanup.
        """
        # Optional: add retry around PermissionError on Windows if needed
        try:
            os.replace(os.fspath(tmp_path), os.fspath(final_path))
            return True
        except Exception:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise

    @partial(jax.named_call, name="VTKWriter._publish_pvd_if_latest")
    def _publish_pvd_if_latest(
        self,
        batch: str,
        writer: str,
        epoch: int,
        final_path: Path,
        tmp_path: Path,
    ) -> bool:
        """
        Publish a completed .pvd write if its epoch matches the latest known
        epoch for the given (batch, writer). Otherwise, discard temp.

        Parameters
        ----------
        batch : str
            Batch directory name.
        writer : str
            Writer key.
        epoch : int
            Epoch recorded when the task was scheduled.
        final_path : Path
            Destination .pvd file path.
        tmp_path : Path
            Temporary .pvd file path to be atomically renamed into place.

        Returns
        -------
        bool
            True if the file was published; False if discarded due to staleness.
        """
        current = self._current_epoch_for_pvd(batch, writer)
        if current != epoch:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            return False
        return self._replace_atomic(final_path, tmp_path)

    @partial(jax.named_call, name="VTKWriter.block_until_ready")
    def block_until_ready(self):
        """
        Wait until all scheduled writer tasks complete.

        This will wait for all pending futures, propagate exceptions (if any),
        and clear the pending set.
        """
        if self._pending_futures:
            cf.wait(self._pending_futures)
            for f in self._pending_futures:
                f.result()
            self._pending_futures.clear()

    @partial(jax.named_call, name="VTKWriter.save")
    def save(
        self,
        state: "State",
        system: "System",
        *,
        trajectory: bool = False,
        trajectory_axis: int = 0,
        batch0: int = 0,
    ):
        """
        Schedule writing of a :class:`jaxdem.State` / :class:`jaxdem.System` pair to VTK files.

        This public entry point interprets leading axes (batch vs. trajectory),
        performs any required axis swapping and flattening, and then writes the
        resulting frames using the registered writers. It also creates per-batch
        ParaView ``.pvd`` collections referencing the generated files.

        Parameters
        ----------
        state : State
            The simulation :class:`jaxdem.State` object to be saved. Its array leaves
            must end with ``(N, dim)``.
        system : System
            The :class:`jaxdem.System` object corresponding to `state`. Leading axes
            must be compatible (or broadcastable) with those of `state`.
        trajectory : bool, optional
            If ``True``, interpret ``trajectory_axis`` as time and write a
            trajectory; if ``False``, interpret the leading axis as batch.
        trajectory_axis : int, optional
            The axis in `state`/`system` to treat as the trajectory (time) axis
            when ``trajectory=True``. This axis is swapped to the front prior
            to writing.
        batch0 : in
            Initial value of batch from where to start counting the batches.
        """
        if self._counter % self.save_every != 0:
            self._counter += 1
            return

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

        state = jax.tree_util.tree_map(
            lambda x: x
            if isinstance(x, np.ndarray) and x.flags["C_CONTIGUOUS"]
            else np.asarray(x, order="C"),
            state,
        )
        system = jax.tree_util.tree_map(
            lambda x: x
            if isinstance(x, np.ndarray) and x.flags["C_CONTIGUOUS"]
            else np.asarray(x, order="C"),
            system,
        )

        match state.pos.ndim:
            case 2:
                directory = self.directory / Path(f"batch_{batch0:08d}")
                self._append_manifest(directory, system)
                self._schedule_frame_writes(state, system, directory)
            case 3:
                if trajectory:
                    T, _, _ = state.pos.shape
                    directory = self.directory / Path(f"batch_{batch0:08d}")
                    sys_list = [
                        jax.tree_util.tree_map(lambda x, i=i: x[i], system)
                        for i in range(T)
                    ]
                    self._append_manifest_batch(directory, sys_list)
                    for i in range(T):
                        st = jax.tree_util.tree_map(lambda x: x[i], state)
                        sys = sys_list[i]
                        self._schedule_frame_writes(st, sys, directory)
                else:
                    B, _, _ = state.pos.shape
                    for j in range(B):
                        st = jax.tree_util.tree_map(lambda x, j=j: x[j], state)
                        sys = jax.tree_util.tree_map(lambda x, j=j: x[j], system)
                        directory = self.directory / Path(f"batch_{batch0+j:08d}")
                        self._append_manifest(directory, sys)
                        self._schedule_frame_writes(st, sys, directory)
            case 4:
                T, B, _, _ = state.pos.shape
                for j in range(B):
                    directory = self.directory / Path(f"batch_{batch0+j:08d}")
                    sys_list = [
                        jax.tree_util.tree_map(lambda x, i=i, j=j: x[i, j], system)
                        for i in range(T)
                    ]
                    self._append_manifest_batch(directory, sys_list)
                    for i in range(T):
                        st = jax.tree_util.tree_map(lambda x, i=i, j=j: x[i, j], state)
                        sys = sys_list[i]
                        self._schedule_frame_writes(st, sys, directory)

        with self._lock:
            manifest_snapshot = {
                batch: {
                    writer: {
                        "_pvd_epoch": info.get("_pvd_epoch", None),
                        "frames": sorted(k for k in info.keys() if isinstance(k, int)),
                    }
                    for writer, info in writers.items()
                }
                for batch, writers in self._manifest.items()
            }

        for batch, writers in manifest_snapshot.items():
            for writer, info in writers.items():
                if (
                    self.max_queue_size
                    and len(self._pending_futures) >= self.max_queue_size
                ):
                    _, self._pending_futures = cf.wait(
                        self._pending_futures, return_when=cf.FIRST_COMPLETED
                    )
                self._pending_futures.add(
                    self._pool.submit(
                        self._build_pvd_one,
                        batch,
                        writer,
                        info["frames"],
                        info["_pvd_epoch"],
                    )
                )

        self._counter += 1

    @partial(jax.named_call, name="VTKWriter._schedule_frame_writes")
    def _schedule_frame_writes(self, state_np, system_np, directory: Path):
        """
        Queue per-writer tasks for a single frame (non-blocking).

        Parameters
        ----------
        state : State
            State snapshot (arrays converted to NumPy for VTK).
        system : System
            System snapshot (arrays converted to NumPy for VTK).
        directory : Path
            Directory where the per-writer frame files will be written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        batch = directory.name
        frame = int(system_np.step_count)

        for cls, writer_name in zip(self._writer_classes, self.writers):
            final_path = (
                directory / f"{writer_name}_{int(system_np.step_count):08d}.vtp"
            )
            epoch = self._current_epoch_for_vtp(batch, writer_name, frame)
            d = final_path.parent
            base = final_path.name
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"temp_{base}", suffix=".tmp", dir=os.fspath(d)
            )
            os.close(fd)  # let VTK open the file by path

            def write_one_file(
                tmp_path: Path = Path(tmp_path),
                final_path: Path = Path(final_path),
                state=state_np,
                system=system_np,
                binary: bool = self.binary,
                batch: str = batch,
                writer_name: str = writer_name,
                frame: int = frame,
                epoch: int = epoch,
                cls=cls,
            ) -> bool:
                try:
                    cls.write(state, system, tmp_path, binary)
                    return self._publish_vtp_if_latest(
                        batch, writer_name, frame, epoch, final_path, tmp_path
                    )
                except Exception:
                    try:
                        os.remove(tmp_path)
                    except FileNotFoundError:
                        pass
                    raise

            if (
                self.max_queue_size
                and len(self._pending_futures) >= self.max_queue_size
            ):
                _, self._pending_futures = cf.wait(
                    self._pending_futures, return_when=cf.FIRST_COMPLETED
                )
            self._pending_futures.add(self._pool.submit(write_one_file))

    @partial(jax.named_call, name="VTKWriter._build_pvd_one")
    def _build_pvd_one(
        self,
        batch: str,
        writer: str,
        frames: List[int],
        epoch: int,
        time_format: str = ".12g",
    ) -> None:
        """
        Build a ParaView ``.pvd`` time-collection file for one (batch, writer).

        Uses the internal manifest to list frames in sorted order and to
        populate timesteps from recorded simulation times.

        Parameters
        ----------
        batch : str
            Batch directory name (e.g., 'batch_00000000').
        writer : str
            Writer key (e.g., 'spheres').
        frames : List[int]
            Sorted frame indices to include.
        epoch : int
            Latest epoch for this writer's PVD; used to avoid stale publishes.
        time_format : str
            Python format specifier applied to the timestep when writing the XML
            attribute.

        Returns
        -------
        None
        """

        vtk_file_element = ET.Element(
            "VTKFile",
            type="Collection",
            version="0.1",
            byte_order="LittleEndian",
        )
        collection_element = ET.SubElement(vtk_file_element, "Collection")

        with self._lock:
            by_writer = self._manifest.get(batch, {}).get(writer, {})
            for frame in frames:
                t = by_writer.get(frame, {}).get("time", 0.0)
                name = f"{writer}_{frame:08d}.vtp"
                ET.SubElement(
                    collection_element,
                    "DataSet",
                    timestep=format(t, time_format),
                    file=f"{batch}/{name}",
                )

        pvd_path = self.directory / f"{batch}_{writer}.pvd"
        d = pvd_path.parent
        base = pvd_path.name
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"temp_{base}", suffix=".tmp", dir=os.fspath(d)
        )
        os.close(fd)

        ET.ElementTree(vtk_file_element).write(
            tmp_path, encoding="utf-8", xml_declaration=True
        )

        self._publish_pvd_if_latest(batch, writer, epoch, pvd_path, Path(tmp_path))


__all__ = ["VTKWriter"]
