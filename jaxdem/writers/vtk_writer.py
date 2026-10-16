# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the high-level VTKWriter frontend."""

from __future__ import annotations

import jax

import numpy as np
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
import xml.etree.ElementTree as ET

from ..factory import _normalize_key
from .async_base import BaseAsyncWriter

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass
class VTKWriter(BaseAsyncWriter):
    """
    High-level front end for writing simulation data to VTK files.

    This class converts JAX-based :class:`jaxdem.State` and
    :class:`jaxdem.System` pytrees into VTK files. It handles batches,
    trajectories, and dispatch to registered :class:`jaxdem.VTKBaseWriter`
    subclasses.

    How leading axes are interpreted
    --------------------------------
    Let particle positions have shape ``(..., N, dim)``, where ``N`` is the
    number of particles and ``dim`` is 2 or 3. Define ``L = state.pos_c.ndim - 2``,
    i.e., the number of *leading* axes before ``(N, dim)``.

    - ``L == 0`` — single snapshot
        The input is one frame. It is written directly into
        ``frames/batch_00000000/`` (no batching, no trajectory).

    - ``trajectory=False`` (default)
        The writer treats all leading axes as **batch** axes (not time). If
        multiple batch axes exist, the writer **flattens** them into a single
        batch axis:
        ``(B, N, dim)`` with ``B = prod(shape[:L])``. The writer writes each
        batch ``b`` as a single snapshot under its own subdirectory
        ``frames/batch_XXXXXXXX/``. No trajectory is implied.

        - Example: ``(B, N, dim)`` → B separate directories with one frame each.
        - Example: ``(B1, B2, N, dim)`` → flatten to ``(B1*B2, N, dim)`` and treat as above.

    - ``trajectory=True``
        The writer treats ``trajectory_axis`` as **time** ``T`` and the other
        leading axes as batch axes, preserving their original order. Multiple
        batch axes are flattened in row-major order. The logical layout is
        ``(T, B, N, dim)`` with ``B = prod(other leading axes)``.

        - If there is only time (``L == 1``): ``(T, N, dim)`` — a single batch
            directory ``frames/batch_00000000/`` contains a time series with ``T``
            frames.
        - If there is time plus batching (``L >= 2``): ``(T, B, N, dim)`` — each
            batch ``b`` gets its own directory ``frames/batch_XXXXXXXX/`` containing
            a time series (``T`` frames) for that batch.

    Equivalent logical layouts are:
    - ``(N, dim)`` → single snapshot
    - ``(B, N, dim)`` → batches (no time)
    - ``(T, N, dim)`` → single batch with a trajectory
    - ``(T, B, N, dim)`` → per-batch trajectories

    Concrete writers receive per-frame NumPy arrays. A single :class:`System`
    can be shared across batch snapshots; otherwise its leading axes must match
    the State. Trajectories require matching System axes to supply each frame's
    time and step count. Mixed per-leaf System layouts are not inferred.
    """

    writers: list[str] = field(default_factory=list)
    """
    Names of the registered :class:`VTKBaseWriter` subclasses to use for
    writing. If empty, use all registered subclasses. Name matching follows
    registry keys: case-insensitive, and spaces, underscores, and hyphens
    are ignored. The spelling given here sets the output file and ``.pvd``
    names.
    """

    binary: bool = True
    """
    If True, write VTK files in binary format.
    If False, write files in ASCII format.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _manifest: dict[str, dict[str, dict[int, float]]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Validate writer names before cleaning output or starting workers."""
        from importlib import import_module

        try:
            import_module("vtk")
        except ModuleNotFoundError as exc:
            if exc.name in {"vtk", "vtkmodules"}:
                raise ImportError(
                    "VTK output requires pip install 'JaxDEM[io]'."
                ) from exc
            raise
        from . import VTKBaseWriter

        if not self.writers:
            self.writers = list(VTKBaseWriter._registry.keys())

        unknown = [
            name
            for name in self.writers
            if _normalize_key(name) not in VTKBaseWriter._registry
        ]
        if unknown:
            raise KeyError(
                f"Unknown VTKBaseWriter name(s) {unknown}. "
                f"Available: {list(VTKBaseWriter._registry)}"
            )

        BaseAsyncWriter.__post_init__(self)
        self._lock = threading.Lock()
        self._manifest = {}

    def save(
        self,
        state: State,
        system: System,
        *,
        trajectory: bool = False,
        trajectory_axis: int = 0,
        batch0: int = 0,
    ) -> None:
        """
        Schedule writing of a :class:`jaxdem.State` / :class:`jaxdem.System` pair to VTK files.

        This public entry point interprets leading batch or trajectory axes,
        wraps each configuration with its domain, and queues the frames for
        background output.

        Parameters
        ----------
        state : State
            The simulation :class:`jaxdem.State` object to save.
        system : System
            The :class:`jaxdem.System` object corresponding to `state`.
        trajectory : bool, optional
            If ``True``, interpret ``trajectory_axis`` as time.
        trajectory_axis : int, optional
            The leading axis in `state`/`system` to treat as time. Negative
            indices count from the last leading axis, before ``(N, dim)``.
        batch0 : int, optional
            The starting batch index for the input data.
        """
        if not self._should_save():
            return

        leading = state.pos_c.shape[:-2]
        shared = system.dt.ndim == 0
        if (not shared and system.dt.shape != leading) or (trajectory and shared):
            raise ValueError(
                "VTK output requires matching State and System leading axes; batch snapshots may share one System."
            )
        shift = system.domain.shift
        for _ in leading:
            shift = jax.vmap(
                shift,
                in_axes=(0, None if shared else 0),
                out_axes=(0, None if shared else 0),
            )
        state, system = shift(state, system)

        # Kick off non-blocking device->host transfers; the blocking
        # `jax.device_get` happens later inside a background worker thread.
        def _start_copy(x: Any) -> Any:
            if hasattr(x, "copy_to_host_async"):
                x.copy_to_host_async()
            return x

        jax.tree.map(_start_copy, (state, system))
        self.submit(
            self._write_frames, state, system, trajectory, trajectory_axis, batch0
        )

    def _write_frames(
        self,
        state: State,
        system: System,
        trajectory: bool,
        trajectory_axis: int,
        batch0: int,
    ) -> None:
        """
        Worker-side entry point: transfers data to the host, slices frames,
        and writes the VTK files and .pvd manifests. Runs on a background
        thread.
        """
        state_cpu = jax.device_get(state)
        system_cpu = jax.device_get(system)

        leading = state_cpu.pos_c.shape[:-2]
        shared = system_cpu.dt.ndim == 0
        if trajectory:
            if not leading or not -len(leading) <= trajectory_axis < len(leading):
                raise ValueError("trajectory_axis must select a leading State axis.")
            axis = trajectory_axis % len(leading)
            batch_shape = leading[:axis] + leading[axis + 1 :]
        else:
            batch_shape = leading

        dirty_pvds: set[tuple[str, str]] = set()
        for batch, batch_index in enumerate(np.ndindex(batch_shape)):
            for frame in range(leading[axis] if trajectory else 1):
                index = list(batch_index)
                if trajectory:
                    index.insert(axis, frame)
                frame_index = tuple(index)
                st = (
                    state_cpu
                    if not frame_index
                    else jax.tree.map(lambda x: x[frame_index], state_cpu)
                )
                sys = (
                    system_cpu
                    if shared
                    else jax.tree.map(lambda x: x[frame_index], system_cpu)
                )
                self._process_frame(st, sys, batch0 + batch, dirty_pvds)

        # Update pvd files based on the manifest
        for batch_str, name in dirty_pvds:
            self._update_pvd(batch_str, name)

    def _process_frame(
        self, state: Any, system: Any, batch_idx: int, dirty_pvds: set[tuple[str, str]]
    ) -> None:
        """
        Slices the state/system for a single frame, updates the manifest, and
        writes the per-writer files. Runs on a background worker thread.
        """
        from . import VTKBaseWriter

        b_idx = int(batch_idx)
        batch_str = f"batch_{b_idx:08d}"
        directory = self.directory / batch_str
        directory.mkdir(parents=True, exist_ok=True)

        step_count = int(system.step_count)
        sim_time = float(system.time)

        for name in self.writers:
            cls = cast(
                type[VTKBaseWriter], VTKBaseWriter._registry[_normalize_key(name)]
            )
            if not cls.is_active(state, system):
                continue

            filename = directory / f"{name}_{step_count:08d}.vtp"
            try:
                cls.write(state, system, filename, self.binary)
            except Exception as exc:
                raise RuntimeError(
                    f"VTK writer {name!r} failed for {filename}: {exc}"
                ) from exc

            if not filename.exists():
                raise RuntimeError(
                    f"VTK writer {name!r} did not create expected file {filename}"
                )

            # Only register the frame in the manifest once the file exists.
            with self._lock:
                self._manifest.setdefault(batch_str, {}).setdefault(name, {})[
                    step_count
                ] = sim_time
            dirty_pvds.add((batch_str, name))

    def _update_pvd(self, batch_str: str, writer_name: str) -> None:
        """
        Updates the .pvd collection file for a given batch and writer.
        """
        with self._lock:
            data = self._manifest[batch_str][writer_name]
            frames = sorted(data.keys())
            root = ET.Element("VTKFile", type="Collection", version="0.1")
            coll = ET.SubElement(root, "Collection")
            for f in frames:
                rel_file = Path(batch_str) / f"{writer_name}_{f:08d}.vtp"
                ET.SubElement(
                    coll,
                    "DataSet",
                    timestep=format(data[f], ".12g"),
                    file=rel_file.as_posix(),
                )
            pvd_path = self.directory / f"{batch_str}_{writer_name}.pvd"
            ET.ElementTree(root).write(pvd_path, encoding="utf-8", xml_declaration=True)
