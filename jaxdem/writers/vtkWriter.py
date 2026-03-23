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

from .async_base import BaseAsyncWriter

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass
class VTKWriter(BaseAsyncWriter):
    """
    High-level front end for writing simulation data to VTK files.

    This class orchestrates the conversion of JAX-based :class:`jaxdem.State` and
    :class:`jaxdem.System` pytrees into VTK files, handling batches, trajectories,
    and dispatch to registered :class:`jaxdem.VTKBaseWriter` subclasses.

    How leading axes are interpreted
    --------------------------------
    Let particle positions have shape ``(..., N, dim)``, where ``N`` is the
    number of particles and ``dim`` is 2 or 3. Define ``L = state.pos_c.ndim - 2``,
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

    writers: list[str] = field(default_factory=list)
    """
    A list of strings specifying which registered :class:`VTKBaseWriter`
    subclasses should be used for writing. If empty, all available
    subclasses will be used.
    """

    binary: bool = True
    """
    If True, VTK files will be written in binary format.
    If False, files will be written in ASCII format.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _manifest: dict[str, dict[str, dict[int, float]]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """
        Validate configuration and resolve writer classes from the registry.
        """
        BaseAsyncWriter.__post_init__(self)
        self._lock = threading.Lock()
        self._manifest = {}
        from . import VTKBaseWriter

        if not self.writers:
            self.writers = list(VTKBaseWriter._registry.keys())

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

        This public entry point interprets leading axes (batch vs. trajectory),
        performs any required axis swapping and flattening, and then pushes the
        data to the background writer queue.

        Parameters
        ----------
        state : State
            The simulation :class:`jaxdem.State` object to be saved.
        system : System
            The :class:`jaxdem.System` object corresponding to `state`.
        trajectory : bool, optional
            If ``True``, interpret ``trajectory_axis`` as time.
        trajectory_axis : int, optional
            The axis in `state`/`system` to treat as the trajectory axis.
        batch0 : int, optional
            The starting batch index for the input data.
        """
        state_cpu = jax.device_get(state)
        system_cpu = jax.device_get(system)

        Ndim = state_cpu.pos_c.ndim
        if trajectory:
            state_cpu = jax.tree.map(
                lambda x: np.swapaxes(x, trajectory_axis, 0), state_cpu
            )
            system_cpu = jax.tree.map(
                lambda x: np.swapaxes(x, trajectory_axis, 0), system_cpu
            )

        if Ndim >= 4:
            L = Ndim - 2

            def reshape_fn(x: Any) -> Any:
                if not hasattr(x, "reshape"):
                    return x
                new_shape = (
                    (x.shape[0], np.prod(x.shape[1:L]), *x.shape[L:])
                    if trajectory
                    else (np.prod(x.shape[:L]), *x.shape[L:])
                )
                return x.reshape(new_shape)

            state_cpu = jax.tree.map(reshape_fn, state_cpu)
            system_cpu = jax.tree.map(reshape_fn, system_cpu)

        Ndim = state_cpu.pos_c.ndim
        dirty_pvds: set[tuple[str, str]] = set()

        if Ndim == 2:
            self._process_frame(state_cpu, system_cpu, batch0, dirty_pvds)
        elif Ndim == 3:
            if trajectory:
                for i in range(state_cpu.pos_c.shape[0]):
                    st = jax.tree.map(lambda x: x[i], state_cpu)
                    sys = jax.tree.map(lambda x: x[i], system_cpu)
                    self._process_frame(st, sys, batch0, dirty_pvds)
            else:
                for j in range(state_cpu.pos_c.shape[0]):
                    st = jax.tree.map(lambda x: x[j], state_cpu)
                    sys = jax.tree.map(lambda x: x[j], system_cpu)
                    self._process_frame(st, sys, batch0 + j, dirty_pvds)
        elif Ndim == 4:
            T, B = state_cpu.pos_c.shape[:2]
            for j in range(B):
                for i in range(T):
                    st = jax.tree.map(lambda x: x[i, j], state_cpu)
                    sys = jax.tree.map(lambda x: x[i, j], system_cpu)
                    self._process_frame(st, sys, batch0 + j, dirty_pvds)

        # Update pdv files based on the manifest
        for batch_str, name in dirty_pvds:
            self.submit(self._update_pvd, batch_str, name)

    def _process_frame(
        self, state: Any, system: Any, batch_idx: int, dirty_pvds: set[tuple[str, str]]
    ) -> None:
        """
        Slices the state/system for a single frame, updates the manifest, and submits tasks.
        """
        from . import VTKBaseWriter

        b_idx = int(batch_idx)
        batch_str = f"batch_{b_idx:08d}"
        directory = self.directory / batch_str
        directory.mkdir(parents=True, exist_ok=True)

        step_count = int(system.step_count)
        sim_time = float(system.time)

        for name in self.writers:
            cls = cast(type[VTKBaseWriter], VTKBaseWriter._registry[name])
            if not cls.is_active(state, system):
                continue

            self._manifest.setdefault(batch_str, {}).setdefault(name, {})[
                step_count
            ] = sim_time
            dirty_pvds.add((batch_str, name))

            filename = directory / f"{name}_{step_count:08d}.vtp"

            # Submit individual file write task
            self.submit(cls.write, state, system, filename, self.binary)

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
