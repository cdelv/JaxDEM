# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Asynchronous HDF5 snapshots using the existing serialization format."""

from __future__ import annotations

import jax

import os
import tempfile
import threading
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .async_base import BaseAsyncWriter

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass
class H5Writer(BaseAsyncWriter):
    """Write complete State/System pairs in background workers.

    Each selected :meth:`save <jaxdem.writers.h5_writer.H5Writer.save>` call produces ``snapshot_XXXXXXXX.h5`` in
    ``directory``. Files contain the original arrays, including any leading
    batch or trajectory axes, without wrapping or otherwise changing geometry.
    Read them with :func:`jaxdem.utils.h5.load`, which returns ``(state, system)``.

    Numbering follows submission order and continues after existing snapshots
    when reopening a directory. Completed files are published atomically and
    never replace an existing file. Only complete snapshots appear under the
    final names; failed writes remove their temporary files.

    The inherited queue limits, ``save_every``, context management, and
    completion/error reporting work like :class:`VTKWriter <jaxdem.writers.vtk_writer.VTKWriter>`. Call
    :meth:`block_until_ready <jaxdem.writers.async_base.BaseAsyncWriter.block_until_ready>` before reading pending files, or use a context
    manager to wait and close. HDF5 support requires ``JaxDEM[io]``.
    """

    directory: Path = Path("./h5")
    _next_frame: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        """Check the optional dependency before preparing output or workers."""
        try:
            import_module("h5py")
        except ModuleNotFoundError as exc:
            if exc.name == "h5py":
                raise ImportError(
                    "HDF5 output requires pip install 'JaxDEM[io]'."
                ) from exc
            raise
        BaseAsyncWriter.__post_init__(self)
        self._next_frame = 1 + max(
            (
                int(path.stem.removeprefix("snapshot_"))
                for path in self.directory.glob("snapshot_*.h5")
                if path.stem.removeprefix("snapshot_").isdigit()
            ),
            default=-1,
        )

    def save(self, state: State, system: System) -> None:
        """Capture a snapshot and queue its host transfer and HDF5 write.

        JAX buffers are immutable and retained until writing completes.
        Mutable NumPy buffers are copied before submission, and the State and
        System trees are detached from later attribute assignments.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot save to a closed writer")
            if not self._should_save():
                return
            filename = self.directory / f"snapshot_{self._next_frame:08d}.h5"
            self._next_frame += 1

        def capture(array: Any) -> Any:
            if isinstance(array, jax.Array):
                array.copy_to_host_async()
            return array.copy() if isinstance(array, np.ndarray) else array

        snapshot = jax.tree.map(capture, (state, system))
        self.submit(self._write_snapshot, snapshot, filename, jax.config.x64_enabled)

    def _write_snapshot(
        self, snapshot: tuple[State, System], filename: Path, x64: bool
    ) -> None:
        """Write a host snapshot, then publish it without overwriting files."""
        from ..utils import h5

        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{filename.name}.", suffix=".tmp", dir=self.directory
        )
        os.close(descriptor)
        try:
            # Worker threads do not inherit thread-local JAX precision settings.
            with jax.enable_x64(x64):
                h5.save(jax.device_get(snapshot), temporary)
            # A same-directory hard link publishes the closed file atomically
            # and fails if another writer has already claimed this name.
            os.link(temporary, filename)
        finally:
            Path(temporary).unlink(missing_ok=True)


__all__ = ["H5Writer"]
