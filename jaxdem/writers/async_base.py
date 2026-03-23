# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Defines the base infrastructure for asynchronous data writing."""

from __future__ import annotations

import shutil
import threading
import queue
import atexit
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass(slots=True)
class BaseAsyncWriter:
    """
    Infrastructure for non-blocking JAX data writing.

    This class provides a sequential background worker thread and a task queue
    to ensure that slow disk I/O operations and device-to-host transfers do not
    block the main simulation loop.

    Subclasses must implement the :meth:`write_frame` method to define specific
    file formats (e.g., VTK, HDF5).
    """

    directory: Path = Path("./frames")
    """
    The root directory where simulation frames will be saved.
    """

    save_every: int = 1
    """
    Frequency of saving. A frame is pushed to the queue every `save_every` calls to the :meth:`save` method.
    """

    clean: bool = True
    """
    If True, the `directory` is deleted and recreated upon initialization.
    Basic safety checks are performed to prevent deleting the current
    working directory or the system root.
    """

    _counter: int = field(default=0, init=False)
    _queue: queue.Queue[tuple[int, Any, Any, dict[str, Any]] | None] = field(
        default_factory=queue.Queue, init=False
    )
    _thread: threading.Thread | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._counter = 0
        self._queue = queue.Queue()
        self._thread = None

        self.directory = Path(self.directory)
        self.save_every = int(self.save_every)

        if self.clean and self.directory.exists():
            if self.directory.resolve() != Path.cwd().resolve():
                shutil.rmtree(self.directory)

        self.directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _worker(self) -> None:
        """Internal background worker loop for non-blocking I/O."""
        while True:
            item = self._queue.get()
            if item is None:
                break

            frame_idx, state_cpu, system_cpu, kwargs = item
            try:
                # Subclasses implement the actual disk writing logic here.
                # Data is guaranteed to be CPU-side (NumPy arrays).
                self.write_frame(frame_idx, state_cpu, system_cpu, **kwargs)
            except Exception as e:
                print(f"Writer error at frame {frame_idx}: {e}")
            finally:
                self._queue.task_done()

    def write_frame(
        self, frame_idx: int, state: State, system: System, **kwargs: dict[str, Any]
    ) -> None:
        """
        Implementation-specific write logic.

        Must be overridden by subclasses. Receives CPU-side data.

        Parameters:
        -----------
        frame_idx : int
            The index of the frame being written.
        state : Pytree
            CPU-side snapshot of the simulation state (NumPy arrays).
        system : Pytree
            CPU-side snapshot of the simulation system (NumPy arrays).
        **kwargs : dict
            Additional format-specific parameters passed from :meth:`save`.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Blocks the main thread until all pending writes are finished.
        This method is automatically called at program exit via `atexit`.
        """
        if self._thread and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()
            self._thread = None

    def __del__(self) -> None:
        """Ensures the writer is closed before object destruction."""
        try:
            self.close()
        except Exception as e:
            print(f"Error during writer destruction: {e}")

    def __enter__(self) -> BaseAsyncWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = ["BaseAsyncWriter"]
