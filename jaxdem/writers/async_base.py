# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Defines the base infrastructure for asynchronous data writing."""

from __future__ import annotations

import atexit
import logging
import queue
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class BaseAsyncWriter:
    """
    Infrastructure for non-blocking JAX data writing.

    This class provides a pool of background worker threads and a task queue
    to ensure that slow disk I/O operations and device-to-host transfers do not
    block the main simulation loop.
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

    max_workers: int = 8
    """
    The number of background worker threads to use for parallel I/O.
    """

    max_queue_size: int = 512
    """
    Maximum number of pending tasks in the background queue. When the queue
    is full, :meth:`submit` blocks until a worker frees a slot, providing
    backpressure so memory cannot grow without bound when the simulation
    outruns disk I/O. Set to ``0`` for an unbounded queue.
    """

    _queue: queue.Queue[
        tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None
    ] = field(default_factory=queue.Queue, init=False)
    _threads: list[threading.Thread] = field(default_factory=list, init=False)
    _save_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._queue = queue.Queue(maxsize=max(0, int(self.max_queue_size)))
        self._threads = []
        self._save_calls = 0
        self.directory = Path(self.directory)
        self.save_every = int(self.save_every)

        if self.clean and self.directory.exists():
            if self._is_safe_to_clean(self.directory):
                shutil.rmtree(self.directory)

        self.directory.mkdir(parents=True, exist_ok=True)
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)
        atexit.register(self.close)

    def _is_safe_to_clean(self, path: Path) -> bool:
        """
        Return True if and only if it is safe to delete the target directory.

        Cleaning is refused when `path` resolves to:
          - the current working directory,
          - any ancestor of the current working directory, or
          - the filesystem root (or drive root on Windows).
        """
        p = path.resolve()
        cwd = Path.cwd().resolve()

        # never nuke CWD, any parent of CWD, or the root/drive
        if p == cwd or p in cwd.parents:
            return False
        if p == Path(p.anchor):  # '/' on POSIX, 'C:\\' on Windows
            return False
        return True

    def _worker(self) -> None:
        """Internal background worker loop for non-blocking I/O."""
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            func, args, kwargs = item
            try:
                func(*args, **kwargs)
            except Exception:
                _log.exception(
                    "AsyncWriter task %r failed",
                    getattr(func, "__qualname__", func),
                )
            finally:
                self._queue.task_done()

    def _should_save(self) -> bool:
        """
        Implements the ``save_every`` skipping logic.

        Increments the internal call counter and returns True on the first
        call and every ``save_every``-th call thereafter. Call this at the
        top of :meth:`save` in subclasses.
        """
        count = self._save_calls
        self._save_calls = count + 1
        return self.save_every <= 1 or count % self.save_every == 0

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """
        Pushes a task to the background worker queue.

        Blocks when the queue is full (see :attr:`max_queue_size`).
        """
        self._queue.put((func, args, kwargs))

    def close(self) -> None:
        """
        Blocks the main thread until all pending writes are finished and
        shuts down the background threads.
        """
        if self._threads:
            for _ in range(len(self._threads)):
                self._queue.put(None)
            for t in self._threads:
                t.join()
            self._threads = []

    def block_until_ready(self) -> None:
        """
        Waits until all currently pending tasks in the queue are completed.
        """
        self._queue.join()

    def __del__(self) -> None:
        """Ensures the writer is closed before object destruction."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> BaseAsyncWriter:
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()


__all__ = ["BaseAsyncWriter"]
