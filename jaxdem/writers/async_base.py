# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Defines the base infrastructure for asynchronous data writing."""

from __future__ import annotations

import atexit
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ._directory import prepare_output_directory

_log = logging.getLogger(__name__)


class AsyncWriterError(RuntimeError):
    """Raised when one or more background writer tasks fail."""

    def __init__(self, failures: tuple[tuple[str, Exception], ...]) -> None:
        self.failures = failures
        details = "; ".join(
            f"{task}: {type(exc).__name__}: {exc}" for task, exc in failures
        )
        super().__init__(
            f"{len(failures)} asynchronous writer task(s) failed: {details}"
        )


@dataclass(slots=True)
class BaseAsyncWriter:
    """
    Infrastructure for non-blocking JAX data writing.

    This class uses a pool of background worker threads and a task queue so
    that slow disk I/O operations and device-to-host transfers do not
    block the main simulation loop.
    """

    directory: Path = Path("./frames")
    """
    The root directory where the writer saves simulation frames.
    """

    save_every: int = 1
    """
    Save frequency. The writer pushes a frame to the queue on the first call
    and on every `save_every`-th call to the :meth:`save` method.
    """

    clean: bool = False
    """
    If True, the writer deletes and recreates `directory` on initialization.
    Safety checks prevent deleting the current working directory or the
    system root.
    """

    max_workers: int = 8
    """
    The number of background worker threads to use for parallel I/O.
    """

    max_queue_size: int = 512
    """
    Maximum number of pending tasks in the background queue. When the queue
    is full, :meth:`submit` blocks until a worker frees a slot. This
    backpressure keeps memory bounded when the simulation outruns disk I/O.
    Set to ``0`` for an unbounded queue.
    """

    _queue: queue.Queue[
        tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None
    ] = field(default_factory=queue.Queue, init=False)
    _threads: list[threading.Thread] = field(default_factory=list, init=False)
    _save_calls: int = field(default=0, init=False)
    _failures: list[tuple[str, Exception]] = field(default_factory=list, init=False)
    _failures_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _lifecycle_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.max_workers = int(self.max_workers)
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.max_queue_size = int(self.max_queue_size)
        if self.max_queue_size < 0:
            raise ValueError("max_queue_size must be non-negative")
        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._threads = []
        self._save_calls = 0
        self._failures = []
        self._failures_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self.save_every = int(self.save_every)
        self.directory = prepare_output_directory(self.directory, clean=self.clean)
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._threads.append(t)
        atexit.register(self.close)

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
            except Exception as exc:
                task = str(getattr(func, "__qualname__", func))
                _log.exception(
                    "AsyncWriter task %r failed",
                    task,
                )
                with self._failures_lock:
                    self._failures.append((task, exc))
            finally:
                self._queue.task_done()

    def _raise_failures(self) -> None:
        """Raise and consume failures recorded by background workers."""
        with self._failures_lock:
            failures = tuple(self._failures)
            self._failures.clear()
        if failures:
            raise AsyncWriterError(failures) from failures[0][1]

    def _should_save(self) -> bool:
        """
        Implements the ``save_every`` skipping logic.

        Increments the internal call counter and returns True on the first
        call and every ``save_every``-th call after it. Call this at the
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
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Cannot submit to a closed writer")
            self._queue.put((func, args, kwargs))

    def close(self) -> None:
        """
        Blocks the main thread until all pending writes are finished and
        shuts down the background threads.

        Raises :class:`AsyncWriterError` after shutdown if any task failed.
        Reported failures are consumed, so a later call does not report the
        same failures again.
        """
        with self._lifecycle_lock:
            if not self._closed:
                self._closed = True
                threads = tuple(self._threads)
                for _ in threads:
                    self._queue.put(None)
            else:
                threads = tuple(self._threads)
        for thread in threads:
            thread.join()
        with self._lifecycle_lock:
            self._threads = []
        self._raise_failures()

    def block_until_ready(self) -> None:
        """
        Waits until all pending tasks in the queue complete.

        Raises :class:`AsyncWriterError` after the queue drains if any task
        failed. Reported failures are consumed, so a later completion boundary
        does not report the same failures again.
        """
        self._queue.join()
        self._raise_failures()

    def __del__(self) -> None:
        """Closes the writer before object destruction."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> BaseAsyncWriter:
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.close()


__all__ = ["AsyncWriterError", "BaseAsyncWriter"]
