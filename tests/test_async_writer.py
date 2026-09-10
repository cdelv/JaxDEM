# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Tests for observable asynchronous writer failures."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

import jaxdem as jdem
from jaxdem.writers import VTKBaseWriter, VTKWriter
from jaxdem.writers.async_base import AsyncWriterError, BaseAsyncWriter


def test_block_until_ready_reports_all_failures_after_draining(tmp_path: Any) -> None:
    writer = BaseAsyncWriter(directory=tmp_path, max_workers=2)
    completed = threading.Event()

    def fail(message: str) -> None:
        raise ValueError(message)

    writer.submit(fail, "first")
    writer.submit(fail, "second")
    writer.submit(completed.set)

    with pytest.raises(AsyncWriterError) as caught:
        writer.block_until_ready()

    assert completed.is_set()
    assert sorted(str(exc) for _, exc in caught.value.failures) == ["first", "second"]
    writer.close()


def test_close_reports_failure_after_clean_shutdown(tmp_path: Any) -> None:
    writer = BaseAsyncWriter(directory=tmp_path, max_workers=2)
    completed = threading.Event()

    def fail() -> None:
        raise OSError("disk full")

    writer.submit(fail)
    writer.submit(completed.set)

    with pytest.raises(AsyncWriterError, match="disk full"):
        writer.close()

    assert completed.is_set()
    assert writer._threads == []
    writer.close()


@pytest.mark.parametrize("failure", ["raise", "missing"])
def test_vtk_write_failure_reaches_completion_boundary(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    class BrokenWriter:
        @classmethod
        def is_active(cls, state: Any, system: Any) -> bool:
            return True

        @classmethod
        def write(
            cls, state: Any, system: Any, filename: Any, binary: bool
        ) -> None:
            if failure == "raise":
                raise OSError("injected write failure")

    monkeypatch.setitem(VTKBaseWriter._registry, "broken", BrokenWriter)
    writer = VTKWriter(directory=tmp_path, writers=["broken"], max_workers=1)
    state = object()
    system = SimpleNamespace(step_count=3, time=0.25)
    writer.submit(writer._process_frame, state, system, 0, set())

    expected = "injected write failure" if failure == "raise" else "did not create"
    with pytest.raises(AsyncWriterError, match=expected):
        writer.close()

    assert writer._threads == []


def test_vtk_public_save_writes_output_and_manifest(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WorkingWriter:
        @classmethod
        def is_active(cls, state: Any, system: Any) -> bool:
            return True

        @classmethod
        def write(
            cls, state: Any, system: Any, filename: Any, binary: bool
        ) -> None:
            filename.write_text("frame")

    monkeypatch.setitem(VTKBaseWriter._registry, "working", WorkingWriter)
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([1.0])
    )
    system = jdem.System.create(state.shape)

    with VTKWriter(directory=tmp_path, writers=["working"], max_workers=1) as writer:
        writer.save(state, system)
        writer.block_until_ready()

    frame = tmp_path / "batch_00000000" / "working_00000000.vtp"
    manifest = tmp_path / "batch_00000000_working.pvd"
    assert frame.read_text() == "frame"
    assert 'file="batch_00000000/working_00000000.vtp"' in manifest.read_text()
