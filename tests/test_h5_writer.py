"""Asynchronous HDF5 snapshots preserve simulation data and restart behavior."""

import hashlib
import json
import threading

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils import h5
from jaxdem.writers.async_base import AsyncWriterError
from jaxdem.writers import h5_writer


def _ballistic(dim=2):
    state = jd.State.create(
        pos=jnp.array([[-0.2, 0.3, 0.4], [1.1, 1.6, 0.7]])[:, :dim],
        vel=jnp.array([[0.2, 0.3, 0.1], [-0.1, 0.2, -0.3]])[:, :dim],
        rad=jnp.array([0.1, 0.2]),
        mass=jnp.array([2.0, 3.0]),
    )
    system = jd.System.create(
        state=state,
        collider_type="",
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 2.0)},
        dt=0.125,
    )
    return jd.System.initialize(state, system)


def _assert_equal(expected, actual):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("x64", [False, True])
def test_snapshot_keeps_exact_arrays_and_continues(tmp_path, dim, x64):
    with jax.enable_x64(x64):
        state, system = _ballistic(dim)
        with jd.H5Writer(directory=tmp_path, max_workers=2) as writer:
            writer.save(state, system)
            writer.block_until_ready()
            path = tmp_path / "snapshot_00000000.h5"
            with h5py.File(path) as file:
                manifest = json.loads(file.attrs["jaxdem_serialization_manifest"])
                assert manifest["version"] == 3
                assert manifest["storage"] == "hdf5"
                np.testing.assert_array_equal(file["root/0/pos_c"], state.pos_c)
                assert file["root/0/pos_c"].dtype == state.pos_c.dtype
            restored = h5.load(str(path))
        _assert_equal((state, system), restored)
        _assert_equal(
            jd.System.step(state, system, n=3), jd.System.step(*restored, n=3)
        )


@pytest.mark.parametrize("layout", ["batch", "trajectory", "batched_trajectory"])
def test_snapshot_preserves_all_leading_axes(tmp_path, layout):
    state, system = _ballistic()
    if layout != "trajectory":
        state = jd.State.stack([state, jd.State.create(pos=state.pos + 0.1)])
        system = jd.System.stack([system, system])
    if layout != "batch":
        _, _, (state, system) = jd.System.trajectory_rollout(
            state, system, strides=jnp.array([0, 1, 2])
        )
    with jd.H5Writer(directory=tmp_path) as writer:
        writer.save(state, system)
    restored = h5.load(str(tmp_path / "snapshot_00000000.h5"))
    _assert_equal((state, system), restored)
    assert len(list(tmp_path.glob("*.h5"))) == 1


def test_save_frequency_and_reopening_preserve_previous_snapshots(tmp_path):
    state = jd.State.create(pos=jnp.ones((1, 2)), vel=jnp.array([[0.5, -0.25]]))
    system = jd.System.create(state=state, collider_type="", dt=0.125)
    with jd.H5Writer(directory=tmp_path, save_every=2, max_workers=3) as writer:
        for _ in range(6):
            writer.save(state, system)
            state, system = jd.System.step(state, system)
    files = sorted(tmp_path.glob("*.h5"))
    assert [path.name for path in files] == [f"snapshot_{i:08d}.h5" for i in range(3)]
    for path, step in zip(files, [0, 2, 4], strict=True):
        loaded_state, loaded_system = h5.load(str(path))
        np.testing.assert_array_equal(
            loaded_state.pos, [[1 + step / 16, 1 - step / 32]]
        )
        assert loaded_system.step_count == step
    saved = {path: path.read_bytes() for path in files}
    with jd.H5Writer(directory=tmp_path) as writer:
        writer.save(state, system)
    for path, contents in saved.items():
        assert path.read_bytes() == contents
    assert h5.load(str(tmp_path / "snapshot_00000003.h5"))[1].step_count == 6
    with jd.H5Writer(directory=tmp_path, clean=True) as writer:
        writer.save(state, system)
    assert [path.name for path in tmp_path.glob("*.h5")] == ["snapshot_00000000.h5"]


@pytest.mark.parametrize("host", [False, True])
def test_save_captures_inputs_before_the_worker_runs(tmp_path, host):
    state, system = _ballistic()
    if host:
        state, system = jax.tree.map(lambda x: np.array(x, copy=True), (state, system))
    expected = jax.tree.map(lambda x: np.array(x, copy=True), (state, system))
    release = threading.Event()

    def wait():
        assert release.wait(timeout=10)

    with jd.H5Writer(directory=tmp_path, max_workers=1) as writer:
        writer.submit(wait)
        try:
            writer.save(state, system)
            assert not list(tmp_path.glob("*.h5"))
            if host:
                state.pos_c[...] += 99
                system.force_manager.gravity[...] += 17
            state.vel = jnp.full_like(state.vel, 42)
            system.time = jnp.array(123.0)
        finally:
            release.set()
    _assert_equal(expected, h5.load(str(tmp_path / "snapshot_00000000.h5")))


@pytest.mark.parametrize("failure", ["serialize", "publish", "collision"])
def test_failure_is_reported_without_publishing_partial_or_replacing_data(
    tmp_path, monkeypatch, failure
):
    state, system = _ballistic()
    writer = jd.H5Writer(directory=tmp_path, max_workers=1)
    path = tmp_path / "snapshot_00000000.h5"
    h5.save((state, system), str(path))
    digest = hashlib.sha256(path.read_bytes()).digest()

    if failure == "serialize":

        def fail_save(obj, filename):
            with h5py.File(filename, "w") as file:
                file.create_dataset("partial", data=[1])
            raise OSError("disk full")

        monkeypatch.setattr(h5, "save", fail_save)
    elif failure == "publish":

        def fail_link(*args):
            raise OSError("cannot publish")

        monkeypatch.setattr(h5_writer.os, "link", fail_link)

    writer.save(state, system)
    with pytest.raises(AsyncWriterError):
        writer.block_until_ready()
    writer.close()
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    _assert_equal((state, system), h5.load(str(path)))


def test_save_after_close_rejects_even_a_skipped_call(tmp_path):
    state, system = _ballistic()
    writer = jd.H5Writer(directory=tmp_path, save_every=2)
    writer.save(state, system)
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.save(state, system)


def test_missing_dependency_does_not_clean_existing_data(tmp_path, monkeypatch):
    marker = tmp_path / "keep.txt"
    marker.write_text("preserve")

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(h5_writer, "import_module", missing)
    with pytest.raises(ImportError, match=r"JaxDEM\[io\]"):
        jd.H5Writer(directory=tmp_path, clean=True)
    assert marker.read_text() == "preserve"
