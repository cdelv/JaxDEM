# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Sphere mode survives persistence and continued dynamics."""

import jax.numpy as jnp

import h5py
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils import h5
from jaxdem.utils.serialization_schema import make_serialization_manifest
from tests.test_checkpoints import _assert_leaves_equal


def _contact_system(
    has_clumps: bool = False, only_spheres: bool | None = None
) -> tuple[jd.State, jd.System]:
    positions = jnp.array([[0.0, -0.25], [0.0, 0.25], [1.15, 0.4]])
    offsets = jnp.zeros_like(positions)
    if has_clumps:
        offsets = offsets.at[:2].set(positions[:2])
        positions = positions.at[:2].set(0.0)
    mode_kw = {} if only_spheres is None else {"only_spheres": only_spheres}
    state = jd.State.create(
        pos=positions,
        pos_p=offsets,
        vel=jnp.array([[0.1, -0.05], [0.1, -0.05], [-0.1, 0.05]]),
        ang_vel=jnp.array([[0.2], [0.2], [-0.1]]),
        rad=jnp.full(3, 0.65),
        mass=jnp.array([2.0, 2.0, 1.5]),
        inertia=jnp.array([[0.4], [0.4], [0.3]]),
        clump_id=jnp.array([0, 0, 1] if has_clumps else [2, 0, 1]),
        **mode_kw,
    )
    material = jd.Material.create("elastic", density=1.0, young=10.0, poisson=0.25)
    system = jd.System.create(
        state=state,
        dt=0.001,
        mat_table=jd.MaterialTable.from_materials([material]),
        force_manager_kw={"gravity": jnp.array([0.0, -0.3])},
    )
    return jd.System.step(state, system, n=2)


def _assert_continuation(
    original: tuple[jd.State, jd.System],
    restored: tuple[jd.State, jd.System],
) -> None:
    assert type(restored[0].only_spheres) is bool
    assert restored[0].only_spheres is original[0].only_spheres
    _assert_leaves_equal(original[0], restored[0], "state")
    _assert_leaves_equal(original[1], restored[1], "system")
    expected_state, expected_system = jd.System.step(*original, n=3)
    actual_state, actual_system = jd.System.step(*restored, n=3)
    assert actual_state.only_spheres is restored[0].only_spheres
    assert not np.array_equal(expected_state.pos, original[0].pos)
    _assert_leaves_equal(
        expected_state, actual_state, "continued state", rtol=2e-6, atol=1e-9
    )
    _assert_leaves_equal(
        expected_system, actual_system, "continued system", rtol=2e-6, atol=1e-9
    )


@pytest.mark.parametrize("has_clumps,only_spheres", [(False, True), (True, False)])
def test_round_trip_retains_sphere_mode_and_continuation(
    tmp_path, has_clumps, only_spheres
):
    original = _contact_system(has_clumps, only_spheres)
    assert original[0].only_spheres is only_spheres
    assert "only_spheres" not in original[1].metadata

    path = str(tmp_path / "state.h5")
    h5.save(original, path)
    with h5py.File(path) as file:
        assert "only_spheres" in file["root/0"]
        assert "only_spheres" not in file["root/1"]
    _assert_continuation(original, h5.load(path))

    if only_spheres:
        state_path = str(tmp_path / "standalone.h5")
        h5.save(original[0], state_path)
        standalone = h5.load(state_path)
        assert standalone.only_spheres is True
        _assert_leaves_equal(original[0], standalone, "standalone state")

    pytest.importorskip("orbax.checkpoint")
    ckpt_dir = tmp_path / "checkpoint"
    with jd.CheckpointWriter(ckpt_dir) as writer:
        writer.save(*original)
    with jd.CheckpointLoader(ckpt_dir) as loader:
        restored = loader.load()
    _assert_continuation(original, restored)


@pytest.mark.parametrize("has_clumps,legacy_facets", [(False, False), (True, True)])
def test_hdf5_schema_three_without_sphere_mode_uses_default(
    tmp_path, has_clumps, legacy_facets
):
    original = _contact_system(has_clumps)
    path = str(tmp_path / "legacy.h5")
    h5.save(original, path)
    with h5py.File(path, "r+") as file:
        del file["root/0/only_spheres"]
        if legacy_facets:
            state_node = file["root/0"]
            del state_node["facet_ids_extra"]
            del state_node["facet_vertices_extra"]
            del state_node.attrs["__facet_incidence_layout__"]
    with pytest.warns(RuntimeWarning) as warnings:
        restored = h5.load(path)
    assert any(
        "missing saved fields" in str(w.message) and "only_spheres" in str(w.message)
        for w in warnings
    )
    _assert_continuation(original, restored)


def test_hdf5_missing_sphere_mode_does_not_allow_other_missing_fields(tmp_path):
    state = jd.State.create(pos=jnp.zeros((2, 2)))
    path = str(tmp_path / "damaged.h5")
    h5.save(state, path)
    with h5py.File(path, "r+") as file:
        del file["root/only_spheres"]
        del file["root/mass"]
    with pytest.raises(RuntimeError, match="missing=.*mass"):
        h5.load(path)


@pytest.mark.parametrize("has_clumps,legacy_facets", [(False, False), (True, True)])
def test_orbax_schema_three_without_sphere_mode_uses_default(
    tmp_path, has_clumps, legacy_facets
):
    ocp = pytest.importorskip("orbax.checkpoint")
    from jaxdem.writers.checkpoints import _checkpoint_tree

    original = state, system = _contact_system(has_clumps)
    state_payload = ocp.tree.serialize_tree(
        _checkpoint_tree(state), keep_empty_nodes=True
    )
    if legacy_facets:
        del state_payload["facet_ids_extra"]
        del state_payload["facet_vertices_extra"]
    metadata = system.metadata
    metadata.update(
        checkpoint_metadata_version=3,
        serialization_manifest=make_serialization_manifest("orbax", "state_system"),
        state_shape=state.shape,
        bond_id_shape=state.bond_id.shape,
    )
    state_metadata = {"shape": state.shape}
    if not legacy_facets:
        state_metadata.update(
            facet_incidence_layout=1,
            facet_ids_extra_shape=state.facet_ids_extra.shape,
        )
    directory = tmp_path / "checkpoint"
    with jd.CheckpointWriter(directory) as writer:
        writer.checkpointer.save(
            int(system.step_count),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state_payload),
                system=ocp.args.StandardSave(_checkpoint_tree(system)),
                state_metadata=ocp.args.JsonSave(state_metadata),
                system_metadata=ocp.args.JsonSave(metadata),
            ),
        )
    with jd.CheckpointLoader(directory) as loader:
        restored = loader.load()
    assert restored[0].only_spheres is False
    _assert_continuation(original, restored)
