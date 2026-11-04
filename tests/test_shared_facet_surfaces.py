# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared contact surfaces retain physical nodes and survive state operations."""

import jax
import jax.numpy as jnp

from dataclasses import fields, replace
import json

import numpy as np
import pytest

import jaxdem as jdem


def _surface(dim):
    if dim == 2:
        points = jnp.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        faces = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    else:
        points = jnp.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
        )
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    state = jdem.State.create(
        pos=points,
        mass=jnp.array([1.0, 2.0, 3.0, 4.0]),
        rad=jnp.full(4, 0.1),
        species_id=jnp.ones(4, dtype=int),
        bond_id=[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
    )
    return state, faces


def _faces(state):
    ids = np.concatenate((state.facet_id[:, None], state.facet_ids_extra), axis=1)
    vertices = np.concatenate(
        (state.facet_vertices[:, None, :], state.facet_vertices_extra), axis=1
    )
    return {
        int(ids[node, slot]): tuple(vertices[node, slot])
        for node, slot in np.argwhere(ids >= 0)
    }


@pytest.mark.parametrize("dim", [2, 3])
def test_closed_surface_uses_original_nodes_and_preserves_particle_properties(dim):
    original, elements = _surface(dim)
    state = jdem.State.add_contact_facets(original, elements)
    assert state.N == original.N == 4
    assert state.is_valid
    state.validate()
    assert _faces(state) == dict(enumerate(map(tuple, elements)))
    assert state.facet_ids_extra.shape == (4, dim - 1)
    assert original.facet_ids_extra.shape == (4, 0)
    np.testing.assert_array_equal(original.facet_id, -np.ones(4))
    topology_fields = {
        "facet_id",
        "facet_vertices",
        "facet_ids_extra",
        "facet_vertices_extra",
    }
    for entry in fields(original):
        if entry.name not in topology_fields:
            actual = jax.tree.leaves(getattr(state, entry.name))
            expected = jax.tree.leaves(getattr(original, entry.name))
            for value, before in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("dim", [2, 3])
def test_merge_offsets_extra_only_faces_and_pads_ordinary_particles(dim):
    original, elements = _surface(dim)
    surface = jdem.State.add_contact_facets(original, elements)
    # The final face has no first-incidence slot: ID allocation must inspect extras.
    assert int(jnp.max(surface.facet_id)) < len(elements) - 1
    merged = jdem.State.merge(surface, [original, surface])
    merged.validate()
    expected = dict(enumerate(map(tuple, elements)))
    expected.update(
        {i + len(elements): tuple(face + 8) for i, face in enumerate(elements)}
    )
    assert _faces(merged) == expected
    np.testing.assert_array_equal(merged.facet_ids_extra[4:8], -1)
    np.testing.assert_array_equal(merged.facet_vertices_extra[4:8], -1)
    np.testing.assert_array_equal(merged.bond_id[8:], original.bond_id + 8)
    added = jdem.State.add_facet(surface, jnp.eye(dim), rigid=False, thickness=0.1)
    assert int(added.facet_id[-1]) == len(elements)
    added.validate()


@pytest.mark.parametrize("dim", [2, 3])
def test_stack_pads_incidence_capacity_and_supports_jit_vmap(dim):
    original, elements = _surface(dim)
    surface = jdem.State.add_contact_facets(original, elements)
    batched = jdem.State.stack([surface, original])
    assert batched.is_valid
    batched.validate()
    assert batched.facet_vertices_extra.shape == (2, 4, dim - 1, dim)
    restored, plain = jdem.State.unstack(batched)
    assert _faces(restored) == _faces(surface)
    np.testing.assert_array_equal(plain.facet_ids_extra, -1)

    @jax.jit
    def translate(states):
        return jax.vmap(lambda state: replace(state, pos_c=state.pos_c + 0.25))(states)

    translated = translate(batched)
    translated.validate()
    np.testing.assert_allclose(translated.pos, batched.pos + 0.25)
    np.testing.assert_array_equal(
        translated.facet_vertices_extra, batched.facet_vertices_extra
    )
    merged = jdem.State.merge(batched, batched)
    merged.validate()
    assert merged.facet_vertices_extra.shape == (2, 8, dim - 1, dim)


def test_connected_facet_can_reuse_contact_node_and_add_new_flexible_node():
    original = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        mass=jnp.array([2.0, 3.0]),
        rad=jnp.array([0.1, 0.1]),
        bond_id=[[1], [0]],
    )
    first = jdem.State.add_connected_facet(original, [0, 1], rigid=False)
    result = jdem.State.add_connected_facet(
        first, [1, jnp.array([1.0, 1.0])], rigid=False, thickness=0.1, mass=4.0
    )
    result.validate()
    assert result.N == 3
    assert _faces(result) == {0: (0, 1), 1: (1, 2)}
    np.testing.assert_array_equal(result.mass, [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(result.bond_id[:2], original.bond_id)
    np.testing.assert_allclose(result.pos[:2], original.pos)
    assert first.facet_ids_extra.shape == (2, 0)


def test_contact_surface_can_attach_to_a_rigid_clump_without_changing_body():
    rigid = jdem.State.add_clump(
        jdem.State.create(dim=2),
        pos=jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        rad=jnp.full(3, 0.1),
        mass=jnp.full(3, 3.0),
    )
    surface = jdem.State.add_contact_facets(rigid, [[0, 1], [1, 2], [2, 0]])
    surface.validate()
    np.testing.assert_array_equal(surface.clump_id, rigid.clump_id)
    np.testing.assert_array_equal(surface.mass, rigid.mass)
    np.testing.assert_array_equal(surface.pos_c, rigid.pos_c)
    np.testing.assert_array_equal(surface.pos_p, rigid.pos_p)
    np.testing.assert_array_equal(surface.inertia, rigid.inertia)


@pytest.mark.parametrize("dim", [2, 3])
def test_connected_rigid_existing_nodes_preserve_body_properties_and_frame(dim):
    if dim == 2:
        points = jnp.array([[0.0, 0.0], [1.5, 0.2], [0.2, 1.0]])
        faces = [[0, 1], [1, 2]]
        closing_face = [2, 0]
        inertia = jnp.array([0.4])
    else:
        points = jnp.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.2], [0.0, 1.0, 0.3], [0.2, 0.1, 1.2]]
        )
        faces = [[0, 1, 2], [1, 2, 3]]
        closing_face = [2, 0, 3]
        inertia = jnp.array([0.4, 0.7, 0.9])
    rigid = jdem.State.add_clump(
        jdem.State.create(dim=dim),
        pos=points,
        rad=jnp.full(len(points), 0.1),
        mass=jnp.full(len(points), 7.0),
    )
    rigid = replace(
        rigid, inertia=jnp.broadcast_to(inertia, (len(points), len(inertia)))
    )
    first = jdem.State.add_contact_facets(rigid, faces)
    attached = jdem.State.add_connected_facet(first, closing_face, rigid=True)
    attached.validate()
    assert _faces(attached) == dict(enumerate(map(tuple, [*faces, closing_face])))
    for name in (
        "pos_c",
        "pos_p",
        "vel",
        "force",
        "ang_vel",
        "torque",
        "inertia",
        "mass",
        "volume",
        "clump_id",
        "rad",
        "bond_id",
        "fixed",
        "mat_id",
        "species_id",
    ):
        np.testing.assert_array_equal(getattr(attached, name), getattr(first, name))
    np.testing.assert_array_equal(attached.q.w, first.q.w)
    np.testing.assert_array_equal(attached.q.xyz, first.q.xyz)


@pytest.mark.parametrize("dim", [2, 3])
def test_connected_rigid_new_node_still_updates_total_mass_and_center(dim):
    points = jnp.eye(dim)
    rigid = jdem.State.add_facet(
        jdem.State.create(dim=dim), points, rigid=True, thickness=0.1, mass=3.0
    )
    new_point = jnp.full(dim, 1.5)
    specs = [*range(1, dim), new_point]
    result = jdem.State.add_connected_facet(
        rigid, specs, rigid=True, thickness=0.1, mass=2.0
    )
    assert result.is_valid
    assert result.N == dim + 1
    np.testing.assert_array_equal(result.clump_id, 0)
    np.testing.assert_allclose(result.mass, 5.0)
    expected_center = (3.0 * jnp.mean(points, axis=0) + 2.0 * new_point) / 5.0
    tolerance = 2e-6 if result.pos.dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(
        result.pos_c,
        jnp.broadcast_to(expected_center, result.pos_c.shape),
        atol=tolerance,
    )
    np.testing.assert_allclose(result.pos[:dim], points, atol=tolerance)
    np.testing.assert_allclose(result.pos[-1], new_point, atol=tolerance)


@pytest.mark.parametrize(
    "elements, message",
    [
        ([[0, 0]], "distinct"),
        ([[0, 4]], "lie in"),
        ([[-1, 1]], "lie in"),
        ([[0.0, 1.0]], "integers"),
        ([[0, 1, 2]], "shape"),
        ([[0, 1], [1, 0]], "Duplicate"),
    ],
)
def test_contact_surface_rejects_invalid_topology(elements, message):
    original, _ = _surface(2)
    with pytest.raises(ValueError, match=message):
        jdem.State.add_contact_facets(original, elements)
    np.testing.assert_array_equal(original.facet_id, -1)


def test_contact_surface_rejects_duplicate_existing_face_and_incompatible_nodes():
    original, _ = _surface(2)
    first = jdem.State.add_contact_facets(original, [[0, 1]])
    with pytest.raises(ValueError, match="Duplicate"):
        jdem.State.add_contact_facets(first, [[1, 0]])
    with pytest.raises(ValueError, match="same species_id"):
        jdem.State.add_contact_facets(
            replace(original, species_id=jnp.array([0, 1, 1, 1])), [[0, 1]]
        )
    with pytest.raises(ValueError, match="independent nodes"):
        jdem.State.add_contact_facets(
            replace(original, clump_id=jnp.array([0, 0, 1, 2])), [[0, 2]]
        )


def test_validation_detects_missing_shared_incidence():
    original, elements = _surface(2)
    surface = jdem.State.add_contact_facets(original, elements)
    broken = replace(
        surface,
        facet_ids_extra=surface.facet_ids_extra.at[0, 0].set(-1),
        facet_vertices_extra=surface.facet_vertices_extra.at[0, 0].set(-1),
    )
    with pytest.raises(ValueError, match="Every facet vertex"):
        broken.validate()


@pytest.mark.parametrize("shared", [False, True])
def test_orbax_preserves_empty_and_shared_incidence_shapes(tmp_path, shared):
    pytest.importorskip("orbax.checkpoint")
    original, elements = _surface(2)
    state = jdem.State.add_contact_facets(original, elements) if shared else original
    system = jdem.System.create(state=state)
    directory = tmp_path / "checkpoint"
    with jdem.CheckpointWriter(directory) as writer:
        writer.save(state, system)
    with jdem.CheckpointLoader(directory) as loader:
        restored, _ = loader.load()
    restored.validate()
    for before, after in zip(
        jax.tree.leaves(state), jax.tree.leaves(restored), strict=True
    ):
        assert before.dtype == after.dtype
        np.testing.assert_array_equal(after, before)


def test_orbax_loads_legacy_state_and_neighbor_cache_without_new_fields(tmp_path):
    ocp = pytest.importorskip("orbax.checkpoint")
    from jaxdem.utils.serialization_schema import make_serialization_manifest

    state = jdem.State.create(pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]), rad=jnp.ones(2))
    system = jdem.System.create(
        state=state,
        collider_type="NeighborList",
        collider_kw={"cutoff": 5.0, "max_neighbors": 8},
    )
    state, system = jdem.System.initialize(state, system)
    # Recreate the serialized field layout from before shared contact surfaces.
    state_payload = ocp.tree.serialize_tree(state, keep_empty_nodes=True)
    del state_payload["facet_ids_extra"]
    del state_payload["facet_vertices_extra"]
    system_payload = ocp.tree.serialize_tree(
        jax.tree.map(
            lambda x: None if isinstance(x, jax.Array) and x.size == 0 else x, system
        ),
        keep_empty_nodes=True,
    )
    del system_payload["collider"]["facet_id_snapshot"]
    metadata = system.metadata
    metadata.update(
        checkpoint_metadata_version=3,
        serialization_manifest=make_serialization_manifest("orbax", "state_system"),
        state_shape=state.shape,
        bond_id_shape=state.bond_id.shape,
    )
    directory = tmp_path / "legacy_checkpoint"
    with jdem.CheckpointWriter(directory) as writer:
        writer.checkpointer.save(
            int(system.step_count),
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state_payload),
                system=ocp.args.StandardSave(system_payload),
                state_metadata=ocp.args.JsonSave({"shape": state.shape}),
                system_metadata=ocp.args.JsonSave(metadata),
            ),
        )
    with jdem.CheckpointLoader(directory) as loader:
        restored, restored_system = loader.load()
    restored.validate()
    assert restored.facet_ids_extra.shape == (state.N, 0)
    assert restored.facet_vertices_extra.shape == (state.N, 0, state.dim)
    assert int(restored_system.collider.facet_id_snapshot) == -2
    expected, _ = system.step(state, system)
    actual, _ = restored_system.step(restored, restored_system)
    np.testing.assert_allclose(actual.pos, expected.pos)
    np.testing.assert_allclose(actual.vel, expected.vel)


@pytest.mark.parametrize("legacy_version", [None, 1, 3])
def test_hdf5_current_shared_and_legacy_first_incidence_roundtrip(
    tmp_path, legacy_version
):
    h5py = pytest.importorskip("h5py")
    from jaxdem.utils import h5

    original, elements = _surface(3)
    faces = elements if legacy_version is None else elements[:1]
    state = jdem.State.add_contact_facets(original, faces)
    path = tmp_path / "surface.h5"
    h5.save(state, str(path))
    if legacy_version is not None:
        with h5py.File(path, "r+") as payload:
            del payload["root/facet_ids_extra"]
            del payload["root/facet_vertices_extra"]
            del payload["root"].attrs["__facet_incidence_layout__"]
            if legacy_version == 1:
                del payload.attrs["jaxdem_serialization_manifest"]
        with pytest.warns(RuntimeWarning, match="missing saved fields"):
            restored = h5.load(str(path))
    else:
        restored = h5.load(str(path))
    restored.validate()
    assert _faces(restored) == _faces(state)
    for before, after in zip(
        jax.tree.leaves(state), jax.tree.leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(after, before)


def test_hdf5_current_layout_rejects_missing_extra_fields(tmp_path):
    h5py = pytest.importorskip("h5py")
    from jaxdem.utils import h5

    original, elements = _surface(2)
    state = jdem.State.add_contact_facets(original, elements)
    path = tmp_path / "incomplete.h5"
    h5.save(state, str(path))
    with h5py.File(path, "r+") as payload:
        del payload["root/facet_ids_extra"]
        del payload["root/facet_vertices_extra"]
    with pytest.raises(RuntimeError, match="fields do not match"):
        h5.load(str(path))


@pytest.mark.parametrize(
    "corruption",
    ["missing_shape", "missing_layout", "missing_both", "capacity", "version"],
)
def test_orbax_rejects_incomplete_or_mismatched_incidence_metadata(
    tmp_path, corruption
):
    pytest.importorskip("orbax.checkpoint")
    original, elements = _surface(2)
    state = jdem.State.add_contact_facets(original, elements)
    system = jdem.System.create(state=state)
    directory = tmp_path / "checkpoint"
    with jdem.CheckpointWriter(directory) as writer:
        writer.save(state, system)
    path = directory / "0" / "state_metadata" / "metadata"
    metadata = json.loads(path.read_text())
    if corruption in {"missing_shape", "missing_both"}:
        del metadata["facet_ids_extra_shape"]
    if corruption in {"missing_layout", "missing_both"}:
        del metadata["facet_incidence_layout"]
    if corruption == "capacity":
        metadata["facet_ids_extra_shape"][-1] = 0
    if corruption == "version":
        metadata["facet_incidence_layout"] = 2
    path.write_text(json.dumps(metadata))
    with jdem.CheckpointLoader(directory) as loader:
        with pytest.raises(RuntimeError, match="incidence|layout"):
            loader.load()


def test_hdf5_rejects_unknown_incidence_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    from jaxdem.utils import h5

    original, elements = _surface(2)
    path = tmp_path / "future.h5"
    h5.save(jdem.State.add_contact_facets(original, elements), str(path))
    with h5py.File(path, "r+") as payload:
        payload["root"].attrs["__facet_incidence_layout__"] = 2
    with pytest.raises(RuntimeError, match="Unsupported HDF5 facet incidence layout"):
        h5.load(str(path))
