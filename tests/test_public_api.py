# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unit tests for small public APIs: registry-key normalization,
``System.with_integrators``, ``refresh_collider``, ``trajectory_rollout``
frame options, and VTK writer-name resolution."""

from pathlib import Path

import jax.numpy as jnp
import pytest

import jaxdem as jdem


def _two_spheres() -> tuple[jdem.State, jdem.System]:
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
        rad=jnp.array([1.0, 1.0]),
    )
    system = jdem.System.create(state.shape)
    return state, system


def test_registry_keys_are_normalized():
    """All spellings of a registry key resolve to the same class."""
    classes = {
        type(jdem.Collider.create("CellList", state=_two_spheres()[0])),
        type(jdem.Collider.create("cell_list", state=_two_spheres()[0])),
        type(jdem.Collider.create("celllist", state=_two_spheres()[0])),
        type(jdem.Collider.create("Cell-List", state=_two_spheres()[0])),
    }
    assert len(classes) == 1

    with pytest.raises(KeyError, match="Unknown Collider"):
        jdem.Collider.create("not_a_collider")


@pytest.mark.parametrize(
    "collider_type", ["naive", "cell_list", "multi_cell_list", "neighbor_list"]
)
def test_refresh_collider_after_resize(collider_type):
    state, _ = _two_spheres()
    collider_kw = {"state": state}
    if collider_type == "neighbor_list":
        collider_kw["cutoff"] = 2.5
    if collider_type == "naive":
        collider_kw = {}
    system = jdem.System.create(
        state.shape, collider_type=collider_type, collider_kw=collider_kw
    )

    # Resizing the state: stateful colliders rebuild their N-sized buffers.
    bigger = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [4.5, 0.0]]),
        rad=jnp.ones(4),
    )
    refreshed = jdem.colliders.refresh_collider(bigger, system.collider)
    assert type(refreshed) is type(system.collider)
    if hasattr(refreshed, "neighbor_list"):
        assert refreshed.neighbor_list.shape[0] == bigger.N

    # Editing positions at the same N: the refreshed collider must step.
    state.pos_c = state.pos_c + 10.0
    system.collider = jdem.colliders.refresh_collider(state, system.collider)
    state, system = system.step(state, system)
    assert bool(jnp.all(jnp.isfinite(state.pos)))


def test_trajectory_rollout_initial_frame_via_zero_stride():
    state, system = _two_spheres()
    _, _, traj = system.trajectory_rollout(state, system, n=3, stride=1)
    assert traj[0].pos.shape[0] == 3
    # A leading 0 stride records the initial (step-0) state as the first frame.
    _, _, traj_init = system.trajectory_rollout(
        state, system, strides=jnp.array([0, 1, 1, 1])
    )
    assert traj_init[0].pos.shape[0] == 4
    assert bool(jnp.all(traj_init[0].pos[0] == state.pos))


def test_vtk_writer_name_normalization(tmp_path: Path):
    state, system = _two_spheres()
    # Underscored user-facing spellings must resolve to the registered writers.
    writer = jdem.VTKWriter(
        directory=tmp_path, writers=["spheres", "facet_spheres", "domain"]
    )
    writer.save(state, system)
    writer.close()
    files = sorted(p.name for p in (tmp_path / "batch_00000000").glob("*.vtp"))
    assert any(f.startswith("spheres") for f in files)

    with pytest.raises(KeyError, match="Unknown VTKBaseWriter"):
        jdem.VTKWriter(directory=tmp_path, writers=["not_a_writer"])
