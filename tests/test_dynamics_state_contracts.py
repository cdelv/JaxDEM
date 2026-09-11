"""Focused contracts for stepping, state validation, and thermal reductions."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.colliders import DynamicCellList, NeighborList
from jaxdem.utils.quaternion import Quaternion
from jaxdem.utils.thermal import (
    compute_potential_energy,
    compute_temperature,
    scale_to_temperature,
    compute_translational_kinetic_energy,
    compute_translational_kinetic_energy_per_particle,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(slots=True)
class _TerminalRotationalKick(jdem.RotationIntegrator):
    @staticmethod
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> tuple[jdem.State, jdem.System]:
        state.ang_vel = jnp.full_like(state.ang_vel, 2.0)
        return state, system


def _free_system(state: jdem.State) -> jdem.System:
    return jdem.System.create(
        state=state,
        dt=0.1,
        rotation_integrator_type=None,
    )


def test_traced_neighbor_constructor_uses_explicit_static_capacity() -> None:
    state = jdem.State.create(pos=jnp.array([[0.0, 0.0], [0.75, 0.0]]))

    @jax.jit
    def create(st: jdem.State) -> NeighborList:
        return NeighborList.Create(
            st,
            cutoff=1.0,
            max_neighbors=2,
            secondary_collider_kw={"search_range": 1},
        )

    collider = create(state)
    assert collider.neighbor_list.shape == (2, 2)
    host_cell_list = DynamicCellList.Create(
        state, cell_size=1.0, search_range=jnp.asarray(1)
    )
    assert host_cell_list.neighbor_mask.shape == (9, 2)


def test_static_step_supports_reverse_mode_and_dynamic_step_is_explicit() -> None:
    state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
    system = _free_system(state)

    def terminal_x(vx: jax.Array) -> jax.Array:
        initial = dataclasses.replace(state, vel=jnp.array([[vx, 0.0]]))
        final, _ = jdem.System.step(initial, system, n=3)
        return final.pos_c[0, 0]

    np.testing.assert_allclose(jax.grad(terminal_x)(2.0), 0.3, rtol=1e-6)

    def rollout_terminal_x(vx: jax.Array) -> jax.Array:
        initial = dataclasses.replace(state, vel=jnp.array([[vx, 0.0]]))
        final, _, _ = jdem.System.trajectory_rollout(
            initial, system, n=2, stride=2, save_fn=lambda s, _: s.pos_c
        )
        return final.pos_c[0, 0]

    np.testing.assert_allclose(jax.grad(rollout_terminal_x)(2.0), 0.4, rtol=1e-6)
    dynamic, _ = jdem.System.step_dynamic(state, system, n=jnp.asarray(2))
    assert int(dynamic.N) == 1
    with pytest.raises(ValueError, match="Python integer"):
        jdem.System.step(state, system, n=jnp.asarray(2))


def test_mutable_quaternion_refreshes_explicitly_and_replace_is_automatic() -> None:
    state = jdem.State.create(pos=jnp.zeros((1, 2)), pos_p=jnp.array([[1.0, 0.0]]))
    q = Quaternion.create(jnp.array([[2**-0.5]]), jnp.array([[0.0, 0.0, 2**-0.5]]))
    state.q.w = q.w
    state.q.xyz = q.xyz
    with pytest.raises(ValueError, match="cache is stale"):
        state.validate()
    state.refresh_rotation_cache()
    np.testing.assert_allclose(state.pos, jnp.array([[0.0, 1.0]]), atol=1e-6)
    state.validate()

    identity = Quaternion.create(jnp.ones((1, 1)), jnp.zeros((1, 3)))
    state = dataclasses.replace(state, q=identity)
    np.testing.assert_allclose(state.pos, jnp.array([[1.0, 0.0]]), atol=1e-6)

    changed = dataclasses.replace(state, q=q)
    np.testing.assert_allclose(changed.pos, jnp.array([[0.0, 1.0]]), atol=1e-6)


def test_opt_in_validation_checks_physics_but_construction_stays_permissive() -> None:
    state = jdem.State.create(pos=jnp.zeros((1, 2)), mass=jnp.array([0.0]))
    assert state.mass[0] == 0.0
    with pytest.raises(ValueError, match="positive mass"):
        state.validate()

    state = jdem.State.create(pos=jnp.zeros((1, 2)), mat_id=jnp.array([1]))
    system = _free_system(state)
    with pytest.raises(ValueError, match="material table"):
        system.validate(state)

    invalid_ids = dataclasses.replace(state, clump_id=state.clump_id.astype(float))
    with pytest.raises(ValueError, match="integer dtype"):
        invalid_ids.validate()
    with pytest.raises(ValueError, match="mat_id entries must be nonnegative"):
        dataclasses.replace(state, mat_id=jnp.array([-1])).validate()
    with pytest.raises(ValueError, match="species_id entries must be nonnegative"):
        dataclasses.replace(state, species_id=jnp.array([-1])).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        dataclasses.replace(state, rad=jnp.array([-1.0])).validate()

    malformed_q = Quaternion.create(jnp.ones((2, 1)), jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="array shapes"):
        dataclasses.replace(state, q=malformed_q).validate()
    with pytest.raises(ValueError, match="non-finite"):
        dataclasses.replace(
            state, facet_vertices=jnp.full(state.facet_vertices.shape, jnp.inf)
        ).validate()


def test_clump_validation_handles_sparse_ids_and_detects_disagreement() -> None:
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        clump_id=jnp.array([1, 1, 3, 3]),
        mass=jnp.array([2.0, 2.0, 4.0, 4.0]),
    )
    state.validate()
    with pytest.raises(ValueError, match="mass.*not replicated"):
        dataclasses.replace(state, mass=state.mass.at[1].set(3.0)).validate()


def test_validation_rejects_nonboolean_fixed_before_dynamic_masking() -> None:
    state = jdem.State.create(pos=jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match="boolean dtype"):
        dataclasses.replace(state, fixed=state.fixed.astype(int)).validate()


def test_thermal_reductions_preserve_all_leading_snapshot_axes() -> None:
    first = jdem.State.create(
        pos=jnp.zeros((2, 2)), vel=jnp.array([[1.0, 0.0], [0.0, 2.0]])
    )
    second = dataclasses.replace(first, vel=2.0 * first.vel)
    batch = jdem.State.stack([first, second])
    trajectory = jax.tree.map(lambda x: jnp.stack([x, x]), batch)

    assert compute_translational_kinetic_energy_per_particle(batch).shape == (2, 2)
    assert compute_translational_kinetic_energy(batch).shape == (2,)
    assert compute_temperature(batch, False, False).shape == (2,)
    assert compute_translational_kinetic_energy(trajectory).shape == (2, 2)
    np.testing.assert_allclose(
        compute_translational_kinetic_energy(batch), jnp.array([2.5, 10.0])
    )

    first_system = _free_system(first)
    second_system = _free_system(second)
    second_table = dataclasses.replace(
        second_system.mat_table,
        props={
            name: 2.0 * value for name, value in second_system.mat_table.props.items()
        },
        pair={
            name: 2.0 * value for name, value in second_system.mat_table.pair.items()
        },
    )
    second_system = dataclasses.replace(second_system, mat_table=second_table)
    systems = jdem.System.stack([first_system, second_system])
    assert compute_potential_energy(batch, systems).shape == (2,)


def test_rescaling_removes_mass_weighted_body_drift() -> None:
    # Body 0 has two members but total mass 2; body 1 has one member and mass 6.
    state = jdem.State.create(
        pos=jnp.zeros((3, 2)),
        vel=jnp.array([[3.0, 0.0], [3.0, 0.0], [-1.0, 0.0]]),
        mass=jnp.array([2.0, 2.0, 6.0]),
        clump_id=jnp.array([0, 0, 1]),
    )
    integrator = jdem.LinearIntegrator.create(
        "verlet_rescaling", temperature=1.0, subtract_drift=True
    )
    system = dataclasses.replace(
        _free_system(state), linear_integrator=integrator, step_count=jnp.asarray(1)
    )
    state, _ = integrator.finalize_step(state, system)

    member_mass = jnp.array([1.0, 1.0, 6.0])
    drift = jnp.sum(state.vel * member_mass[:, None], axis=0) / member_mass.sum()
    np.testing.assert_allclose(drift, 0.0, atol=1e-6)


def test_rescaling_drift_is_correct_below_unit_total_mass() -> None:
    state = jdem.State.create(
        pos=jnp.zeros((3, 2)),
        vel=jnp.array([[2.0, 0.0], [2.0, 0.0], [-1.0, 0.0]]),
        mass=jnp.array([0.2, 0.2, 0.6]),
        clump_id=jnp.array([0, 0, 1]),
    )
    integrator = jdem.LinearIntegrator.create(
        "verlet_rescaling", temperature=1.0, subtract_drift=True
    )
    system = dataclasses.replace(
        _free_system(state), linear_integrator=integrator, step_count=jnp.asarray(1)
    )
    state, _ = integrator.finalize_step(state, system)
    member_mass = jnp.array([0.1, 0.1, 0.6])
    drift = jnp.sum(state.vel * member_mass[:, None], axis=0) / member_mass.sum()
    np.testing.assert_allclose(drift, 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rescale_every": 0.5},
        {"temperature": float("nan")},
        {"k_B": float("inf")},
    ],
)
def test_rescaling_constructor_rejects_invalid_host_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        jdem.LinearIntegrator.create("verlet_rescaling", **kwargs)


def test_rescaling_runs_after_the_rotational_terminal_kick() -> None:
    state = jdem.State.create(pos=jnp.zeros((1, 2)), mass=jnp.ones(1))
    integrator = jdem.LinearIntegrator.create(
        "verlet_rescaling", temperature=3.0, can_rotate=True
    )
    system = dataclasses.replace(
        _free_system(state),
        linear_integrator=integrator,
        rotation_integrator=_TerminalRotationalKick(),
    )
    state, system = jdem.System.step(state, system)
    np.testing.assert_allclose(
        compute_temperature(state, can_rotate=True, subtract_drift=False),
        3.0,
        rtol=1e-6,
    )


def test_temperature_scaling_excludes_and_preserves_fixed_particles() -> None:
    state = jdem.State.create(
        pos=jnp.zeros((2, 2)),
        vel=jnp.array([[10.0, 0.0], [1.0, 0.0]]),
        fixed=jnp.array([True, False]),
    )
    np.testing.assert_allclose(compute_temperature(state, False, False), 0.5)
    scaled = scale_to_temperature(state, 2.0, False, False)
    np.testing.assert_allclose(scaled.vel[0], jnp.array([10.0, 0.0]))


def test_temperature_scaling_uses_body_mass_once_for_clump_drift() -> None:
    state = jdem.State.create(
        pos=jnp.zeros((3, 2)),
        vel=jnp.array([[3.0, 0.0], [3.0, 0.0], [-1.0, 0.0]]),
        mass=jnp.array([2.0, 2.0, 6.0]),
        clump_id=jnp.array([0, 0, 1]),
    )
    scaled = scale_to_temperature(state, 1.0, False, True)
    member_mass = jnp.array([1.0, 1.0, 6.0])
    drift = jnp.sum(scaled.vel * member_mass[:, None], axis=0) / member_mass.sum()
    np.testing.assert_allclose(drift, 0.0, atol=1e-6)


def test_public_state_guards_survive_python_optimization() -> None:
    code = """
import jax.numpy as jnp
from jaxdem import State
a = State.create(pos=jnp.zeros((1, 2)))
b = State.create(pos=jnp.zeros((1, 3)))
try:
    State.stack([a, b])
except ValueError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run([sys.executable, "-O", "-c", code], check=False)
    assert result.returncode == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n": -1}, "nonnegative"),
        ({"n": 1, "stride": 0.5}, "stride"),
        ({"strides": jnp.array([0.5])}, "integer array"),
        ({"strides": jnp.zeros((1, 1), dtype=int)}, "1D integer array"),
    ],
)
def test_rollout_rejects_invalid_counts(
    kwargs: dict[str, object], message: str
) -> None:
    state = jdem.State.create(pos=jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match=message):
        jdem.System.trajectory_rollout(state, _free_system(state), **kwargs)
