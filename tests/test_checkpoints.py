# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Comprehensive checkpoint round-trip tests.

Each test creates a State/System, runs one step, saves a checkpoint, loads it
back and verifies that every JAX leaf matches the original.
"""

from __future__ import annotations

import tempfile
import os
from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jdem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmpdir() -> str:
    """Return a fresh temporary checkpoint directory path."""
    return os.path.join(tempfile.mkdtemp(), "ckpt")


def _assert_leaves_equal(
    original,
    restored,
    label: str = "",
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert that every JAX leaf of *original* and *restored* is close."""
    leaves_orig = jax.tree_util.tree_leaves(original)
    leaves_rest = jax.tree_util.tree_leaves(restored)
    assert len(leaves_orig) == len(
        leaves_rest
    ), f"{label}: leaf count mismatch {len(leaves_orig)} vs {len(leaves_rest)}"
    for i, (lo, lr) in enumerate(zip(leaves_orig, leaves_rest)):
        if jnp.issubdtype(lo.dtype, jnp.floating):
            assert jnp.allclose(
                lo, lr, rtol=rtol, atol=atol
            ), f"{label} leaf {i}: {lo} != {lr}"
        else:
            assert jnp.array_equal(lo, lr), f"{label} leaf {i}: {lo} != {lr}"


def _round_trip(
    state: jdem.State, system: jdem.System
) -> Tuple[jdem.State, jdem.System]:
    """Save then load a checkpoint and return the restored pair."""
    d = _tmpdir()
    with jdem.CheckpointWriter(d) as w:
        w.save(state, system)
    return jdem.CheckpointLoader(d).load()


def _save_step_check(state: jdem.State, system: jdem.System, n: int = 1) -> None:
    """Run *n* steps, round-trip through a checkpoint, assert equality."""
    state, system = system.step(state, system, n=n)
    state_r, system_r = _round_trip(state, system)
    _assert_leaves_equal(state, state_r, "state")
    _assert_leaves_equal(system, system_r, "system")


# ===================================================================
# Force-model tests
# ===================================================================


class TestForceModels:
    """Checkpoint round-trips for every registered force model."""

    def test_spring(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape, force_model_type="spring")
        _save_step_check(state, system)

    def test_wca(self):
        lj_mat = jdem.MaterialTable.from_materials(
            [jdem.Material.create("lj", density=1.0, epsilon=1.0)]
        )
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape, force_model_type="wca", mat_table=lj_mat
        )
        _save_step_check(state, system)

    def test_wca_shifted(self):
        lj_mat = jdem.MaterialTable.from_materials(
            [jdem.Material.create("lj", density=1.0, epsilon=1.0)]
        )
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape, force_model_type="wca_shifted", mat_table=lj_mat
        )
        _save_step_check(state, system)

    def test_lennard_jones(self):
        lj_mat = jdem.MaterialTable.from_materials(
            [jdem.Material.create("lj", density=1.0, epsilon=1.0)]
        )
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [2.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape, force_model_type="lennardjones", mat_table=lj_mat
        )
        _save_step_check(state, system)

    def test_law_combiner_spring_wca(self):
        """LawCombiner compositing spring + WCA."""
        mat = jdem.Material.create("elastic", density=1.0, young=1e4, poisson=0.3)
        mat_lj = jdem.Material.create("lj", density=1.0, epsilon=1.0)
        mat_table = jdem.MaterialTable.from_materials([mat, mat_lj])

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            mat_id=jnp.array([0, 1]),
        )
        system = jdem.System.create(
            state.shape,
            force_model_type="lawcombiner",
            force_model_kw=dict(
                laws=(
                    jdem.ForceModel.create("spring"),
                    jdem.ForceModel.create("wca"),
                )
            ),
            mat_table=mat_table,
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)

        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")

        # Verify law types survived round-trip
        assert isinstance(system_r.force_model, jdem.LawCombiner)
        assert len(system_r.force_model.laws) == 2

    def test_force_router(self):
        """ForceRouter with mixed species interactions."""
        mat = jdem.Material.create("elastic", density=1.0, young=1e4, poisson=0.3)
        mat_lj = jdem.Material.create("lj", density=1.0, epsilon=1.0)

        router = jdem.ForceRouter.from_dict(
            S=2,
            mapping={
                (0, 0): jdem.ForceModel.create("spring"),
                (1, 1): jdem.ForceModel.create("wca"),
                (0, 1): jdem.LawCombiner(
                    laws=(
                        jdem.ForceModel.create("spring"),
                        jdem.ForceModel.create("wca"),
                    )
                ),
            },
        )

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]),
            rad=jnp.array([1.0, 1.0, 1.0]),
            species_id=jnp.array([0, 0, 1]),
        )
        system = jdem.System.create(
            state.shape,
            force_model_type="forcerouter",
            force_model_kw=dict(table=router.table),
            mat_table=jdem.MaterialTable.from_materials([mat, mat_lj]),
        )

        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)

        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")

        # Structural checks
        assert isinstance(system_r.force_model, jdem.ForceRouter)
        assert len(system_r.force_model.table) == 2
        assert len(system_r.force_model.table[0]) == 2

    def test_force_router_single_species(self):
        """ForceRouter with a single species (1×1 table)."""
        router = jdem.ForceRouter.from_dict(
            S=1,
            mapping={(0, 0): jdem.ForceModel.create("spring")},
        )

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            species_id=jnp.array([0, 0]),
        )
        system = jdem.System.create(
            state.shape,
            force_model_type="forcerouter",
            force_model_kw=dict(table=router.table),
        )

        _save_step_check(state, system)

    def test_law_combiner_empty_laws(self):
        """LawCombiner with no laws (zero-force default)."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape,
            force_model_type="lawcombiner",
        )
        _save_step_check(state, system)


# ===================================================================
# Domain tests
# ===================================================================


class TestDomains:
    """Checkpoint round-trips for every domain type."""

    def test_free_domain(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [3.0, 4.0]]),
            rad=jnp.array([0.5, 0.5]),
        )
        system = jdem.System.create(state.shape, domain_type="free")
        _save_step_check(state, system)

    def test_periodic_domain(self):
        state = jdem.State.create(
            pos=jnp.array([[0.1, 0.1], [9.9, 9.9]]),
            rad=jnp.array([0.5, 0.5]),
        )
        system = jdem.System.create(
            state.shape,
            domain_type="periodic",
            domain_kw=dict(box_size=10.0 * jnp.ones(2), anchor=jnp.zeros(2)),
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        assert jnp.allclose(system.domain.box_size, system_r.domain.box_size)

    def test_reflect_domain(self):
        state = jdem.State.create(
            pos=jnp.array([[0.5, 0.5]]),
            vel=jnp.array([[-1.0, 0.0]]),
            rad=jnp.array([0.4]),
        )
        system = jdem.System.create(
            state.shape,
            domain_type="reflect",
            domain_kw=dict(
                box_size=10.0 * jnp.ones(2),
                anchor=jnp.zeros(2),
                restitution_coefficient=1.0,
            ),
        )
        _save_step_check(state, system)

    def test_reflect_sphere_domain(self):
        state = jdem.State.create(
            pos=jnp.array([[0.5, 0.5]]),
            vel=jnp.array([[-1.0, 0.0]]),
            rad=jnp.array([0.4]),
        )
        system = jdem.System.create(
            state.shape,
            domain_type="reflectsphere",
            domain_kw=dict(box_size=10.0 * jnp.ones(2), anchor=jnp.zeros(2)),
        )
        _save_step_check(state, system)


# ===================================================================
# Integrator / minimizer tests
# ===================================================================


class TestIntegrators:
    """Checkpoint round-trips for integrator and minimizer types."""

    def test_velocity_verlet_2d(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape,
            linear_integrator_type="verlet",
            rotation_integrator_type="",
        )
        _save_step_check(state, system)

    def test_velocity_verlet_3d_with_rotation(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape,
            linear_integrator_type="verlet",
            rotation_integrator_type="verletspiral",
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        # Quaternion round-trip
        assert jnp.allclose(state.q.w, state_r.q.w)
        assert jnp.allclose(state.q.xyz, state_r.q.xyz)

    def test_euler_integrator(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape,
            linear_integrator_type="euler",
            rotation_integrator_type="",
        )
        _save_step_check(state, system)

    def test_gradient_descent(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape,
            linear_integrator_type="lineargradientdescent",
            rotation_integrator_type="",
            linear_integrator_kw=dict(learning_rate=1e-4),
        )
        _save_step_check(state, system)

    def test_noop_integrators(self):
        """Deactivated integrators (empty-string keys)."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape,
            linear_integrator_type="",
            rotation_integrator_type="",
        )
        _save_step_check(state, system)


# ===================================================================
# Collider tests
# ===================================================================


class TestColliders:
    """Checkpoint round-trips for collider types."""

    def test_naive(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]),
            rad=jnp.array([1.0, 1.0, 1.0]),
        )
        system = jdem.System.create(state.shape, collider_type="naive")
        _save_step_check(state, system)

    def test_noop_collider(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape, collider_type="")
        _save_step_check(state, system)

    def test_cell_list(self):
        state = jdem.State.create(
            pos=jnp.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]),
            rad=jnp.array([0.5, 0.5, 0.5]),
        )
        system = jdem.System.create(
            state.shape,
            collider_type="CellList",
            collider_kw=dict(state=state),
        )
        _save_step_check(state, system)

    def test_static_cell_list(self):
        state = jdem.State.create(
            pos=jnp.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]),
            rad=jnp.array([0.5, 0.5, 0.5]),
        )
        system = jdem.System.create(
            state.shape,
            collider_type="StaticCellList",
            collider_kw=dict(state=state),
        )
        _save_step_check(state, system)

    def test_neighbor_list(self):
        state = jdem.State.create(
            pos=jnp.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]]),
            rad=jnp.array([0.5, 0.5, 0.5]),
        )
        system = jdem.System.create(
            state.shape,
            collider_type="NeighborList",
            collider_kw=dict(
                state=state,
                cutoff=2.0,
                skin=0.1,
                secondary_collider_type="CellList",
                secondary_collider_kw=dict(state=state),
                max_neighbors=8,
            ),
        )
        _save_step_check(state, system)


# ===================================================================
# Material / MaterialTable tests
# ===================================================================


class TestMaterials:
    """Checkpoint round-trips with different material configurations."""

    def test_default_elastic(self):
        """Default single elastic material."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        assert sorted(system.mat_table.props.keys()) == sorted(
            system_r.mat_table.props.keys()
        )

    def test_lj_material(self):
        lj_mat = jdem.MaterialTable.from_materials(
            [jdem.Material.create("lj", density=1.0, epsilon=1.0)]
        )
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape, force_model_type="wca", mat_table=lj_mat
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        assert "epsilon" in system_r.mat_table.props
        assert "epsilon_eff" in system_r.mat_table.pair

    def test_multi_material_elastic(self):
        """Two elastic materials with harmonic matchmaker."""
        elastic = jdem.Material.create(
            "elastic", density=2500.0, young=2.0e5, poisson=0.25
        )
        frictional = jdem.Material.create(
            "elasticfrict", density=1200.0, young=8.0e4, poisson=0.35, mu=0.6
        )
        harmonic_matcher = jdem.MaterialMatchmaker.create("harmonic")
        mat_table = jdem.MaterialTable.from_materials(
            [elastic, frictional], matcher=harmonic_matcher, fill=0.0
        )

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            mat_id=jnp.array([0, 1]),
        )
        system = jdem.System.create(
            state.shape, force_model_type="spring", mat_table=mat_table
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")

        assert "mu" in system_r.mat_table.props
        assert jnp.allclose(
            system.mat_table.pair["young_eff"],
            system_r.mat_table.pair["young_eff"],
        )

    def test_linear_matchmaker(self):
        """Two elastic materials with linear matchmaker."""
        elastic = jdem.Material.create(
            "elastic", density=2500.0, young=2.0e5, poisson=0.25
        )
        frictional = jdem.Material.create(
            "elasticfrict", density=1200.0, young=8.0e4, poisson=0.35, mu=0.6
        )
        linear_matcher = jdem.MaterialMatchmaker.create("linear")
        mat_table = jdem.MaterialTable.from_materials(
            [elastic, frictional], matcher=linear_matcher, fill=0.0
        )

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            mat_id=jnp.array([0, 1]),
        )
        system = jdem.System.create(
            state.shape, force_model_type="spring", mat_table=mat_table
        )
        _save_step_check(state, system)

    def test_mixed_elastic_lj(self):
        """Mixed elastic + LJ materials (requires both young_eff and epsilon_eff)."""
        mat = jdem.Material.create("elastic", density=1.0, young=1e4, poisson=0.3)
        mat_lj = jdem.Material.create("lj", density=1.0, epsilon=1.0)
        mat_table = jdem.MaterialTable.from_materials([mat, mat_lj])

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            mat_id=jnp.array([0, 1]),
        )
        system = jdem.System.create(
            state.shape,
            force_model_type="lawcombiner",
            force_model_kw=dict(
                laws=(
                    jdem.ForceModel.create("spring"),
                    jdem.ForceModel.create("wca"),
                )
            ),
            mat_table=mat_table,
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")

        assert "young" in system_r.mat_table.props
        assert "epsilon" in system_r.mat_table.props
        assert "young_eff" in system_r.mat_table.pair
        assert "epsilon_eff" in system_r.mat_table.pair


# ===================================================================
# State-feature tests
# ===================================================================


class TestStateFeatures:
    """Checkpoint round-trips for various State features."""

    def test_fixed_particles(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            fixed=jnp.array([True, False]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)
        state_r, _ = _round_trip(state, system)
        assert jnp.array_equal(state.fixed, state_r.fixed)

    def test_bond_id(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]),
            rad=jnp.array([1.0, 1.0, 1.0]),
            bond_id=jnp.array([0, 1, 1]),
        )
        system = jdem.System.create(state.shape, interact_same_bond_id=False)
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        assert jnp.array_equal(state.bond_id, state_r.bond_id)
        assert bool(system_r.interact_same_bond_id) == False

    def test_species_id(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            species_id=jnp.array([0, 1]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)
        state_r, _ = _round_trip(state, system)
        assert jnp.array_equal(state.species_id, state_r.species_id)

    def test_mat_id(self):
        mat_a = jdem.Material.create("lj", density=1.0, epsilon=1.0)
        mat_b = jdem.Material.create("lj", density=1.0, epsilon=2.0)
        mat_table = jdem.MaterialTable.from_materials([mat_a, mat_b])

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
            mat_id=jnp.array([0, 1]),
        )
        system = jdem.System.create(
            state.shape, force_model_type="wca", mat_table=mat_table
        )
        state, system = system.step(state, system)
        state_r, _ = _round_trip(state, system)
        assert jnp.array_equal(state.mat_id, state_r.mat_id)

    def test_velocity_and_force(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            vel=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)
        state_r, _ = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")

    def test_many_particles(self):
        """Larger system with 50 particles."""
        N = 50
        key = jax.random.PRNGKey(42)
        pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=20.0)
        rad = 0.3 * jnp.ones(N)
        state = jdem.State.create(pos=pos, rad=rad)
        system = jdem.System.create(state.shape)
        _save_step_check(state, system)

    def test_single_particle(self):
        """Edge case: one particle."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0]]),
            rad=jnp.array([1.0]),
        )
        system = jdem.System.create(state.shape)
        _save_step_check(state, system)


# ===================================================================
# Clump (rigid body) tests
# ===================================================================


class TestClumps:
    """Checkpoint round-trips involving rigid-body clumps."""

    def test_clump_basic(self):
        """Dumbbell clump + free sphere."""
        mat = jdem.Material.create("elastic", density=2.0, young=1e4, poisson=0.3)
        mat_table = jdem.MaterialTable.from_materials([mat])

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0]]),
            rad=jnp.array([1.0]),
            mat_table=mat_table,
        )
        state.fixed = jnp.array([True])
        state = jdem.State.add_clump(
            state,
            pos=jnp.array([[0.0, 4.0], [0.0, 4.0]]),
            pos_p=jnp.array([[-0.4, 0.0], [0.4, 0.0]]),
            rad=jnp.array([0.4, 0.4]),
        )
        state = jdem.utils.compute_clump_properties(state, mat_table)

        system = jdem.System.create(
            state.shape,
            dt=1e-4,
            force_model_type="spring",
            mat_table=mat_table,
            force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
        )

        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        assert jnp.array_equal(state.clump_id, state_r.clump_id)
        assert jnp.array_equal(state.fixed, state_r.fixed)


# ===================================================================
# Bonded force model tests
# ===================================================================


class TestBondedForceModel:
    """Checkpoint round-trips for bonded force models (deformable particles)."""

    def test_deformable_particle_2d(self):
        """2D deformable particle (no step due to pre-existing donate bug)."""
        vertices = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
        )
        elements = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
        adjacency = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

        dp = jdem.BondedForceModel.create(
            "deformableparticlemodel",
            vertices=vertices,
            elements=elements,
            edges=elements,
            element_adjacency=adjacency,
            em=1.0,
            eb=0.5,
            el=0.3,
            gamma=0.1,
        )
        state = jdem.State.create(pos=vertices)
        system = jdem.System.create(
            state.shape, bonded_force_model=dp, interact_same_bond_id=False
        )
        # Give non-trivial velocities (skip step to avoid pre-existing donate bug)
        state.vel = jnp.array([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]])

        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        assert system_r.bonded_force_model is not None
        assert type(system_r.bonded_force_model).__name__ == "DeformableParticleModel"


# ===================================================================
# ForceManager / gravity tests
# ===================================================================


class TestForceManager:
    """Checkpoint round-trips for force manager configurations."""

    def test_gravity(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        assert jnp.allclose(
            system.force_manager.gravity, system_r.force_manager.gravity
        )

    def test_no_gravity(self):
        """Default (zero) gravity."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)
        _save_step_check(state, system)


# ===================================================================
# Custom force function tests
# ===================================================================


class TestCustomForceFunctions:
    """Checkpoint round-trips for user-supplied force/energy functions."""

    def test_single_force_with_energy(self):
        """Force function + energy function, both restored."""
        from tests.custom_forces import harmonic_trap, harmonic_trap_energy

        state = jdem.State.create(pos=jnp.array([[2.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(
                force_functions=[(harmonic_trap, harmonic_trap_energy)],
            ),
            dt=0.01,
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)

        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")

        fm = system_r.force_manager
        assert len(fm.force_functions) == 1
        assert fm.force_functions[0].__name__ == "harmonic_trap"
        assert fm.energy_functions[0].__name__ == "harmonic_trap_energy"

    def test_force_without_energy(self):
        """Force function with no energy companion."""
        from tests.custom_forces import constant_push

        state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(
                force_functions=[(constant_push,)],
            ),
            dt=0.01,
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)

        _assert_leaves_equal(state, state_r, "state")
        fm = system_r.force_manager
        assert len(fm.force_functions) == 1
        assert fm.force_functions[0].__name__ == "constant_push"

    def test_is_com_flag(self):
        """COM flag is preserved through save/load."""
        from tests.custom_forces import constant_push

        state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(
                force_functions=[(constant_push, None, True)],
            ),
            dt=0.01,
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)

        fm = system_r.force_manager
        assert fm.is_com_force == (True,)
        assert fm.force_functions[0].__name__ == "constant_push"

    def test_multiple_force_functions(self):
        """Multiple force functions, mixed energy and COM flags."""
        from tests.custom_forces import (
            harmonic_trap,
            harmonic_trap_energy,
            constant_push,
        )

        state = jdem.State.create(
            pos=jnp.array([[2.0, 0.0], [3.0, 0.0]]),
            rad=jnp.array([0.5, 0.5]),
        )
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(
                force_functions=[
                    (harmonic_trap, harmonic_trap_energy, False),
                    (constant_push, None, True),
                ],
            ),
            dt=0.01,
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)

        _assert_leaves_equal(state, state_r, "state")
        fm = system_r.force_manager
        assert len(fm.force_functions) == 2
        assert fm.force_functions[0].__name__ == "harmonic_trap"
        assert fm.force_functions[1].__name__ == "constant_push"
        assert fm.energy_functions[0].__name__ == "harmonic_trap_energy"
        assert fm.is_com_force == (False, True)

    def test_custom_force_with_bonded_model(self):
        """Custom force function + bonded model: only user fns serialized."""
        from tests.custom_forces import harmonic_trap, harmonic_trap_energy

        vertices = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
        )
        elements = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
        adjacency = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

        dp = jdem.BondedForceModel.create(
            "deformableparticlemodel",
            vertices=vertices,
            elements=elements,
            edges=elements,
            element_adjacency=adjacency,
            em=1.0,
            eb=0.5,
            el=0.3,
            gamma=0.1,
        )

        state = jdem.State.create(pos=vertices)
        system = jdem.System.create(
            state.shape,
            bonded_force_model=dp,
            interact_same_bond_id=False,
            force_manager_kw=dict(
                force_functions=[(harmonic_trap, harmonic_trap_energy)],
            ),
        )

        # 2 total: 1 user + 1 bonded
        assert len(system.force_manager.force_functions) == 2

        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")

        # Should still have 2 after round-trip (user fn + bonded re-added)
        fm = system_r.force_manager
        assert len(fm.force_functions) == 2
        assert fm.force_functions[0].__name__ == "harmonic_trap"

    def test_restored_force_produces_same_result(self):
        """Verify the restored force function actually works post-load."""
        from tests.custom_forces import harmonic_trap, harmonic_trap_energy

        state = jdem.State.create(pos=jnp.array([[2.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(
                force_functions=[(harmonic_trap, harmonic_trap_energy)],
            ),
            dt=0.1,
        )

        # Step the original
        state, system = system.step(state, system)
        # Save/load
        state_r, system_r = _round_trip(state, system)
        # Step both from the same state
        state_a, _ = system.step(state, system)
        state_b, _ = system_r.step(state_r, system_r)

        _assert_leaves_equal(state_a, state_b, "post-restore step")

    def test_main_module_warning_on_save(self):
        """Saving a __main__-scoped function emits a warning."""
        import types

        # Create a function that looks like it's from __main__
        def _dummy_force(pos, state, system):
            return pos * 0, jnp.zeros_like(state.torque)

        _dummy_force.__module__ = "__main__"

        state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(force_functions=[(_dummy_force,)]),
            dt=0.01,
        )
        state, system = system.step(state, system)

        import warnings

        d = _tmpdir()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with jdem.CheckpointWriter(d) as w:
                w.save(state, system)

        main_warns = [w for w in caught if "__main__" in str(w.message)]
        assert len(main_warns) >= 1, "Expected warning about __main__ function"

    def test_unresolvable_force_skipped_on_load(self):
        """Unresolvable force functions are skipped with a warning at load."""
        from tests.custom_forces import harmonic_trap, harmonic_trap_energy

        state = jdem.State.create(pos=jnp.array([[2.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw=dict(
                force_functions=[(harmonic_trap, harmonic_trap_energy)],
            ),
            dt=0.01,
        )
        state, system = system.step(state, system)

        d = _tmpdir()
        with jdem.CheckpointWriter(d) as w:
            w.save(state, system)

        # Tamper with the saved metadata to make the function unresolvable
        import json
        from pathlib import Path

        meta_file = list(Path(d).rglob("system_metadata/metadata"))[0]
        meta = json.loads(meta_file.read_bytes())
        meta["force_function_metadata"][0]["force"] = "nonexistent_module.fake_force"
        meta_file.write_text(json.dumps(meta))

        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            state_r, system_r = jdem.CheckpointLoader(d).load()

        skip_warns = [w for w in caught if "Could not restore" in str(w.message)]
        assert len(skip_warns) >= 1, "Expected warning about skipped function"
        # The force function should have been dropped
        assert len(system_r.force_manager.force_functions) == 0


# ===================================================================
# System metadata / misc tests
# ===================================================================


class TestSystemMeta:
    """Checkpoint round-trips for system-level metadata."""

    def test_custom_dt(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape, dt=0.001)
        state, system = system.step(state, system)
        _, system_r = _round_trip(state, system)
        assert jnp.allclose(system.dt, system_r.dt)

    def test_step_count_preserved(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system, n=5)
        _, system_r = _round_trip(state, system)
        assert int(system.step_count) == int(system_r.step_count)

    def test_time_preserved(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape, dt=0.01)
        state, system = system.step(state, system, n=10)
        _, system_r = _round_trip(state, system)
        assert jnp.allclose(system.time, system_r.time)

    def test_seed_and_key(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape, seed=42)
        state, system = system.step(state, system)
        _, system_r = _round_trip(state, system)
        assert jnp.array_equal(system.key, system_r.key)


# ===================================================================
# Dimension tests
# ===================================================================


class TestDimensions:
    """Checkpoint round-trips for 2D and 3D systems."""

    def test_2d(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        assert int(system_r.dim) == 2

    def test_3d(self):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        assert int(system_r.dim) == 3


# ===================================================================
# Complex / combined scenarios
# ===================================================================


class TestComplexScenarios:
    """Checkpoint round-trips for multi-feature configurations."""

    def test_periodic_multi_material_verlet(self):
        """Periodic domain + multi-material + velocity verlet."""
        elastic = jdem.Material.create(
            "elastic", density=2500.0, young=2.0e5, poisson=0.25
        )
        frictional = jdem.Material.create(
            "elasticfrict", density=1200.0, young=8.0e4, poisson=0.35, mu=0.6
        )
        mat_table = jdem.MaterialTable.from_materials(
            [elastic, frictional],
            matcher=jdem.MaterialMatchmaker.create("harmonic"),
            fill=0.0,
        )

        state = jdem.State.create(
            pos=jnp.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]]),
            rad=jnp.array([0.5, 0.5, 0.5, 0.5]),
            mat_id=jnp.array([0, 0, 1, 1]),
        )
        system = jdem.System.create(
            state.shape,
            dt=0.001,
            domain_type="periodic",
            domain_kw=dict(box_size=10.0 * jnp.ones(2), anchor=jnp.zeros(2)),
            force_model_type="spring",
            mat_table=mat_table,
            force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
        )
        _save_step_check(state, system)

    def test_reflect_euler_gravity(self):
        """Reflect domain + Euler integrator + gravity."""
        state = jdem.State.create(
            pos=jnp.array([[5.0, 8.0]]),
            vel=jnp.array([[1.0, 0.0]]),
            rad=jnp.array([0.5]),
        )
        system = jdem.System.create(
            state.shape,
            linear_integrator_type="euler",
            rotation_integrator_type="",
            domain_type="reflect",
            domain_kw=dict(
                box_size=10.0 * jnp.ones(2),
                anchor=jnp.zeros(2),
                restitution_coefficient=0.8,
            ),
            force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
        )
        _save_step_check(state, system)

    def test_force_router_periodic_3d(self):
        """ForceRouter + periodic domain + 3D."""
        mat = jdem.Material.create("elastic", density=1.0, young=1e4, poisson=0.3)
        mat_lj = jdem.Material.create("lj", density=1.0, epsilon=1.0)

        router = jdem.ForceRouter.from_dict(
            S=2,
            mapping={
                (0, 0): jdem.ForceModel.create("spring"),
                (1, 1): jdem.ForceModel.create("wca"),
                (0, 1): jdem.ForceModel.create("spring"),
            },
        )

        state = jdem.State.create(
            pos=jnp.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [5.0, 1.0, 1.0]]),
            rad=jnp.array([1.0, 1.0, 1.0]),
            species_id=jnp.array([0, 0, 1]),
        )
        system = jdem.System.create(
            state.shape,
            force_model_type="forcerouter",
            force_model_kw=dict(table=router.table),
            domain_type="periodic",
            domain_kw=dict(box_size=10.0 * jnp.ones(3), anchor=jnp.zeros(3)),
            mat_table=jdem.MaterialTable.from_materials([mat, mat_lj]),
        )

        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        assert isinstance(system_r.force_model, jdem.ForceRouter)

    def test_clump_gravity_reflect(self):
        """Clump + gravity + reflect domain."""
        mat = jdem.Material.create("elastic", density=2.0, young=1e4, poisson=0.3)
        mat_table = jdem.MaterialTable.from_materials([mat])

        state = jdem.State.create(
            pos=jnp.array([[5.0, 1.0]]),
            rad=jnp.array([1.0]),
            mat_table=mat_table,
        )
        state.fixed = jnp.array([True])
        state = jdem.State.add_clump(
            state,
            pos=jnp.array([[5.0, 6.0], [5.0, 6.0]]),
            pos_p=jnp.array([[-0.3, 0.0], [0.3, 0.0]]),
            rad=jnp.array([0.3, 0.3]),
        )
        state = jdem.utils.compute_clump_properties(state, mat_table)

        system = jdem.System.create(
            state.shape,
            dt=1e-4,
            force_model_type="spring",
            mat_table=mat_table,
            domain_type="reflect",
            domain_kw=dict(
                box_size=10.0 * jnp.ones(2),
                anchor=jnp.zeros(2),
                restitution_coefficient=1.0,
            ),
            force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
        )

        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")

    def test_bidisperse_periodic_gd(self):
        """Bidisperse particles + periodic + gradient descent."""
        N = 10
        rad = jnp.concatenate([0.5 * jnp.ones(N // 2), 0.7 * jnp.ones(N // 2)])
        key = jax.random.PRNGKey(0)
        pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=5.0)

        mat_table = jdem.MaterialTable.from_materials(
            [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)],
            matcher=jdem.MaterialMatchmaker.create("harmonic"),
        )

        state = jdem.State.create(pos=pos, rad=rad)
        system = jdem.System.create(
            state.shape,
            dt=0.01,
            linear_integrator_type="lineargradientdescent",
            rotation_integrator_type="",
            linear_integrator_kw=dict(learning_rate=1e-3),
            domain_type="periodic",
            domain_kw=dict(box_size=5.0 * jnp.ones(2), anchor=jnp.zeros(2)),
            mat_table=mat_table,
        )
        _save_step_check(state, system)

    def test_fixed_mixed_species_gravity(self):
        """Fixed + species_id + mat_id + gravity."""
        mat = jdem.Material.create("elastic", density=1.0, young=1e4, poisson=0.3)
        mat_lj = jdem.Material.create("lj", density=1.0, epsilon=1.0)
        mat_table = jdem.MaterialTable.from_materials([mat, mat_lj])

        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [0.0, 3.0], [2.0, 5.0]]),
            rad=jnp.array([1.0, 0.5, 0.5]),
            fixed=jnp.array([True, False, False]),
            mat_id=jnp.array([0, 0, 1]),
            species_id=jnp.array([0, 0, 1]),
        )
        system = jdem.System.create(
            state.shape,
            force_model_type="lawcombiner",
            force_model_kw=dict(
                laws=(
                    jdem.ForceModel.create("spring"),
                    jdem.ForceModel.create("wca"),
                )
            ),
            mat_table=mat_table,
            force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
        )

        state, system = system.step(state, system)
        state_r, system_r = _round_trip(state, system)
        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        assert jnp.array_equal(state.fixed, state_r.fixed)
        assert jnp.array_equal(state.species_id, state_r.species_id)
        assert jnp.array_equal(state.mat_id, state_r.mat_id)


# ===================================================================
# Writer options tests
# ===================================================================


class TestWriterOptions:
    """Tests for CheckpointWriter configuration options."""

    def test_multiple_saves(self):
        """Save multiple steps, load latest."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)

        d = _tmpdir()
        with jdem.CheckpointWriter(d) as w:
            for _ in range(3):
                state, system = system.step(state, system)
                w.save(state, system)

        state_r, system_r = jdem.CheckpointLoader(d).load()
        _assert_leaves_equal(state, state_r, "state")
        assert int(system.step_count) == int(system_r.step_count)

    def test_load_specific_step(self):
        """Save multiple steps, load a specific one."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)

        d = _tmpdir()
        step_counts = []
        with jdem.CheckpointWriter(d) as w:
            for _ in range(3):
                state, system = system.step(state, system)
                step_counts.append(int(system.step_count))
                w.save(state, system)

        # Load the first saved step
        loader = jdem.CheckpointLoader(d)
        state_r, system_r = loader.load(step=step_counts[0])
        assert int(system_r.step_count) == step_counts[0]

    def test_load_missing_step_raises(self):
        """Loading a non-existent step should raise FileNotFoundError."""
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0]]),
            rad=jnp.array([1.0]),
        )
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)

        d = _tmpdir()
        with jdem.CheckpointWriter(d) as w:
            w.save(state, system)

        with pytest.raises(FileNotFoundError):
            jdem.CheckpointLoader(d).load(step=9999)

    def test_load_empty_dir_raises(self):
        """Loading from an empty directory should raise FileNotFoundError."""
        d = _tmpdir()
        os.makedirs(d, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            jdem.CheckpointLoader(d).load()
