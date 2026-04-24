"""Analytical tests for :func:`compute_clump_pair_friction`.

We verify two configurations where the friction coefficient can be read off
from geometry alone:

1. **Radial contact.** Two single-sphere clumps overlapping along a single
   axis. The center-to-center direction of the clumps coincides with the
   sphere-sphere contact line, so the spring contact force is purely normal
   and :math:`\\mu = 0`.

2. **Off-axis contact.** A two-sphere clump contacting a single-sphere
   clump at a point offset from the clump COM-to-COM axis. The total
   contact force is a single radial sphere-sphere force whose direction
   deviates from the COM-to-COM axis by a known angle, giving a closed-
   form :math:`\\mu`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jd
from jaxdem.utils.contacts import compute_clump_pair_friction

jax.config.update("jax_enable_x64", True)


def _build_system(state: jd.State, box_size: float) -> jd.System:
    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    mat_table = jd.MaterialTable.from_materials(
        mats, matcher=jd.MaterialMatchmaker.create("harmonic")
    )
    return jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={"box_size": jnp.ones(state.dim) * box_size},
    )


def test_radial_contact_gives_zero_friction() -> None:
    """Two single-sphere clumps aligned: contact is purely radial so mu = 0."""
    # Two spheres in 2D, each its own clump.
    pos = jnp.array([[0.0, 0.0], [0.8, 0.0]])
    rad = jnp.array([0.5, 0.5])
    state = jd.State.create(
        pos=pos, rad=rad, mass=jnp.ones(2), clump_id=jnp.array([0, 1])
    )
    system = _build_system(state, box_size=10.0)

    state, system, F_clumps, mu, contact_mask = compute_clump_pair_friction(
        state, system
    )

    # Expect a single clump pair (0, 1) in contact; mu there is 0.
    assert bool(contact_mask[0, 1])
    assert bool(contact_mask[1, 0])
    assert not bool(contact_mask[0, 0])
    assert not bool(contact_mask[1, 1])

    # F_clumps is antisymmetric and the force should be repulsive along -x.
    F_01 = np.asarray(F_clumps[0, 1])
    F_10 = np.asarray(F_clumps[1, 0])
    np.testing.assert_allclose(F_01, -F_10, atol=1e-14)
    # Force on clump 0 from clump 1 should push 0 toward -x (away from 1).
    assert F_01[0] < 0.0
    np.testing.assert_allclose(F_01[1], 0.0, atol=1e-14)

    # mu == 0, and symmetric.
    assert float(mu[0, 1]) == 0.0
    assert float(mu[1, 0]) == 0.0


def test_offset_contact_gives_analytical_friction() -> None:
    """Two-sphere clump contacting a single-sphere clump off-axis.

    Geometry::

        clump 0 (COM at (1, 0)):
            sphere A at (0, 0)   pos_p = (-1, 0)
            sphere B at (2, 0)   pos_p = (+1, 0)

        clump 1 (COM at (2, 0.8)):
            sphere C at (2, 0.8) pos_p = (0, 0)

    Only B-C are in contact (distance 0.8, R = 1.0, overlap 0.2).
    The sphere-sphere force on B is purely in the ``-y`` direction.

    With ``n_hat_{01}`` pointing from COM_0=(1,0) to COM_1=(2,0.8),
    the decomposition gives
        |F_n| / |F| = 0.8 / sqrt(1.64)
        |F_t| / |F| = 1.0 / sqrt(1.64)
        mu = (|F_t| / |F_n|) = 1.0 / 0.8 = 1.25
    """
    # Two clumps: [A, B, C]; clump_ids [0, 0, 1].
    pos_c = jnp.array(
        [
            [1.0, 0.0],  # A: shared clump-0 COM
            [1.0, 0.0],  # B: shared clump-0 COM
            [2.0, 0.8],  # C: clump-1 COM
        ]
    )
    pos_p = jnp.array(
        [
            [-1.0, 0.0],  # A body-frame offset
            [1.0, 0.0],  # B body-frame offset
            [0.0, 0.0],  # C (single-sphere clump)
        ]
    )
    rad = jnp.full((3,), 0.5)
    # State.create takes pos as pos_c; we supply pos_p explicitly.
    state = jd.State.create(
        pos=pos_c,
        pos_p=pos_p,
        rad=rad,
        mass=jnp.ones(3),
        clump_id=jnp.array([0, 0, 1]),
    )
    system = _build_system(state, box_size=10.0)

    # Sanity: reconstruct world-frame positions from pos_c + q.rotate(pos_p).
    world_pos = np.asarray(state.pos)
    np.testing.assert_allclose(world_pos[0], [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(world_pos[1], [2.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(world_pos[2], [2.0, 0.8], atol=1e-14)

    state, system, F_clumps, mu, contact_mask = compute_clump_pair_friction(
        state, system
    )

    # Exactly one clump pair in contact.
    assert bool(contact_mask[0, 1])
    assert bool(contact_mask[1, 0])
    assert int(contact_mask.sum()) == 2  # two symmetric entries

    # The only contributing sphere-pair force on clump 0 points in -y.
    F_01 = np.asarray(F_clumps[0, 1])
    np.testing.assert_allclose(F_01[0], 0.0, atol=1e-12)
    assert F_01[1] < 0.0  # sign: repulsion pushes B (and hence clump 0) toward -y

    # mu = 1 / 0.8 = 1.25 exactly, independent of k and overlap magnitude.
    expected_mu = 1.0 / 0.8
    np.testing.assert_allclose(float(mu[0, 1]), expected_mu, rtol=1e-12)
    np.testing.assert_allclose(float(mu[1, 0]), expected_mu, rtol=1e-12)
