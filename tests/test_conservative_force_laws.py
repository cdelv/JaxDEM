# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Analytical force, potential energy, F = -grad(U), and cutoff tests for conservative laws."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.forces.lennardjones import LennardJones


def _pair_energy(pos: jax.Array, state: jd.State, system: jd.System) -> jax.Array:
    """Single-pair interaction energy U(pos_0, pos_1)."""
    history = jnp.zeros((1,), dtype=pos.dtype)
    return system.force_model.energy(0, 1, pos, state, system, history)


def _assert_conservative_pair(
    pos: jax.Array,
    state: jd.State,
    system: jd.System,
    expected_f01: np.ndarray,
    expected_u: float,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    history = jnp.zeros((1,), dtype=pos.dtype)
    f01, t01, _ = system.force_model.force(0, 1, pos, state, system, history)
    f10, t10, _ = system.force_model.force(1, 0, pos, state, system, history)
    f00, t00, _ = system.force_model.force(0, 0, pos, state, system, history)
    u01 = system.force_model.energy(0, 1, pos, state, system, history)
    u00 = system.force_model.energy(0, 0, pos, state, system, history)

    np.testing.assert_allclose(f01, expected_f01, rtol=rtol, atol=atol)
    np.testing.assert_allclose(f10, -f01, rtol=rtol, atol=atol)
    np.testing.assert_allclose(u01, expected_u, rtol=rtol, atol=atol)
    np.testing.assert_allclose(t01, 0.0, atol=1e-7)
    np.testing.assert_allclose(t10, 0.0, atol=1e-7)
    np.testing.assert_allclose(f00, 0.0, atol=1e-7)
    np.testing.assert_allclose(t00, 0.0, atol=1e-7)
    np.testing.assert_allclose(u00, 0.0, atol=1e-7)

    grad_u = jax.grad(_pair_energy)(pos, state, system)
    np.testing.assert_allclose(-grad_u[0], f01, rtol=rtol, atol=atol)
    np.testing.assert_allclose(-grad_u[1], f10, rtol=rtol, atol=atol)


def _make_lj_system(
    pos: jax.Array,
    rad: jax.Array,
    force_model_type: str,
    *,
    force_model_kw: dict | None = None,
) -> tuple[jd.State, jd.System, float]:
    mat0 = jd.Material.create("lj", density=1.0, epsilon=1.5)
    mat1 = jd.Material.create("lj", density=1.0, epsilon=2.4)
    matcher = jd.MaterialMatchmaker.create("linear")
    mat_table = jd.MaterialTable.from_materials([mat0, mat1], matcher=matcher)
    state = jd.State.create(pos=pos, rad=rad, mat_id=jnp.array([0, 1], dtype=int))
    system = jd.System.create(
        state_shape=state.shape,
        force_model_type=force_model_type,
        force_model_kw=force_model_kw or {},
        mat_table=mat_table,
    )
    eps_eff = 0.5 * (1.5 + 2.4)
    return state, system, eps_eff


@pytest.mark.parametrize("dim", [2, 3])
def test_hertzian_force_analytical_conservative_and_cutoff(dim):
    r0, r1 = 0.4, 0.6
    e0, nu0 = 100.0, 0.25
    e1, nu1 = 250.0, 0.35
    mat0 = jd.Material.create("elastic", density=1.0, young=e0, poisson=nu0)
    mat1 = jd.Material.create("elastic", density=1.0, young=e1, poisson=nu1)
    mat_table = jd.MaterialTable.from_materials([mat0, mat1])

    direction = np.array([0.6, 0.8]) if dim == 2 else np.array([2.0, -1.0, 2.0]) / 3.0
    r_overlap = 0.82
    pos = jnp.asarray([np.zeros(dim), -r_overlap * direction], dtype=float)
    state = jd.State.create(
        pos=pos, rad=jnp.array([r0, r1]), mat_id=jnp.array([0, 1], dtype=int)
    )
    system = jd.System.create(
        state_shape=state.shape, force_model_type="hertz", mat_table=mat_table
    )

    r_star = (r0 * r1) / (r0 + r1)
    e_star = 1.0 / ((1.0 - nu0**2) / e0 + (1.0 - nu1**2) / e1)
    k_hertz = (4.0 / 3.0) * e_star * np.sqrt(r_star)
    delta = (r0 + r1) - r_overlap
    expected_f01 = k_hertz * (delta**1.5) * direction
    expected_u = 0.4 * k_hertz * (delta**2.5)
    _assert_conservative_pair(pos, state, system, expected_f01, expected_u)

    # Zero force and energy at contact and beyond.
    history = jnp.zeros((1,), dtype=float)
    for sep in (r0 + r1, r0 + r1 + 0.25):
        pos_sep = jnp.asarray([np.zeros(dim), -sep * direction], dtype=float)
        f, _, _ = system.force_model.force(0, 1, pos_sep, state, system, history)
        u = system.force_model.energy(0, 1, pos_sep, state, system, history)
        np.testing.assert_allclose(f, 0.0, atol=1e-7)
        np.testing.assert_allclose(u, 0.0, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3])
def test_lennard_jones_repulsive_attractive_cutoff_and_validation(dim):
    rad = jnp.array([0.45, 0.55], dtype=float)
    sig = 1.0
    cutoff_ratio = 2.0
    direction = np.array([0.8, -0.6]) if dim == 2 else np.array([1.0, 2.0, 2.0]) / 3.0
    pos_init = jnp.asarray([np.zeros(dim), -0.95 * direction], dtype=float)
    state, system, eps = _make_lj_system(
        pos_init,
        rad,
        "lennardjones",
        force_model_kw={"cutoff_ratio": cutoff_ratio},
    )

    u_rc = 4.0 * eps * ((1.0 / cutoff_ratio) ** 12 - (1.0 / cutoff_ratio) ** 6)

    # Test both repulsive (r < 2^(1/6)*sig) and attractive (2^(1/6)*sig < r < rc) regimes.
    for r in (0.95, 1.45):
        pos = jnp.asarray([np.zeros(dim), -r * direction], dtype=float)
        sr6 = (sig / r) ** 6
        sr12 = sr6 * sr6
        expected_u = 4.0 * eps * (sr12 - sr6) - u_rc
        expected_f_mag = 24.0 * eps * (2.0 * sr12 - sr6) / r
        expected_f01 = expected_f_mag * direction
        if r < (2.0 ** (1.0 / 6.0)) * sig:
            assert expected_f_mag > 0.0
        else:
            assert expected_f_mag < 0.0
        _assert_conservative_pair(pos, state, system, expected_f01, expected_u)

    # Shifted energy approaches 0 smoothly at r -> rc-, and both F and U are 0 for r >= rc.
    history = jnp.zeros((1,), dtype=float)
    rc = cutoff_ratio * sig
    pos_near_rc = jnp.asarray([np.zeros(dim), -(rc - 1e-4) * direction], dtype=float)
    u_near = system.force_model.energy(0, 1, pos_near_rc, state, system, history)
    np.testing.assert_allclose(u_near, 0.0, atol=5e-4)

    # At exact axis-aligned cutoff r == rc and beyond, both F and U are zero.
    axis = np.zeros(dim)
    axis[0] = 1.0
    for sep in (rc, rc + 0.3):
        pos_out = jnp.asarray([np.zeros(dim), -sep * axis], dtype=float)
        f, _, _ = system.force_model.force(0, 1, pos_out, state, system, history)
        u = system.force_model.energy(0, 1, pos_out, state, system, history)
        np.testing.assert_allclose(f, 0.0, atol=1e-7)
        np.testing.assert_allclose(u, 0.0, atol=1e-7)

    for bad_cutoff in (0.0, -1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="cutoff_ratio"):
            LennardJones.Create(cutoff_ratio=bad_cutoff)


@pytest.mark.parametrize("dim", [2, 3])
def test_wca_analytical_conservative_and_smooth_cutoff(dim):
    rad = jnp.array([0.4, 0.6], dtype=float)
    sig = 1.0
    rc = (2.0 ** (1.0 / 6.0)) * sig
    direction = np.array([-0.6, 0.8]) if dim == 2 else np.array([2.0, 2.0, -1.0]) / 3.0
    r = 0.96
    pos = jnp.asarray([np.zeros(dim), -r * direction], dtype=float)
    state, system, eps = _make_lj_system(pos, rad, "wca")

    sr6 = (sig / r) ** 6
    sr12 = sr6 * sr6
    expected_u = 4.0 * eps * (sr12 - sr6) + eps
    expected_f01 = (24.0 * eps * (2.0 * sr12 - sr6) / r) * direction
    _assert_conservative_pair(pos, state, system, expected_f01, expected_u)

    # Both U and F approach 0 continuously at r -> 2^(1/6) * sig, and vanish for r >= rc.
    history = jnp.zeros((1,), dtype=float)
    pos_near = jnp.asarray([np.zeros(dim), -(rc - 1e-4) * direction], dtype=float)
    f_near, _, _ = system.force_model.force(0, 1, pos_near, state, system, history)
    u_near = system.force_model.energy(0, 1, pos_near, state, system, history)
    np.testing.assert_allclose(f_near, 0.0, atol=1e-2)
    np.testing.assert_allclose(u_near, 0.0, atol=1e-5)

    for sep in (rc, rc + 0.2):
        pos_out = jnp.asarray([np.zeros(dim), -sep * direction], dtype=float)
        f, _, _ = system.force_model.force(0, 1, pos_out, state, system, history)
        u = system.force_model.energy(0, 1, pos_out, state, system, history)
        np.testing.assert_allclose(f, 0.0, atol=1e-7)
        np.testing.assert_allclose(u, 0.0, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3])
def test_wca_shifted_analytical_conservative_and_contact_cutoff(dim):
    rad = jnp.array([0.35, 0.65], dtype=float)
    sig = 1.0
    direction = np.array([0.8, 0.6]) if dim == 2 else np.array([-1.0, 2.0, 2.0]) / 3.0
    r = 0.88
    pos = jnp.asarray([np.zeros(dim), -r * direction], dtype=float)
    state, system, eps = _make_lj_system(pos, rad, "wca_shifted")

    sr6 = (sig / r) ** 6
    sr12 = sr6 * sr6
    expected_u = 4.0 * eps * (sr12 - sr6) + (r - sig) * (24.0 * eps / sig)
    expected_f_mag = 24.0 * eps * (2.0 * sr12 - sr6) / r - (24.0 * eps / sig)
    expected_f01 = expected_f_mag * direction
    _assert_conservative_pair(pos, state, system, expected_f01, expected_u)

    # Force-shifted WCA has U(sigma) = 0 and F(sigma) = 0 with smooth turn-on at contact.
    history = jnp.zeros((1,), dtype=float)
    pos_near = jnp.asarray([np.zeros(dim), -(sig - 1e-5) * direction], dtype=float)
    f_near, _, _ = system.force_model.force(0, 1, pos_near, state, system, history)
    u_near = system.force_model.energy(0, 1, pos_near, state, system, history)
    np.testing.assert_allclose(f_near, 0.0, atol=1e-2)
    np.testing.assert_allclose(u_near, 0.0, atol=1e-6)

    axis = np.zeros(dim)
    axis[0] = 1.0
    for sep in (sig, sig + 0.2):
        pos_out = jnp.asarray([np.zeros(dim), -sep * axis], dtype=float)
        f, _, _ = system.force_model.force(0, 1, pos_out, state, system, history)
        u = system.force_model.energy(0, 1, pos_out, state, system, history)
        np.testing.assert_allclose(f, 0.0, atol=1e-7)
        np.testing.assert_allclose(u, 0.0, atol=1e-7)
