# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Analytical smooth-sphere contact geometry and radial force decomposition."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.surface_properties import compute_surface_properties


def _single_sphere_state(radius: float, dim: int) -> jd.State:
    return jd.State.create(
        pos=jnp.zeros((1, dim)), rad=jnp.array([radius]), mass=jnp.ones(1)
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_smooth_sphere_probe_matches_the_exact_radial_contact(dim):
    result = compute_surface_properties(
        _single_sphere_state(0.5, dim),
        _single_sphere_state(0.25, dim),
        target_overlap=0.125,
        n_points=8,
        n_orientations=1,
        n_rolls=1,
        separation_tolerance=1e-5,
    )
    np.testing.assert_allclose(result["separation"], 0.625, rtol=0, atol=2e-6)
    np.testing.assert_allclose(result["mu"], 0.0, atol=2e-6)
    np.testing.assert_allclose(result["mu_asperity"], 0.0, atol=2e-6)
    np.testing.assert_array_equal(result["n_central_contacts"], 1)
    np.testing.assert_array_equal(result["n_tracer_contacts"], 1)
    directions = np.asarray(result["approach_directions"])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=2e-6)
    if dim == 2:
        gaps = np.diff(np.unwrap(np.arctan2(directions[:, 1], directions[:, 0])))
        np.testing.assert_allclose(gaps, 2 * np.pi / 8, atol=2e-6)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("stiffness", [1e-32, 1.0, 1e32])
def test_surface_contact_and_friction_are_independent_of_force_units(dim, stiffness):
    # These force scales require float64 squared norms.
    with jax.enable_x64(True):
        central = _single_sphere_state(0.5, dim)
        tracer = _single_sphere_state(0.3, dim)
        state = jd.State.merge(central, tracer)
        material = jd.Material.create(
            "elastic", young=stiffness, poisson=0.0, density=1.0
        )
        system = jd.System.create(
            state=state,
            mat_table=jd.MaterialTable.from_materials([material]),
            domain_type="periodic",
            domain_kw={"box_size": jnp.full(dim, 8.0)},
            collider_type="naive",
        )
        result = compute_surface_properties(
            central, tracer, target_overlap=0.1, system=system, n_points=16
        )
        np.testing.assert_allclose(result["separation"], 0.7, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result["mu"], 0.0, atol=1e-14)
        np.testing.assert_allclose(result["mu_asperity"], 0.0, atol=1e-14)
        np.testing.assert_array_equal(result["n_central_contacts"], 1)
        np.testing.assert_array_equal(result["n_tracer_contacts"], 1)
