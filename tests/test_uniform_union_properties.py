from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxdem.utils.clumps import _compute_uniform_union_properties

jax.config.update("jax_enable_x64", True)


def _volume_errors_for_overlapping_spheres(n_samples_list: np.ndarray) -> np.ndarray:
    dim = 3
    radius = 1.0
    mass = 1.0
    particle_mass = jnp.asarray(mass, dtype=float)

    pos_exact = jnp.zeros((1, dim), dtype=float)
    rad_exact = jnp.full((1,), radius, dtype=float)
    volume_exact, *_ = _compute_uniform_union_properties(
        pos_exact,
        rad_exact,
        particle_mass,
        n_samples=10_000,
    )

    # Two perfectly overlapping spheres have the same union volume as one sphere,
    # but they force the Monte Carlo branch instead of the analytic single-sphere path.
    pos_mc = jnp.zeros((2, dim), dtype=float)
    rad_mc = jnp.full((2,), radius, dtype=float)

    errors = []
    for n_samples in n_samples_list:
        volume_mc, *_ = _compute_uniform_union_properties(
            pos_mc,
            rad_mc,
            particle_mass,
            n_samples=int(n_samples),
        )
        errors.append(float(jnp.abs(volume_mc - volume_exact) / volume_exact))

    return np.asarray(errors)


def test_uniform_union_volume_error_scales_like_n_to_minus_two_thirds() -> None:
    n_samples_list = np.logspace(3, 6, 4).astype(int)
    volume_rel_error = _volume_errors_for_overlapping_spheres(n_samples_list)

    slope = np.polyfit(np.log(n_samples_list), np.log(volume_rel_error), 1)[0]

    assert np.all(np.diff(volume_rel_error) < 0), volume_rel_error
    assert -0.9 < slope < -0.5, slope
