#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Basic example script for testing the gradient-descent and FIRE minimizers.

This script:

- Creates a random `State` with particles uniformly distributed in a box.
- Uses periodic boundary conditions.
- Uses bidisperse radii (0.5 and 0.7).
- Configures a linear spring force with effective stiffness 1.0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

import jaxdem as jd

jax.config.update("jax_enable_x64", True)

def main():
    # macrostate
    N = 100
    phi = 0.5
    dim = 2
    e_int = 1.0

    # assign bidisperse radii
    rad = jnp.ones(N)
    rad = rad.at[: N // 2].set(0.5)
    rad = rad.at[N // 2:].set(0.7)

    # set the box size for the packing fraction and the radii
    L = (jnp.pi * jnp.sum(rad ** 2) / phi) ** (1 / dim)
    box_size = jnp.ones(dim) * L

    # create microstate
    key = jax.random.PRNGKey(0)
    pos = jax.random.uniform(key, (N, dim), minval=0.0, maxval=L)
    vel = jnp.zeros((N, dim))
    mass = jnp.ones(N)
    state = jd.State.create(pos=pos, vel=vel, rad=rad, mass=mass)
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    dt = 1e-2
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="linearfire",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )

    n_steps = 10000
    state, system, steps, final_pe = jd.minimizers.minimize(state, system, max_steps=n_steps, pe_tol=1e-16, pe_diff_tol=1e-16, initialize=True)

    print(final_pe)

if __name__ == "__main__":
    main()
