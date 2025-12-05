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
    method = "linearfire"

    if method == "lineargradientdescent":
        linear_integrator_kw = dict(learning_rate=1e-1)
    elif method == "linearfire":
        linear_integrator_kw = dict(
            alpha_init=0.1,
            f_inc=1.1,
            f_dec=0.5,
            f_alpha=0.99,
            N_min=5,
            N_bad_max=10,
            dt_max_scale=10.0,
            dt_min_scale=1e-3,
        )
    else:
        raise ValueError(f"Invalid method: {method}")

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type=method,
        linear_integrator_kw=linear_integrator_kw,
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        # collider_type="celllist",
        # collider_kw=dict(state=state),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
            anchor=box_size * 0.0,
        ),
    )

    potential_energy_prev = system.collider.compute_potential_energy(state, system)
    # state, system = system.collider.compute_force(state, system)
    # print(jnp.sum(state.force, axis=0), jnp.sum(jnp.linalg.norm(state.force, axis=1)))

    state, system = system.linear_integrator.initialize(state, system)

    # MAKE MINIMIZATION FUNCTIONS
    # USE A WHILE STEP METHOD TO ADD AN EARLY EXIT CRITERION FOR THE MINIMIZATION FUNCTIONS
    # MAKE A JAMMING ALGORITHM

    # run a short minimization
    n_steps = 10000
    state, system = system.step(state, system, n=n_steps)

    # get the new energy to compare to the old
    potential_energy = system.collider.compute_potential_energy(state, system)
    # state, system = system.collider.compute_force(state, system)
    # print(jnp.sum(state.force, axis=0), jnp.sum(jnp.linalg.norm(state.force, axis=1)))

    print(jnp.sum(potential_energy_prev), jnp.sum(potential_energy))

if __name__ == "__main__":
    main()
