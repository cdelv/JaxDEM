#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Basic example script for testing the gradient-descent and FIRE minimizers.

This script:

- Creates N_systems random `State`s with particles uniformly distributed in a box.
- Uses periodic boundary conditions.
- Uses bidisperse radii (0.5 and 0.7).
- Configures a linear spring force with effective stiffness 1.0.
- Minimizes the energy of each `State` using the FIRE minimizer (or whatever minimizer is specified).
- Prints the final potential energy of each `State`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxdem as jd

jax.config.update("jax_enable_x64", True)

def main():
    # build a set of states and systems using vmap
    N_systems = 10
    N = 10
    phi = 0.4
    dim = 2
    e_int = 1.0
    dt = 1e-2

    def build_microstate(i):
        # assign bidisperse radii
        rad = jnp.ones(N)
        rad = rad.at[: N // 2].set(0.5)
        rad = rad.at[N // 2:].set(0.7)
        
        # set the box size for the packing fraction and the radii
        volume = jnp.sum((jnp.pi ** (dim / 2) / jax.scipy.special.gamma(dim / 2 + 1)) * rad ** dim)
        L = (volume / phi) ** (1 / dim)
        box_size = jnp.ones(dim) * L

        # create microstate
        key = jax.random.PRNGKey(i)
        pos = jax.random.uniform(key, (N, dim), minval=0.0, maxval=L)
        mass = jnp.ones(N)
        mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
        matcher = jd.MaterialMatchmaker.create("harmonic")
        mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
        
        # create system and state
        state = jd.State.create(pos=pos, rad=rad, mass=mass, volume=volume)
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
        return state, system

    state, system = jax.vmap(build_microstate)(jnp.arange(N_systems))

    n_steps = 100000
    state, system, steps, final_pe = jax.vmap(lambda st, sys: jd.minimizers.minimize(st, sys, max_steps=n_steps, pe_tol=1e-16, pe_diff_tol=1e-16, initialize=True))(state, system)
    print(final_pe)
    print(steps)

if __name__ == "__main__":
    main()
