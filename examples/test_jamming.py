#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special
import dataclasses
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

    particle_volume = jnp.sum(
        (jnp.pi ** (state.dim / 2) / jax.scipy.special.gamma(state.dim / 2 + 1))
        * state.rad**state.dim
    )

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

    # minimization parameters
    n_steps = 10000
    pe_tol = 1e-16
    pe_diff_tol = 1e-16

    # jamming parameters
    n_jamming_steps = 1000
    packing_fraction_tolerance = 1e-6
    packing_fraction_increment = 1e-3

    # initial minimization
    state, system, steps, final_pe = jd.minimizers.minimize(state, system, max_steps=n_steps, pe_tol=pe_tol, pe_diff_tol=pe_diff_tol, initialize=True)
    if final_pe > pe_tol:
        raise ValueError("Initial state is jammed")
    
    packing_fraction = particle_volume / jnp.prod(system.domain.box_size)

    # jamming loop
    dim = state.dim
    packing_fraction_low = packing_fraction
    packing_fraction_high = -1.0
    last_unjammed_state = state
    last_unjammed_system = system
    jamming_iteration = 0
    while jamming_iteration < n_jamming_steps:
        jamming_iteration += 1
        state, system, steps, final_pe = jd.minimizers.minimize(state, system, max_steps=n_steps, pe_tol=pe_tol, pe_diff_tol=pe_diff_tol, initialize=False)
        if final_pe > pe_tol:  # jammed
            packing_fraction_high = packing_fraction
            packing_fraction = (packing_fraction_high + packing_fraction_low) / 2.0
            # revert to last unjammed state
            state = last_unjammed_state
            system = last_unjammed_system
            print(f"Jammed on iteration {jamming_iteration} with packing fraction {packing_fraction} and PE {final_pe}")
        else:  # unjammed
            # save the last unjammed state and system
            last_unjammed_state = state
            last_unjammed_system = system
            packing_fraction_low = packing_fraction
            if packing_fraction_high > 0:  # if we have found a jammed state, bisection search
                packing_fraction = (packing_fraction_high + packing_fraction_low) / 2.0
            else:  # increment packing fraction
                packing_fraction += packing_fraction_increment
            print(f"Unjammed on iteration {jamming_iteration} with packing fraction {packing_fraction} and PE {final_pe}")

        if (abs(packing_fraction_high / packing_fraction_low - 1) < packing_fraction_tolerance and packing_fraction_high > 0):
            print("CONVERGED")
            break

        new_box_size = (particle_volume / packing_fraction) ** (1 / dim)
        scale_factor = new_box_size / system.domain.box_size
        new_domain = dataclasses.replace(system.domain, box_size=jnp.ones(dim) * new_box_size)
        system = dataclasses.replace(system, domain=new_domain)
        state = dataclasses.replace(state, pos_c=state.pos_c * scale_factor)

if __name__ == "__main__":
    main()
