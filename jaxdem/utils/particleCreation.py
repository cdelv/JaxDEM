# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions for creating states of GA particles (rigid/DP)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .clumps import _compute_uniform_union_properties
from .quaternion import Quaternion
from .thomsonProblemMesh import generate_thomson_mesh
from ..state import State

# overhaul all ga/dp writing codes
# use whatever meshing scheme you want
# there should be some function to set a state of clumps/dps to a packing fraction
    # define bounding sphere for each clump/dp
    # create analogous state/system of simple spheres
    # randomly initialize their positions in a box which is sized to the desired packing fraction
    # minimize analogue system energy using spring potential
    # take the positions of the analogue system and apply them to the clumps/dps
    # create a simple system with a spring potential and quasistatically compress to the desired true packing fraction (there is a difference between the bounding sphere packing fraction and the clump/dp packing fraction)
    # return the state with the final clump/dp positions and the corresponding box size

asperity_radius = 0.1
particle_radius = 0.5
particle_mass = 1.0

core_type = 'hollow'

nv = 10
N = 1
dim = 3
n_steps = 10_000
n_samples = 10_000_000

# ADD RANDOMIZED ORIENTATIONS!

def placeholder_create(
    N,
    nv,
    dim,
    particle_radius,
    asperity_radius,
    n_steps,
    particle_mass=1.0,
    n_samples=10_000_000,
    core_type='hollow',
):
    if core_type not in ['solid', 'phantom', 'hollow']:
        raise ValueError(
            f"core_type must be one of 'solid', 'phantom', 'hollow'; got {core_type!r}"
        )

    pos, _energy = generate_thomson_mesh(
        nv=nv,
        N=N,
        dim=dim,
        steps=n_steps,
    )
    if pos.ndim == 2:
        pos = pos[None, :, :]
    n_clumps, nv_eff = pos.shape[0], pos.shape[1]

    core_radius = particle_radius - asperity_radius
    pos *= core_radius
    rad = jnp.full((n_clumps, nv_eff), asperity_radius)

    if core_type in ['solid', 'phantom']:
        pos = jnp.concatenate([pos, jnp.zeros((n_clumps, 1, dim))], axis=1)
        rad = jnp.concatenate([rad, jnp.full((n_clumps, 1), core_radius)], axis=1)
        nv_eff = pos.shape[1]

    volume, com, inertia, q, pos_p = _compute_uniform_union_properties(
        pos,
        rad,
        particle_mass,
        n_samples=n_samples,
    )

    if core_type == 'phantom':
        pos = pos[:, :-1, :]
        rad = rad[:, :-1]
        pos_p = pos_p[:, :-1, :]
        nv_eff = pos.shape[1]

    # ----- flatten per-clump arrays to per-sphere for State ------------
    # Every sphere in a clump shares the rigid-body's pos_c (COM), q,
    # mass, volume, inertia, and clump_id. Only pos_p and rad vary per
    # sphere within a clump.
    total = n_clumps * nv_eff
    ang_dim = inertia.shape[-1]  # 1 in 2D, 3 in 3D

    sphere_pos_p = pos_p.reshape(total, dim)
    rad_flat = rad.reshape(total)

    pos_c = jnp.broadcast_to(com[:, None, :], (n_clumps, nv_eff, dim)).reshape(
        total, dim
    )
    q_w = jnp.broadcast_to(q[:, None, 0:1], (n_clumps, nv_eff, 1)).reshape(total, 1)
    q_xyz = jnp.broadcast_to(q[:, None, 1:4], (n_clumps, nv_eff, 3)).reshape(total, 3)
    q_state = Quaternion.create(w=q_w, xyz=q_xyz)

    volume_flat = jnp.broadcast_to(volume[:, None], (n_clumps, nv_eff)).reshape(total)
    inertia_flat = jnp.broadcast_to(
        inertia[:, None, :], (n_clumps, nv_eff, ang_dim)
    ).reshape(total, ang_dim)
    mass_flat = jnp.full((total,), particle_mass, dtype=float)
    clump_id = jnp.broadcast_to(
        jnp.arange(n_clumps, dtype=int)[:, None], (n_clumps, nv_eff)
    ).reshape(total)

    state = State.create(
        pos=pos_c,
        pos_p=sphere_pos_p,
        rad=rad_flat,
        q=q_state,
        volume=volume_flat,
        mass=mass_flat,
        inertia=inertia_flat,
        clump_id=clump_id,
    )

    return state