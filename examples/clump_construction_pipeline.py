# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Rigid geometric-asperity clumps: step-by-step pipeline
=========================================================

Build a relaxed 3D packing of rigid geometric-asperity (GA) clumps stage by
stage using ``jaxdem.utils.particle_creation`` and
``jaxdem.utils.packing_utils``:

1. :func:`~jaxdem.utils.particle_creation.create_ga_state` — build ``N``
   identical clumps and compute their union volume, COM, and principal inertia.
2. :func:`~jaxdem.utils.particle_creation.distribute_bodies` — place
   non-overlapping bounding spheres at ``initial_phi_bb`` and randomize clump
   orientations.
3. :func:`~jaxdem.utils.packing_utils.quasistatic_compress_to_packing_fraction`
   — compress incrementally to the target true-body packing fraction ``target_phi``.
4. Swap integrators with :func:`dataclasses.replace` and run Verlet dynamics.

For a one-call wrapper that bundles all stages, see
:doc:`particle_construction`.
"""

import dataclasses

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jax.numpy as jnp
import numpy as np

import jaxdem as jdem
from jaxdem.integrators import LinearIntegrator, RotationIntegrator
from jaxdem.utils.packing_utils import (
    CompressionReason,
    compute_packing_fraction,
    quasistatic_compress_to_packing_fraction,
)
from jaxdem.utils.particle_creation import create_ga_state, distribute_bodies

# %%
# 1) Build Template Clumps
# ~~~~~~~~~~~~~~~~~~~~~~~~
# ``core_type="solid"`` includes a central core sphere (radius
# ``particle_radius - asperity_radius``) in both contact geometry and mass;
# ``"phantom"`` uses the core only for mass/inertia, and ``"hollow"`` omits it.

N = 6
nv = 6
dim = 3
initial_phi_bb = 0.20  # bounding-sphere fraction for overlap-free placement
target_phi = 0.18  # true sphere-union volume fraction after compression
seed = 0

state = create_ga_state(
    N=N,
    nv=nv,
    dim=dim,
    particle_radius=0.5,
    asperity_radius=0.1,
    particle_type="clump",
    core_type="solid",
    n_samples=2_048,
    seed=seed,
    mesh_type="fibonacci",
)
print(
    f"Built {int(state.clump_id.max()) + 1} clumps ({state.N} total spheres) | "
    f"clump 0 volume={float(state.volume[0]):.4f}, inertia={np.round(state.inertia[0], 4)}"
)

# %%
# 2) Distribute Bounding Spheres and Orient Randomly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Bounding spheres have radius :math:`R_I = \max_i(\|\mathbf{p}_{Ii}\| + a_{Ii})`,
# which is larger than the union volume; placing bounding spheres at
# ``initial_phi_bb`` therefore starts at a lower true-body ``phi``:

state, box_size = distribute_bodies(
    state,
    phi=initial_phi_bb,
    domain_type="periodic",
    seed=seed,
    randomize_orientation=True,
)
print(f"Placed at bounding-sphere phi={initial_phi_bb}: box={np.round(box_size, 3)}")

# %%
# 3) Quasistatic Compression with FIRE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build a FIRE system on the placed periodic box and compress in steps to
# ``target_phi``:

mats = [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
mat_table = jdem.MaterialTable.from_materials(
    mats, matcher=jdem.MaterialMatchmaker.create("harmonic")
)
fire_system = jdem.System.create(
    state_shape=state.shape,
    dt=1e-2,
    minimizer=jdem.minimizers.fire,
    minimizer_kw={"dt": 1e-2},
    domain_type="periodic",
    force_model_type="spring",
    collider_type="naive",
    mat_table=mat_table,
    domain_kw={"box_size": box_size},
)
print(f"True-body phi before compression: {float(compute_packing_fraction(state, fire_system)):.4f}")

compression = quasistatic_compress_to_packing_fraction(
    state,
    fire_system,
    target_phi=target_phi,
    step=0.02,
    force_tol=1e-8,
    max_n_min_steps_per_outer=2_000,
    max_n_outer_steps=16,
)
if compression.reason != CompressionReason.TARGET_REACHED:
    raise RuntimeError(
        f"Compression stopped: {compression.reason.name} "
        f"(minimizer: {compression.minimizer_reason.name})"
    )
state, fire_system, final_phi, final_pe = compression
print(
    f"After compression: phi={float(final_phi):.4f}, PE/node={float(final_pe):.3e}, "
    f"box={np.round(fire_system.domain.box_size, 3)}"
)

# %%
# 4) Switch to Verlet + Spiral Integrators for Dynamics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``dataclasses.replace`` swaps the integrators and ``dt`` while preserving the
# post-compression box size, collider, and material table:

sim_system = dataclasses.replace(
    fire_system,
    linear_integrator=LinearIntegrator.create("verlet"),
    rotation_integrator=RotationIntegrator.create("verletspiral"),
    dt=jnp.asarray(1e-3, dtype=float),
)
state, sim_system = jdem.System.initialize(state, sim_system)
state, sim_system = jdem.System.step(state, sim_system, n=10)
sim_system.check_overflow()
_, _, pe = sim_system.collider.compute_potential_energy(state, sim_system)
print(f"10 Verlet steps completed; final PE = {float(pe):.3e}")
