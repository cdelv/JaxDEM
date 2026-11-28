# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Geometric-asperity deformable particles: step-by-step pipeline
==================================================================

Build and compress a 2D packing of deformable particles (DPs) stage by stage,
extending the workflow in :doc:`clump_construction_pipeline` with a bonded
:class:`~jaxdem.bonded_forces.deformable_particle.DeformableParticleModel`:

1. :func:`~jaxdem.utils.particle_creation.create_ga_state` with ``particle_type="dp"``
2. :func:`~jaxdem.utils.particle_creation.distribute_bodies`
3. :func:`~jaxdem.utils.particle_creation.create_dp_container` on the **placed
   rest state** (keep this container fixed so reference geometry is never reset)
4. :func:`~jaxdem.utils.packing_utils.quasistatic_compress_to_packing_fraction`
   with ``bonded_force_model=container`` active during relaxation
5. Swap to Verlet dynamics with rotation disabled
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
from jaxdem.utils.particle_creation import (
    create_dp_container,
    create_ga_state,
    distribute_bodies,
)

# %%
# 1) Create and Place Deformable Bodies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Unlike rigid clumps, DP nodes carry equal shares ``M_I / n_I`` of body mass
# and reference union volume, move independently, and use ``bond_id`` to mask
# intra-body non-bonded contacts. Here we use ``core_type="hollow"`` (set
# ``core_type="solid"`` to include a central node braced with ``interior_edges="fan"``):

N = 4
nv = 6
dim = 2
initial_phi_bb = 0.15
target_phi = 0.06  # nominal sphere-union fraction (enclosed area is ~3-4x larger for hollow DPs)
seed = 0

state = create_ga_state(
    N=N,
    nv=nv,
    dim=dim,
    particle_radius=0.5,
    asperity_radius=0.1,
    particle_type="dp",
    core_type="hollow",
    seed=seed,
    mesh_type="fibonacci",
)
state, box_size = distribute_bodies(
    state,
    phi=initial_phi_bb,
    domain_type="periodic",
    seed=seed,
    randomize_orientation=True,
)
print(f"Placed {N} DPs ({state.N} vertices) in box={np.round(box_size, 3)}")

# %%
# 2) Build the Bonded DP Container on the Placed Rest State
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Build ``container`` once right after placement so ``initial_edge_lengths``,
# ``initial_element_measures``, and ``initial_body_contents`` record the
# stress-free reference shape. Rebuilding the container after compression would
# overwrite the reference shape and erase stored elastic strain:

container = create_dp_container(
    state,
    em=1.0,
    ec=2.0,
    eb=0.1,
    el=0.5,
    gamma=None,
    interior_edges="fan",
)
print(
    f"Container: {type(container).__name__} | "
    f"elements={container.elements.shape}, edges={container.edges.shape}"
)

# %%
# 3) Quasistatic Compression with DP Bonds Active
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# During each compression step, box rescaling translates each body's centroid
# affinely without distorting its internal mesh, and FIRE relaxes both contact
# and bonded forces together:

mats = [jdem.Material.create("elastic", young=2.0, poisson=0.5, density=1.0)]
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
    bonded_force_model=container,
)
print(f"Nominal phi before compression: {float(compute_packing_fraction(state, fire_system)):.4f}")

compression = quasistatic_compress_to_packing_fraction(
    state,
    fire_system,
    target_phi=target_phi,
    step=0.01,
    force_tol=1e-8,
    max_n_min_steps_per_outer=2_000,
    max_n_outer_steps=15,
)
if compression.reason != CompressionReason.TARGET_REACHED:
    raise RuntimeError(
        f"Compression stopped: {compression.reason.name} "
        f"(minimizer: {compression.minimizer_reason.name})"
    )
state, fire_system, final_phi, final_pe = compression
_, _, contact_pe = fire_system.collider.compute_potential_energy(state, fire_system)
bonded_pe = fire_system.force_manager.compute_potential_energy(state, fire_system)
print(
    f"After compression: nominal phi={float(final_phi):.4f} | "
    f"contact PE={float(contact_pe):.3e}, bonded PE={float(bonded_pe):.3e}"
)

# %%
# 4) Switch to Verlet Dynamics (No Rigid Rotation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Keep the post-compression box and ``bonded_force_model`` intact, and disable
# rigid-body rotation since DP vertices translate independently:

sim_system = dataclasses.replace(
    fire_system,
    linear_integrator=LinearIntegrator.create("verlet"),
    rotation_integrator=RotationIntegrator.create(""),
    dt=jnp.asarray(1e-3, dtype=float),
)
state, sim_system = jdem.System.initialize(state, sim_system)
state, sim_system = jdem.System.step(state, sim_system, n=10)
sim_system.check_overflow()
print(f"10 Verlet steps completed (t = {float(sim_system.time):.3f}).")
