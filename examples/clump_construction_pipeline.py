"""Rigid geometric-asperity clumps: step-by-step pipeline
=========================================================

This example shows the full pipeline to build a mechanically stable
packing of rigid geometric-asperity (GA) clumps at a chosen true-body
packing fraction. It uses the primitives in
``jaxdem.utils.particle_creation`` and ``jaxdem.utils.packing_utils``.

The flow is:

1. :func:`~jaxdem.utils.particle_creation.create_ga_state` — build a
   state of ``N`` identical clumps. Each clump has ``nv`` surface
   asperities on a sphere or ellipsoid surface, plus an optional solid
   core. The function places the asperities with the generalized
   Thomson problem. Monte-Carlo integration over the union volume gives
   the per-clump volume, COM, principal inertia, and principal-axis
   quaternion. The function stores these on the returned
   :class:`~jaxdem.State`.
2. :func:`~jaxdem.utils.particle_creation.distribute_bodies` — build one
   bounding sphere per clump (radius = ``max(|node - centroid| + rad)``)
   and place the spheres uniformly in a box sized for an initial
   bounding-sphere packing fraction. It then FIRE-minimizes the analogue
   sphere system, moves each clump's centroid to the minimized location,
   and gives each clump a random orientation.
3. :func:`~jaxdem.utils.packing_utils.quasistatic_compress_to_packing_fraction`
   — shrink the box in steps toward the target *true-body* packing
   fraction, and minimize after each step.

At the end, a short time-integration with the Verlet + spiral
integrators confirms the resulting state and system are ready to simulate.
"""

# %%
# Imports
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jaxdem as jdem
from jaxdem.utils.packing_utils import (
    compute_packing_fraction,
    quasistatic_compress_to_packing_fraction,
)
from jaxdem.utils.particle_creation import create_ga_state, distribute_bodies

# %%
# Parameters
# -----------------
# 12 identical clumps in 3D, each with 12 surface asperities on a sphere
# of radius 0.5 and a solid core, so the rigid-body property calculation
# captures the enclosed volume.

N = 12
nv = 12
dim = 3
particle_radius = 0.5
asperity_radius = 0.1
# Two distinct packing fractions: the bounding-sphere fraction for the
# initial random placement, and the *true-body* fraction we compress to. The
# one-call :func:`~jaxdem.utils.particle_creation.build_ga_system` exposes the
# former as its ``initial_phi_bb`` parameter (default 0.3).
initial_phi_bb = 0.2  # bounding-sphere packing fraction at placement
target_phi = 0.35  # *true-body* packing fraction after compression
seed = 0

# %%
# 1) Build the template clumps
# -----------------------------
# ``create_ga_state`` does most of the work: Thomson-mesh asperity
# positions, Monte-Carlo union-volume / COM / inertia, and an aligned
# principal-axis quaternion. For ``core_type='solid'`` it adds a central
# sphere of radius ``core_radius = particle_radius - asperity_radius``
# and keeps it in the final state. For ``'phantom'`` the core only feeds
# the property computation, and the function removes it from the state.
# For ``'hollow'`` it adds no core.
state = create_ga_state(
    N=N,
    nv=nv,
    dim=dim,
    particle_radius=particle_radius,
    asperity_radius=asperity_radius,
    particle_type="clump",
    core_type="solid",
    n_samples=100_000,
    seed=seed,
    mesh_kwargs={"steps": 1_000},
)
print(
    f"clumps built: total nodes = {state.N}, clump_ids cover {int(state.clump_id.max()) + 1} bodies"
)

# %%
# 2) Place each clump's bounding sphere at the initial packing fraction
# ---------------------------------------------------------------------
# ``distribute_bodies`` places each body's bounding sphere (center = COM,
# radius = furthest sphere outer edge) uniformly at random in a periodic
# box. It sizes the box so that total bounding-sphere volume / box volume
# = ``initial_phi_bb``. It FIRE-minimizes the analogue sphere system to
# remove overlaps, and gives each clump a random uniform rotation.
state, box_size = distribute_bodies(
    state,
    phi=initial_phi_bb,
    domain_type="periodic",
    seed=seed,
    randomize_orientation=True,
)
print(f"placed at bounding-sphere phi={initial_phi_bb}: box = {np.asarray(box_size)}")

# %%
# 3) Build a FIRE-based system for compression
# --------------------------------------------
# The compression routine expects a system with a FIRE minimizer. We use
# the naive collider for simplicity — the neighbor list would also work.

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

phi_bb = float(compute_packing_fraction(state, fire_system))
print(
    f"before compression: true-body phi = {phi_bb:.4f} (bounding-sphere target was {initial_phi_bb})"
)

# %%
# 4) Quasistatic compression to the target true-body phi
# -------------------------------------------------------
# The compression steps by at most ``step`` in phi per outer iteration.
# It calls ``scale_to_packing_fraction`` then ``minimize`` each time, and
# truncates the final step so it hits the target phi exactly.
state, fire_system, final_phi, final_pe = quasistatic_compress_to_packing_fraction(
    state,
    fire_system,
    target_phi=target_phi,
    step=5e-3,
    max_n_min_steps_per_outer=100_000,
)
print(f"after compression:  phi = {float(final_phi):.4f}  PE = {float(final_pe):.3e}")
print(f"box = {np.asarray(fire_system.domain.box_size)}")

# %%
# 5) Switch to Verlet integrators for dynamics and run a short rollout
# --------------------------------------------------------------------
# ``dataclasses.replace`` returns a copy of the FIRE system with
# new integrators and ``dt``. Every other component stays, including
# the domain with its *post-compression* box size, the material
# table, and the collider, so no manual rebuild is needed. Here we use
# ``verlet`` linear + ``verletspiral`` rotation for a standard Newtonian
# rollout.

import dataclasses
import jax.numpy as jnp
from jaxdem.integrators import LinearIntegrator, RotationIntegrator

sim_system = dataclasses.replace(
    fire_system,
    linear_integrator=LinearIntegrator.create("verlet"),
    rotation_integrator=RotationIntegrator.create("verletspiral"),
    dt=jnp.asarray(1e-3, dtype=float),
)

for k in range(100):
    state, sim_system = sim_system.step(state, sim_system)
_, _, pe = sim_system.collider.compute_potential_energy(state, sim_system)
print(f"100 Verlet steps completed; final PE ~ {float(pe):.3e}")
