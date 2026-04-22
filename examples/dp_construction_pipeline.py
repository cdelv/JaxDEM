# %%
"""Geometric-asperity deformable particles: step-by-step pipeline
==================================================================

This example builds a packing of deformable particles (DPs) whose nodes
are the surface asperities of a geometric-asperity body. Each DP is a
soft body held together by a :class:`~jaxdem.bonded_forces.DeformableParticleModel`
(or a plastic variant) that applies the standard DP energies —
area/volume (``ec``), element-measure (``em``), edges (``el``), bending
(``eb``), and surface tension (``gamma``).

Pipeline:

1. :func:`create_ga_state` with ``particle_type='dp'`` builds a State
   where each node carries ``mass / nv`` and ``union_volume / nv``, and
   shares a ``bond_id`` with every other node in its body.
2. :func:`distribute_bodies` places each body's bounding sphere at the
   target initial packing fraction and applies a uniform-random
   per-body rotation (the rotation physically rotates the DP nodes
   around their centroid — confirmed in the tests).
3. :func:`create_dp_container` builds the bonded-force container on the
   placed state. Surface / interior nodes are auto-detected by a
   per-body convex-hull test (:func:`ga_surface_mask`), so a body
   with a solid core gets an interior node hooked up via "fan"-style
   struts; a hollow body has every node on the surface with no extra
   edges.
4. :func:`quasistatic_compress_to_packing_fraction` compresses to the
   target true-body packing fraction while the DP bonds keep each body
   self-consistent.
5. Build a Verlet system (with the container as ``bonded_force_model``)
   and run a few time steps.
"""

# %%
# Imports
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jaxdem as jd
from jaxdem.utils.particleCreation import (
    create_ga_state,
    create_dp_container,
    distribute_bodies,
)
from jaxdem.utils.packingUtils import (
    compute_packing_fraction,
    quasistatic_compress_to_packing_fraction,
)


# %%
# Parameters
# -----------------
# 6 deformable particles in 3D, 12 surface asperities each. ``core_type``
# is 'hollow' here — try 'solid' to get a central node connected to every
# surface node via ``interior_edges='fan'``.

N = 6
nv = 12
dim = 3
particle_radius = 0.5
asperity_radius = 0.1
initial_phi_bb = 0.2
target_phi = 0.14  # true-body phi — DP hulls are less dense than bounding spheres
seed = 0


# %%
# 1) Build the DP state
# ---------------------
# DP nodes get equal shares of mass and union volume. ``bond_id`` groups
# them by body. No rigid-body frame is applied; each node's ``pos_c`` is
# its world-frame position.

state = create_ga_state(
    N=N,
    nv=nv,
    dim=dim,
    particle_radius=particle_radius,
    asperity_radius=asperity_radius,
    n_steps=1_000,
    particle_type="dp",
    core_type="hollow",
    n_samples=50_000,
    seed=seed,
)
print(f"DP state: N={state.N} nodes, {int(state.bond_id.max()) + 1} bodies")

# %%
# 2) Place + orient bodies
state, box_size = distribute_bodies(
    state,
    phi=initial_phi_bb,
    domain_type="periodic",
    seed=seed,
    randomize_orientation=True,
)
print(f"placed at bounding-sphere phi={initial_phi_bb}: box = {np.asarray(box_size)}")

# %%
# 3) Build the DP container
# -------------------------
# The container is built from the *placed* state so that its
# ``initial_edge_lengths`` / ``initial_bendings`` / etc. reflect the
# actual rest configuration. ``is_surface`` is auto-detected via
# convex-hull per body; for hollow DPs every node is on the hull.
# ``interior_edges='fan'`` connects any interior (core) node to every
# surface node in its body — it's a no-op here because there are no
# interior nodes, but gets exercised for ``core_type='solid'``.

container = create_dp_container(
    state,
    em=1.0,
    ec=1.0,
    eb=0.1,  # bending is stiff for small triangles; keep this modest
    el=0.3,
    gamma=None,  # no surface tension in this example
    interior_edges="fan",
)
print(f"container: {type(container).__name__}")
print(f"   elements: {container.elements.shape if container.elements is not None else None}")
print(f"   edges:    {container.edges.shape if container.edges is not None else None}")
print(f"   adjacency:{container.element_adjacency.shape if container.element_adjacency is not None else None}")


# %%
# 4) Compress with FIRE, DP bonds active
# --------------------------------------
# Compression calls ``scale_to_packing_fraction`` which translates every
# body's centroid uniformly (internal geometry preserved) and then FIRE
# relaxes. Because the DP container is wired into the System, bonded
# forces keep each body coherent; contact forces from other bodies can
# still locally deform it.

mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
mat_table = jd.MaterialTable.from_materials(
    mats, matcher=jd.MaterialMatchmaker.create("harmonic")
)
fire_system = jd.System.create(
    state_shape=state.shape,
    dt=1e-2,
    linear_integrator_type="linearfire",
    rotation_integrator_type="",  # DPs carry no rigid-body orientation
    domain_type="periodic",
    force_model_type="spring",
    collider_type="naive",
    mat_table=mat_table,
    domain_kw={"box_size": box_size},
    bonded_force_model=container,
)

phi0 = float(compute_packing_fraction(state, fire_system))
print(f"before compression: true-body phi = {phi0:.4f}")

state, fire_system, final_phi, final_pe = quasistatic_compress_to_packing_fraction(
    state,
    fire_system,
    target_phi=target_phi,
    step=5e-3,
    max_n_min_steps_per_outer=50_000,
)
print(f"after compression:  phi = {float(final_phi):.4f}  PE = {float(final_pe):.3e}")

# %%
# 5) Short Verlet rollout (dynamics)
# ----------------------------------
sim_system = jd.System.create(
    state_shape=state.shape,
    dt=1e-3,
    linear_integrator_type="verlet",
    rotation_integrator_type="",
    domain_type="periodic",
    force_model_type="spring",
    collider_type="naive",
    mat_table=mat_table,
    domain_kw={"box_size": fire_system.domain.box_size},
    bonded_force_model=container,
)

for _ in range(100):
    state, sim_system = sim_system.step(state, sim_system)
print("100 Verlet steps completed")
