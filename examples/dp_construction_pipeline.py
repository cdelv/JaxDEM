# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Geometric-asperity deformable particles: step-by-step pipeline
==================================================================

This example builds a packing of deformable particles (DPs) whose nodes
are the surface asperities of a geometric-asperity body. Each DP is a
soft body. A :class:`~jaxdem.bonded_forces.deformable_particle.DeformableParticleModel`
(or a plastic variant) holds it together and applies the standard DP
energies: area/volume (``ec``), element-measure (``em``), edges (``el``),
bending (``eb``), and surface tension (``gamma``).

Pipeline:

1. :func:`~jaxdem.utils.particle_creation.create_ga_state` with ``particle_type='dp'`` builds a State
   where each node carries ``mass / nv`` and ``union_volume / nv``, and
   shares a ``bond_id`` with every other node in its body.
2. :func:`~jaxdem.utils.particle_creation.distribute_bodies` places each body's bounding sphere at the
   target initial packing fraction and applies a uniform-random
   per-body rotation. The rotation rotates the DP nodes around their
   centroid.
3. :func:`~jaxdem.utils.particle_creation.create_dp_container` builds the bonded-force container on the
   placed state. A per-body convex-hull test (:func:`~jaxdem.utils.particle_creation.ga_surface_mask`)
   detects surface and interior nodes. A body with a solid core gets an
   interior node connected with "fan"-style struts. A body without a center
   node has every node on the surface, with no extra edges.
4. :func:`~jaxdem.utils.packing_utils.quasistatic_compress_to_packing_fraction` compresses to the
   target nominal packing fraction while the DP bonds keep each body
   self-consistent.
5. Build a Verlet system (with the container as ``bonded_force_model``)
   and run a few time steps.

The nominal packing fraction uses the stored reference sphere-union volume
shares. It is not the instantaneous area/volume enclosed by a deformed DP
mesh, including when the body has no center node. A small dilute 2D
configuration keeps construction and relaxation quick; the same pipeline
accepts 3D meshes.
"""

# %%
# Imports
import jax

import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jaxdem as jdem
from jaxdem.utils.particle_creation import (
    create_ga_state,
    create_dp_container,
    distribute_bodies,
)
from jaxdem.utils.packing_utils import (
    compute_packing_fraction,
    CompressionReason,
    quasistatic_compress_to_packing_fraction,
)

# %%
# Parameters
# -----------------
# Three deformable particles in 2D, six surface asperities each. The default
# ``core_type='hollow'`` adds no center node; every member is still solid.
# Set ``core_type='solid'`` to add a central node connected to every surface
# node via ``interior_edges='fan'``.

N = 3
nv = 6
dim = 2
particle_radius = 0.5
asperity_radius = 0.1
initial_phi_bb = 0.15
target_phi = 0.05  # nominal sphere-union fraction, not enclosed mesh area
seed = 0


# %%
# 1) Build the DP state
# ---------------------
# DP nodes get equal shares of mass and union volume. ``bond_id`` groups
# them by body. ``create_ga_state`` applies no rigid-body frame. Each
# node's ``pos_c`` is its world-frame position.
# For body mass :math:`M_I`, reference union volume :math:`V_{I,0}`, and
# :math:`n_I` nodes, this construction assigns :math:`M_I/n_I` and
# :math:`V_{I,0}/n_I` to each node. Summing those shares recovers the body
# totals. Unlike rigid-clump members, DP nodes have separate translational
# degrees of freedom: their relative positions are controlled by bonds,
# rather than by a shared quaternion and fixed body-frame offsets.

state = create_ga_state(
    N=N,
    nv=nv,
    dim=dim,
    particle_radius=particle_radius,
    asperity_radius=asperity_radius,
    particle_type="dp",
    core_type="hollow",
    n_samples=2_048,  # 2D union properties are analytical; no samples are used
    seed=seed,
    mesh_type="fibonacci",
)
print(f"DP state: N={state.N} nodes, {N} bodies")

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
# Build the container from the *placed* state so that its
# ``initial_edge_lengths`` / ``initial_bendings`` / etc. reflect the
# actual rest configuration. A per-body convex-hull test detects
# ``is_surface``. Here every node is on the hull.
# ``interior_edges='fan'`` connects any interior (core) node to every
# surface node in its body. It is a no-op here because there are no
# interior nodes, but it matters for ``core_type='solid'``.
#
# The reference geometry is part of the constitutive model. Edge and
# element penalties compare current lengths/measures with values recorded
# here, while the content term compares enclosed area/volume with its
# reference value. A rigid placement or rotation leaves those quantities
# unchanged. Constructing a new container after deformation would instead
# declare that deformed shape to be the new reference, removing its elastic
# strain. Keep this container throughout compression and dynamics.

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
print(
    f"   elements: {container.elements.shape if container.elements is not None else None}"
)
print(f"   edges:    {container.edges.shape if container.edges is not None else None}")
print(
    f"   adjacency:{container.element_adjacency.shape if container.element_adjacency is not None else None}"
)


# %%
# 4) Compress with FIRE, DP bonds active
# --------------------------------------
# Compression calls ``scale_to_packing_fraction``, which translates every
# body's centroid uniformly (internal geometry preserved), and then FIRE
# relaxes. Because the System holds the DP container, bonded forces keep
# each body coherent. Contact forces from other bodies can still deform
# it locally.
#
# More explicitly, write a node as
# :math:`\mathbf{x}_{Ii}=\mathbf{c}_I+\mathbf{u}_{Ii}` relative to its
# body's centroid. A box rescale about anchor :math:`\mathbf{a}` applies
#
# .. math::
#
#    \mathbf{c}'_I=\mathbf{a}+s(\mathbf{c}_I-\mathbf{a}),\qquad
#    \mathbf{x}'_{Ii}=\mathbf{c}'_I+\mathbf{u}_{Ii},\qquad
#    s=\left(\frac{\phi}{\phi'}\right)^{1/d}.
#
# The affine event translates each body without straining its internal
# mesh. Its subsequent relaxation can change the enclosed content. The
# nominal :math:`\phi=\sum_I V_{I,0}/V_{\mathrm{box}}` still uses reference
# shares, so an instantaneous enclosed-content fraction would need to be
# measured separately from the current mesh geometry.

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
    bonded_force_model=container,
)

phi0 = float(compute_packing_fraction(state, fire_system))
print(f"before compression: nominal phi = {phi0:.4f}")

compression = quasistatic_compress_to_packing_fraction(
    state,
    fire_system,
    target_phi=target_phi,
    step=0.01,
    force_tol=1e-8,
    max_n_min_steps_per_outer=2_000,
    max_n_outer_steps=12,
)
if compression.reason != CompressionReason.TARGET_REACHED:
    raise RuntimeError(
        f"Compression stopped: {compression.reason.name}; "
        f"minimizer: {compression.minimizer_reason.name}"
    )
state, fire_system, final_phi, final_pe = compression
np.testing.assert_allclose(final_phi, target_phi, atol=1e-9)
print(
    f"after compression:  nominal phi = {float(final_phi):.4f}  PE/node = {float(final_pe):.3e}"
)

# %%
# 5) Short Verlet rollout (dynamics)
# ----------------------------------
# ``dataclasses.replace`` swaps the FIRE setup for plain Verlet
# and keeps everything else. This includes the post-compression box size,
# the material table, the collider, and the DP bonded-force
# container, so no manual rebuild is needed. Passing ``""`` disables
# the rotation integrator: DP nodes carry no rigid-body orientation.

import jax.numpy as jnp

import dataclasses

from jaxdem.integrators import LinearIntegrator, RotationIntegrator

sim_system = dataclasses.replace(
    fire_system,
    linear_integrator=LinearIntegrator.create("verlet"),
    rotation_integrator=RotationIntegrator.create(
        ""
    ),  # disable rotation: DPs have no rigid-body orientation
    dt=jnp.asarray(1e-3, dtype=float),
)

state, sim_system = jdem.System.initialize(state, sim_system)
state, sim_system = sim_system.step(state, sim_system, n=10)
assert np.all(np.isfinite(state.pos))
print("10 Verlet steps completed")
