# %%
"""Catch-all builder: polydisperse GA/DP packings in one call
==============================================================

:func:`~jaxdem.utils.particleCreation.build_ga_system` bundles the full
pipeline (``create_ga_state`` → ``distribute_bodies`` → container build
for DPs → FIRE compression → final System.create with the user's
chosen integrator/collider/material) into a single call. It handles
polydispersity across the bodies — per-body ``particle_radius``,
``vertex_counts``, ``asperity_radius``, ``aspect_ratio``, and Thomson
minimization steps — by grouping bodies with identical specs so
``_compute_uniform_union_properties`` can be vmapped over each group.

This example runs three different configurations in one script:

1. **3D bidisperse rigid clumps** (periodic, neighborlist).
2. **2D clumps with a nontrivial aspect ratio** (periodic).
3. **3D deformable particles** (periodic, neighborlist) — returns the
   bonded-force container alongside ``(state, system)``.
"""

# %%
# Imports
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import jaxdem as jd
from jaxdem.utils.particleCreation import build_ga_system
from jaxdem.utils.packingUtils import compute_packing_fraction


def summarize(label, state, system, container=None):
    phi = float(compute_packing_fraction(state, system))
    box = np.asarray(system.domain.box_size)
    extra = "" if container is None else f"  [{type(container).__name__}]"
    print(f"[{label}]  N={state.N}  phi={phi:.4f}  box={box}{extra}")


# %%
# 1) 3D bidisperse rigid clumps
# ------------------------------
# Six large + six small clumps. The builder groups by unique
# ``(nv, radius, asperity_radius, aspect, n_steps, mass)`` tuples, so
# this triggers two calls to ``create_ga_state`` — one per group —
# rather than twelve.

state, system = build_ga_system(
    particle_radii=[0.5] * 6 + [0.3] * 6,
    vertex_counts=[12] * 6 + [8] * 6,
    asperity_radius=0.1,
    phi=0.30,
    dim=3,
    particle_type="clump",
    core_type="solid",
    # Placement
    domain_type="periodic",
    # Internal FIRE settings
    n_property_samples=30_000,
    compression_step=1e-2,
    max_n_min_steps_per_outer=50_000,
    # Final simulation system
    dt=1e-3,
    linear_integrator_type="verlet",
    rotation_integrator_type="verletspiral",
    collider_type="neighborlist",
    seed=0,
)
summarize("3D bidisperse rigid clumps", state, system)


# %%
# 2) 2D clumps with aspect ratio
# ------------------------------
# Ellipsoidal clumps in 2D: the aspect ratio is passed per-body (one
# length-2 tuple per body) so each clump can have its own ellipse
# shape. Here all four bodies share the same (1.0, 1.4) aspect for
# clarity; passing a shape-``(M, dim)`` array lets you vary it.

state, system = build_ga_system(
    particle_radii=[0.4] * 4,
    vertex_counts=[10] * 4,
    asperity_radius=0.08,
    aspect_ratio=[[1.0, 1.4]] * 4,
    phi=0.35,
    dim=2,
    particle_type="clump",
    core_type="solid",
    domain_type="periodic",
    n_property_samples=20_000,
    compression_step=1e-2,
    max_n_min_steps_per_outer=30_000,
    dt=1e-3,
    linear_integrator_type="verlet",
    rotation_integrator_type="verletspiral",
    collider_type="naive",
    seed=1,
)
summarize("2D aspect-ratio clumps", state, system)


# %%
# 3) 3D deformable particles
# --------------------------
# The DP path returns a third value (the container), which is already
# wired into ``system`` via ``bonded_force_model``. Energies
# ``em, ec, eb, el`` are supplied; ``gamma`` (surface tension) is
# left off for brevity. Pass ``plasticity_type`` + ``dp_tau_s`` to get
# an edge / perimeter / bending plastic variant instead.

state, system, container = build_ga_system(
    particle_radii=[0.5] * 4,
    vertex_counts=[12] * 4,
    asperity_radius=0.1,
    phi=0.12,
    dim=3,
    particle_type="dp",
    core_type="hollow",
    domain_type="periodic",
    n_property_samples=20_000,
    compression_step=1e-2,
    max_n_min_steps_per_outer=50_000,
    dt=1e-3,
    linear_integrator_type="verlet",
    rotation_integrator_type="",  # DPs have no rigid-body orientation
    collider_type="neighborlist",
    dp_em=1.0,
    dp_ec=1.0,
    dp_eb=0.1,  # bending is stiff for small triangles; keep this modest
    dp_el=0.3,
    dp_gamma=None,
    seed=2,
)
summarize("3D DPs", state, system, container)


# %%
# Short rollout on the DP system to confirm everything composes
# -------------------------------------------------------------
for _ in range(100):
    state, system = system.step(state, system)
print("100 Verlet steps on the DP system completed cleanly.")
