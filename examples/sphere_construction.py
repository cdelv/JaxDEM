# %%
"""Polydisperse sphere packing in one call
============================================

:func:`~jaxdem.utils.particleCreation.build_sphere_system` is the sphere
counterpart of :func:`~jaxdem.utils.particleCreation.build_ga_system`:
give it per-particle radii and a target packing fraction, and it
returns a minimized ``State`` plus a ready-to-step ``System``. Under
the hood it draws a loose random configuration via
:func:`random_sphere_configuration`, then quasistatically compresses
to ``phi``.
"""

# %%
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import numpy as np

from jaxdem.utils.particleCreation import build_sphere_system
from jaxdem.utils.packingUtils import compute_packing_fraction


def summarize(label, state, system):
    phi = float(compute_packing_fraction(state, system))
    box = np.asarray(system.domain.box_size)
    print(f"[{label}]  N={state.N}  phi={phi:.4f}  box={box}")


# %%
# 1) Monodisperse 3D sphere packing, periodic, neighborlist
state, system = build_sphere_system(
    particle_radii=[0.1] * 64,
    phi=0.55,
    dim=3,
    dt=1e-3,
    collider_type="neighborlist",
    seed=0,
)
summarize("3D monodisperse, periodic, neighborlist", state, system)


# %%
# 2) Bidisperse 2D disk packing, reflect walls, naive collider
state, system = build_sphere_system(
    particle_radii=[0.08] * 40 + [0.12] * 10,
    phi=0.70,
    dim=2,
    dt=1e-3,
    domain_type="reflect",
    collider_type="naive",
    seed=1,
)
summarize("2D bidisperse, reflect, naive", state, system)


# %%
# 3) Short Verlet rollout on the 2D system
for _ in range(100):
    state, system = system.step(state, system)
print("100 Verlet steps on the 2D system completed.")
