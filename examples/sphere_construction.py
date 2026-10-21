r"""Polydisperse sphere packing in one call
============================================

:func:`~jaxdem.utils.particle_creation.build_sphere_system` is the sphere
counterpart of :func:`~jaxdem.utils.particle_creation.build_ga_system`.
Give it per-particle radii and a target packing fraction, and it
returns a minimized ``State`` plus a ready-to-step ``System``. It first
draws a loose random configuration with
:func:`random_sphere_configuration` at ``initial_phi`` (default 0.3,
or the target ``phi`` if that is lower). It then compresses the
configuration quasistatically to ``phi`` with FIRE (time step
``fire_dt``).

The small, dilute configurations below keep this construction example quick.
Minimized here means that residual forces are small; it does not imply a
rigid or jammed contact network. Length, mass, and energy use consistent
reduced units, with the sphere radii setting the length scale.

The target fraction determines the box volume, rather than the amount of
overlap between particles. For radii :math:`r_i`,

.. math::

    \phi = \frac{\sum_i v_d(r_i)}{V_{\mathrm{box}}},
    \qquad v_2(r)=\pi r^2,\qquad v_3(r)=\frac{4\pi}{3}r^3.

Thus a cubic box has side :math:`L=(\sum_i v_d(r_i)/\phi)^{1/d}`.
The 2D example uses two radii, so each population contributes according to
its area, not simply its particle count. Changing from 2D to 3D changes
both this volume law and the number of position coordinates.

Construction alternates a small box rescale with force relaxation. For a
change from :math:`\phi` to :math:`\phi'`, all box lengths are multiplied
by :math:`s=(\phi/\phi')^{1/d}` while particle radii remain fixed. FIRE then
removes the forces introduced by moving particle centers closer together.
Its ``fire_dt`` controls the minimization trajectory; the final ``dt``
controls the subsequent physical integration. They serve different purposes.
"""

# %%
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

import numpy as np

from jaxdem.utils.particle_creation import build_sphere_system
from jaxdem.utils.packing_utils import compute_packing_fraction


def summarize(label, state, system, target_phi):
    phi = float(compute_packing_fraction(state, system))
    box = np.asarray(system.domain.box_size)
    np.testing.assert_allclose(phi, target_phi, atol=1e-9)
    state, system = system.evaluate_forces(state, system)
    max_force = float(np.max(np.linalg.norm(np.asarray(state.force), axis=-1)))
    assert np.isfinite(max_force) and max_force < 1e-6
    system.check_overflow()
    print(f"[{label}]  N={state.N}  phi={phi:.4f}  box={box}")
    print(f"  maximum residual force = {max_force:.2e}")


# %%
# 1) Monodisperse 3D sphere packing, periodic, neighbor_list
state, system = build_sphere_system(
    particle_radii=[0.1] * 8,
    phi=0.20,
    dim=3,
    dt=1e-3,
    collider_type="neighbor_list",
    initial_phi=0.15,
    compression_step=0.025,
    compression_force_tol=1e-8,
    max_n_min_steps_per_outer=2_000,
    seed=0,
)
summarize("3D monodisperse, periodic, neighbor_list", state, system, 0.20)


# %%
# 2) Bidisperse 2D disk packing, reflect walls, naive collider
state, system = build_sphere_system(
    particle_radii=[0.08] * 6 + [0.12] * 2,
    phi=0.30,
    dim=2,
    dt=1e-3,
    domain_type="reflect",
    collider_type="naive",
    initial_phi=0.25,
    compression_step=0.025,
    compression_force_tol=1e-8,
    max_n_min_steps_per_outer=2_000,
    seed=1,
)
summarize("2D bidisperse, reflect, naive", state, system, 0.30)


# %%
# 3) Short Verlet rollout on the 2D system
state, system = system.initialize(state, system)
state, system = system.step(state, system, n=10)
assert np.all(np.isfinite(state.pos))
system.check_overflow()
print("10 Verlet steps on the 2D system completed.")
