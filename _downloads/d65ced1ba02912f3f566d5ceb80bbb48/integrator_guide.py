# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Integrators and Minimizers
------------------------------

A :py:class:`~jaxdem.system.System` holds one
:py:class:`~jaxdem.integrators.LinearIntegrator` (positions, velocities) and
one :py:class:`~jaxdem.integrators.RotationIntegrator` (orientations, angular
velocities), plus an optional `Optax <https://optax.readthedocs.io/en/latest/>`_
minimizer for driving the system to a potential-energy minimum.
"""

# %%
# The Per-Step Integration Pipeline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each call to :py:meth:`~jaxdem.system.System.step` executes:
#
# 1. ``user_pre_step_actions`` — optional user callbacks (e.g. advancing shear strain)
# 2. :py:meth:`LinearIntegrator.step_before_force <jaxdem.integrators.Integrator.step_before_force>`
#    and :py:meth:`RotationIntegrator.step_before_force <jaxdem.integrators.Integrator.step_before_force>`
# 3. :py:meth:`~jaxdem.domains.Domain.apply` — boundary enforcement (e.g. wall reflections)
# 4. *Collider + bonded forces + force manager* — compute forces and torques
# 5. :py:meth:`RotationIntegrator.step_after_force <jaxdem.integrators.Integrator.step_after_force>`
#    then :py:meth:`LinearIntegrator.step_after_force <jaxdem.integrators.Integrator.step_after_force>`
# 6. ``bonded_force_model.update_reference_state`` and ``user_post_step_actions``
#
# Rotation finishes before linear ``step_after_force`` so thermostats can
# measure both translational and rotational velocities.

from dataclasses import replace

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem

# %%
# Built-in Integrators
# ~~~~~~~~~~~~~~~~~~~~
# Select integrators via ``linear_integrator_type`` / ``rotation_integrator_type``
# and pass parameters through ``*_kw`` (pass ``None`` or ``""`` to freeze
# translation or rotation):
#
# .. list-table::
#    :header-rows: 1
#
#    * - Name
#      - Class
#      - Role & key parameters
#    * - ``"verlet"``
#      - :py:class:`~jaxdem.integrators.velocity_verlet.VelocityVerlet`
#      - Default 2nd-order conservative translational integrator.
#    * - ``"euler"``
#      - :py:class:`~jaxdem.integrators.direct_euler.DirectEuler`
#      - 1st-order symplectic (semi-implicit) Euler.
#    * - ``"langevin"``
#      - :py:class:`~jaxdem.integrators.langevin.Langevin`
#      - Stochastic drag + thermal noise (``gamma``, ``temperature``, ``k_B``).
#    * - ``"verlet_rescaling"``
#      - :py:class:`~jaxdem.integrators.velocity_verlet_rescaling.VelocityVerletRescaling`
#      - Velocity Verlet with kinetic-temperature rescaling (``temperature``, ``k_B``, ``rescale_every``, ``can_rotate``).
#    * - ``"verletspiral"``
#      - :py:class:`~jaxdem.integrators.velocity_verlet_spiral.VelocityVerletSpiral`
#      - Default 2nd-order rigid-body rotation integrator.
#    * - ``"spiral"``
#      - :py:class:`~jaxdem.integrators.spiral.Spiral`
#      - 1st-order quaternion rotation integrator.
#
# Smooth frictionless spheres can set ``rotation_integrator_type=None``.

state = jdem.State.create(pos=jnp.zeros((1, 3)))
system = jdem.System.create(
    state.shape,
    linear_integrator_type="langevin",
    rotation_integrator_type=None,
    linear_integrator_kw={"gamma": 0.5, "temperature": 0.1, "k_B": 1.0},
)
print(
    "Linear:",
    type(system.linear_integrator).__name__,
    f"(gamma={system.linear_integrator.gamma})",
    "| Rotation:",
    type(system.rotation_integrator).__name__,
)

# %%
# Choosing a Timestep: Measuring Integration Error
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For a linear two-particle contact, the reduced mass and contact frequency are
# :math:`m_{\mathrm{eff}} = m_i m_j / (m_i + m_j)` and
# :math:`\omega = \sqrt{k_{\mathrm{eff}} / m_{\mathrm{eff}}}`.
# Choose :math:`\Delta t` well below the contact time :math:`\pi / \omega` and
# verify by repeating at :math:`\Delta t / 2`.
#
# Below we integrate a compressed stationary pair up to :math:`\omega T = 1.3`
# (before separation at :math:`\pi / 2`), where the exact overlap is
# :math:`\delta(t) = \delta_0 \cos(\omega t)`:

contact_state = jdem.State.create(
    pos=jnp.array([[-0.45, 0.0], [0.45, 0.0]]),
    rad=jnp.full(2, 0.5),
    mass=jnp.ones(2),
)
contact_materials = jdem.MaterialTable.from_materials(
    [jdem.Material.create("elastic", young=1.0, poisson=0.25, density=1.0)]
)
stiffness = float(contact_materials.young_eff[0, 0])
omega = np.sqrt(stiffness / 0.5)
initial_overlap = float(contact_state.rad.sum() - jnp.diff(contact_state.pos[:, 0])[0])
initial_energy = 0.5 * stiffness * initial_overlap**2
duration = 1.3 / omega
step_counts = np.array([8, 16, 32, 64])


def contact_observables(st: jdem.State, sys: jdem.System):
    sep = jnp.linalg.norm(st.pos[1] - st.pos[0])
    overlap = jnp.maximum(0.0, st.rad.sum() - sep)
    ke = 0.5 * jnp.sum(st.mass[:, None] * st.vel**2)
    return sys.time, overlap, ke + 0.5 * stiffness * overlap**2


fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for method, label, color in (
    ("euler", "Symplectic Euler", "tab:orange"),
    ("verlet", "Velocity Verlet", "tab:blue"),
):
    contact_system = jdem.System.create(
        state=contact_state,
        dt=duration / step_counts[0],
        collider_type="naive",
        force_model_type="spring",
        linear_integrator_type=method,
        rotation_integrator_type=None,
        mat_table=contact_materials,
    )
    overlap_errors = []
    for count in step_counts:
        trial_sys = replace(contact_system, dt=jnp.asarray(duration / count))
        trial_st, trial_sys = jdem.System.initialize(contact_state, trial_sys)
        final_st, final_sys = jdem.System.step_dynamic(
            trial_st, trial_sys, n=jnp.asarray(count)
        )
        _, final_overlap, _ = contact_observables(final_st, final_sys)
        overlap_errors.append(
            abs(float(final_overlap) / initial_overlap - np.cos(omega * duration))
        )

    trial_st, trial_sys = jdem.System.initialize(contact_state, contact_system)
    _, _, (times, _, energies) = jdem.System.trajectory_rollout(
        trial_st, trial_sys, n=int(step_counts[0]), save_fn=contact_observables
    )
    axes[0].plot(
        np.r_[0.0, omega * np.asarray(times)],
        np.r_[0.0, np.asarray(energies) / initial_energy - 1.0],
        "o-",
        color=color,
        label=label,
        markersize=4,
    )
    axes[1].loglog(
        omega * duration / step_counts, overlap_errors, "o-", color=color, label=label
    )
    print(f"{label}: overlap errors = {np.array(overlap_errors)}")

resolution = omega * duration / step_counts
axes[1].loglog(resolution, 0.08 * resolution, "k--", alpha=0.5, label=r"$O(\Delta t)$")
axes[1].loglog(
    resolution, 0.05 * resolution**2, "k:", alpha=0.6, label=r"$O(\Delta t^2)$"
)
axes[0].axhline(0.0, color="0.5", linewidth=0.8)
axes[0].set(
    xlabel=r"Dimensionless time $\omega t$",
    ylabel=r"Relative energy error $E/E_0 - 1$",
    title="Energy error at coarsest timestep",
)
axes[1].set(
    xlabel=r"Timestep resolution $\omega\Delta t$",
    ylabel=r"Overlap error $|\delta(T)-\delta_{\mathrm{exact}}(T)|/\delta_0$",
    title="Convergence under timestep refinement",
)
for ax in axes:
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize="small")
plt.show()

# %%
# Energy Minimizers
# ~~~~~~~~~~~~~~~~~
# Minimizers are Optax optimizer constructor functions (such as
# :py:func:`~jaxdem.minimizers.optimizers.fire`,
# :py:func:`~jaxdem.minimizers.optimizers.damped_newtonian`, or `optax.adam
# <https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html>`_)
# passed via ``minimizer=`` / ``minimizer_kw=``.
#
# .. note::
#    **Minimizers and integrators are independent.**
#    :py:meth:`~jaxdem.system.System.minimize` does not advance ``system.time``
#    or touch integrator state, and :py:meth:`~jaxdem.system.System.step`
#    ignores the minimizer. You can alternate minimization and dynamics freely.
#
# :py:meth:`~jaxdem.system.System.minimize` runs ``jax.lax.while_loop`` until
# both force and torque residuals meet tolerance, returning a
# :py:class:`~jaxdem.minimizers.routines.MinimizationResult` with fields
# ``state``, ``system``, ``steps``, ``energy``, ``converged``, ``reason``, and
# ``info`` (``force_max``, ``torque_max``):

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
)
system = jdem.System.create(
    state.shape,
    minimizer=jdem.fire,
    minimizer_kw={"dt": 0.05},
    mat_table=contact_materials,
)

result = jdem.System.minimize(state, system, max_steps=500, force_tol=1e-6)
state, system, steps, pe = result
print(
    f"Converged={bool(result.converged)}, steps={int(steps)}, PE={float(pe):.3e}, "
    f"force_max={float(result.info.force_max):.3e}, reason={result.reason}"
)
if not bool(result.converged):
    raise RuntimeError(f"Relaxation failed with reason {result.reason}.")

# %%
# Minimizer Details: History, AD, Custom Targets, and Checkpointing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * **Force-only vs value-based optimizers:**
#   :py:func:`~jaxdem.minimizers.optimizers.fire` and
#   :py:func:`~jaxdem.minimizers.optimizers.damped_newtonian` use physical
#   forces/torques directly (compatible with facets and contact history).
#   Value-based optimizers (e.g. L-BFGS / conjugate gradient) use analytical
#   gradients when available or reverse-mode AD through collider energy—which
#   requires :py:class:`~jaxdem.colliders.naive.NaiveSimulator`,
#   :py:class:`~jaxdem.colliders.neighbor_list.NeighborList`, or static cell
#   lists (``static=True``).
# * **Contact history during minimization:** Stored pair history is frozen by
#   default. Pass ``history_update="path"`` to advance it along accepted steps
#   or ``history_update="endpoint"`` to advance once from start to finish
#   (all modes run with ``quasistatic=True``).
# * **Custom objectives:** Pass a differentiable ``target_fn(state, system) -> scalar``
#   to :py:meth:`~jaxdem.system.System.create` (e.g. to add an external penalty
#   or target pressure) and minimize with any Optax optimizer.
# * **Composite optimizers & checkpoints:** Checkpoints store the import path
#   of ``minimizer`` and ``minimizer_kw``. To checkpoint an ``optax.chain(...)``
#   optimizer, wrap it in a top-level function inside an importable module:
#
#   .. code-block:: python
#
#      # my_optimizers.py
#      import optax
#
#      def clipped_adam(learning_rate=1e-3, max_grad_norm=1.0):
#          return optax.chain(
#              optax.clip_by_global_norm(max_grad_norm),
#              optax.adam(learning_rate),
#          )
#
#      system = jdem.System.create(
#          state.shape,
#          minimizer=my_optimizers.clipped_adam,
#          minimizer_kw={"learning_rate": 1e-4, "max_grad_norm": 0.5},
#      )
#
# Particles with :py:attr:`state.fixed <jaxdem.state.State.fixed>` set to
# ``True`` keep their prescribed velocity during dynamics and stay fixed during
# minimization; see :doc:`../auto_examples/state_guide`.
