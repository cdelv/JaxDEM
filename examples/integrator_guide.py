r"""Integrators and Minimizers
----------------------------------------

:py:class:`~jaxdem.integrators.LinearIntegrator` and
:py:class:`~jaxdem.integrators.RotationIntegrator` advance the simulation
state in time. A :py:class:`~jaxdem.system.System` holds one of each — one
for translational degrees of freedom (position, velocity) and one for
rotational degrees of freedom (orientation, angular velocity).

JaxDEM also supports **energy minimization**: instead of integrating the
equations of motion, you can drive the system towards a potential-energy
minimum. Minimizers are not integrator subclasses — they are `optax`
gradient-transformation constructor functions (such as
:py:func:`jaxdem.minimizers.fire`) passed to
:py:meth:`~jaxdem.system.System.create` via ``minimizer`` /
``minimizer_kw``, as described in the Minimizers section below.

Let's see how to choose, configure, and swap them.
"""

# %%
# Linear vs Rotation Integrators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every integration step calls **both** integrators in sequence:
#
# 1. :py:meth:`~jaxdem.integrators.LinearIntegrator.step_before_force`
# 2. :py:meth:`~jaxdem.integrators.RotationIntegrator.step_before_force`
# 3. *Force evaluation*
# 4. :py:meth:`~jaxdem.integrators.RotationIntegrator.step_after_force`
# 5. :py:meth:`~jaxdem.integrators.LinearIntegrator.step_after_force`
#
# Rotation finishes first so a linear thermostat can measure both terminal
# velocities. This split lets you mix and match: you could pair a Velocity Verlet linear
# integrator with a SPIRAL rotation integrator, or disable rotation
# entirely while keeping translation active.

import numpy as np
import jax.numpy as jnp
import jaxdem as jdem

state = jdem.State.create(pos=jnp.zeros((1, 3)))
system = jdem.System.create(
    state.shape,
    linear_integrator_type="verlet",
    rotation_integrator_type="verletspiral",
)
print("Linear integrator:", type(system.linear_integrator).__name__)
print("Rotation integrator:", type(system.rotation_integrator).__name__)


# %%
# Choosing an Integrator
# ~~~~~~~~~~~~~~~~~~~~~~~~
# You select integrators by their registered name when calling
# :py:meth:`~jaxdem.system.System.create`. Some integrators accept
# additional keyword arguments through ``linear_integrator_kw`` or
# ``rotation_integrator_kw``. Consult the API reference for the specific
# parameters of each integrator.
#
# Start with ``verlet`` for inertial, conservative translation and
# ``verletspiral`` for rigid-body rotation. Use ``langevin`` when stochastic
# drag and thermal forcing are part of the intended model, or
# ``verlet_rescaling`` for explicit kinetic-temperature control. ``euler`` is a
# simpler first-order method; it generally needs smaller steps for comparable
# accuracy. Smooth frictionless spheres can disable rotation.
#
# JaxDEM imposes no unit system: positions, masses, stiffnesses, and time must
# be consistent. For a linear contact, the oscillation scale is
# ``sqrt(m_effective / k_effective)``. Choose dt well below the shortest
# relevant contact time, then repeat at dt/2 over the same physical duration.
# Compare the quantity of interest and, for conservative dynamics, energy
# drift. An integrator name or a visually plausible trajectory is not an
# accuracy check; the factory's default dt is not a stability guarantee.
#
# To connect this choice to the equations of motion, for position-dependent
# forces and constant mass the translational velocity-Verlet update is
#
# .. math::
#
#    \mathbf x_{n+1}=\mathbf x_n+\Delta t\,\mathbf v_n
#        +\tfrac12\Delta t^2\mathbf a_n,\qquad
#    \mathbf a_{n+1}=\mathbf F(\mathbf x_{n+1})/m,
#
#    \mathbf v_{n+1}=\mathbf v_n
#        +\tfrac12\Delta t(\mathbf a_n+\mathbf a_{n+1}).
#
# Explicit initialization supplies :math:`\mathbf a_0`; starting with a
# stale acceleration changes the first update. For a linear two-particle
# contact the reduced mass and angular frequency are
#
# .. math::
#
#    m_{\mathrm{eff}}=\frac{m_i m_j}{m_i+m_j},\qquad
#    \omega=\sqrt{k_{\mathrm{eff}}/m_{\mathrm{eff}}}.
#
# The dimensionless resolution parameter is :math:`\omega\Delta t`.
# For an undamped isolated linear impact the contact duration is
# :math:`\pi/\omega`; a dense packing can have faster collective modes,
# so this pair scale is a useful starting estimate rather than a universal
# stability bound. Timestep refinement tests the accuracy of the observable
# being measured. Minimization, described below, instead seeks force and
# torque balance and does not represent this physical time evolution.


# %%
# Available Linear Integrators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The following linear integrators are registered:

# (the empty key ``""`` is a registered no-op; we filter it out)
print("Linear integrators:", sorted(k for k in jdem.LinearIntegrator._registry if k))

# %%
# Available Rotation Integrators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The following rotation integrators are registered:

print(
    "Rotation integrators:", sorted(k for k in jdem.RotationIntegrator._registry if k)
)

# %%
# See the API documentation of each class for constructor parameters and
# algorithmic details.

# %%
# Deactivating an Integrator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Passing ``None`` as the integrator type (the **empty string** ``""`` is
# equivalent) selects the base no-op integrator, which leaves the
# corresponding degrees of freedom untouched. This is useful when you want
# to freeze translation or rotation.

# Freeze rotation — only translate:
system = jdem.System.create(
    state.shape,
    linear_integrator_type="verlet",
    rotation_integrator_type=None,
)
print("Rotation integrator (deactivated):", type(system.rotation_integrator).__name__)

# %%
# Freeze translation — only rotate:
system = jdem.System.create(
    state.shape,
    linear_integrator_type=None,
    rotation_integrator_type="verletspiral",
)
print("Linear integrator (deactivated):", type(system.linear_integrator).__name__)


# %%
# Passing Constructor Arguments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Some integrators take additional parameters. Pass them via
# ``linear_integrator_kw`` or ``rotation_integrator_kw``:

system = jdem.System.create(
    state.shape,
    linear_integrator_type="langevin",
    rotation_integrator_type=None,
    linear_integrator_kw={"gamma": 0.5, "temperature": 0.1, "k_B": 1.0},
)
print("Langevin gamma:", system.linear_integrator.gamma)


# %%
# Minimizers
# ~~~~~~~~~~~~
# Minimizers in JaxDEM are standard `optax` optimizers that descend the potential
# energy landscape. You configure them by passing the constructor function (such
# as `jaxdem.fire`, `jaxdem.damped_newtonian`, or standard `optax` optimizers like
# `optax.adam`) to `minimizer`, along with any keyword arguments in `minimizer_kw`.
#
# Inside `System.create`, JaxDEM constructs the optimizer and wraps it in a
# custom wrapper (`CustomGradientTransformation`) that tracks the constructor
# function and arguments for serialization.
#
# Force-only optimizers (``requires_value=False``, including FIRE and damped
# Newtonian) also support contact history and facet force conventions.
# Minimization freezes stored pair history by default. Pass
# ``advance_history=True`` to ``System.minimize`` to update displacement-dependent
# history after accepted steps. Both modes disable velocity-dependent forces.
# Value-based optimizers use analytical physical gradients when available,
# otherwise they differentiate energy with stored history fixed. For frictional
# laws, that energy gradient can differ from the mechanical contact force.
# AD through collider energy requires Naive or NeighborList; direct CellList
# and MultiCellList traversals do not support reverse-mode differentiation.
# A custom ``target_fn`` uses ordinary AD through the State mapping.
#
# Every optimizer uses the same continuous body coordinates: positions and
# unwrapped angles in 2D, or raw quaternions normalized for the State in 3D.
# Rotations use a fixed reference throughout minimization, preserving the
# meaning of accumulated optimizer gradients and parameter history.
#
# For checkpoint serialization, JaxDEM saves the import path of the constructor
# function (e.g. ``"jaxdem.minimizers.fire"`` or ``"optax.adam"``) and the dictionary
# of keyword parameters. Upon restoration, it resolves and calls the function to
# recreate the optimizer.
#
# We therefore *recommend* defining custom constructor functions and custom
# target functions (``target_fn``) in an importable module (not in
# ``__main__``), so you can restore system snapshots later. This is a
# recommendation, not a requirement: minimization itself works exactly the
# same with functions defined in your main script or a notebook. Only
# saving and reloading the system through a checkpoint needs a resolvable
# import path.
#
# .. note::
#    **Minimizers and integrators are independent.**
#
#    The minimizer and the linear/rotation integrators live in separate
#    fields of the system and never touch each other's configuration or
#    internal state: :py:meth:`~jaxdem.system.System.minimize` does not
#    advance time or use the integrators, and
#    :py:meth:`~jaxdem.system.System.step` ignores the minimizer. You can
#    first minimize and then integrate, integrate and then minimize, or
#    alternate between them in any order without one affecting the other.
#
# .. note::
#    **Using Composite Optimizers (e.g., `optax.chain`)**
#
#    If you want to use a composite optimizer like `optax.chain`, you cannot pass
#    it directly as an instantiated object because the checkpoint writer does not
#    support arbitrary nested object serialization.
#
#    Instead, define a simple wrapper function that constructs the chain. As
#    above, the minimizer works no matter where you define this function. You
#    only need to place it in a separate importable module (not your main
#    script or ``__main__``) so the checkpoint writer can save its import
#    path and restore the minimizer on load.
#
#    .. code-block:: python
#
#       # In an importable module, e.g., my_optimizers.py
#       import optax
#
#       def my_chained_optimizer(learning_rate=1e-3, max_grad_norm=1.0):
#           return optax.chain(
#               optax.clip_by_global_norm(max_grad_norm),
#               optax.adam(learning_rate)
#           )
#
#       # Then pass it to the system creation:
#       system = jdem.System.create(
#           ...,
#           minimizer=my_optimizers.my_chained_optimizer,
#           minimizer_kw={"learning_rate": 1e-4, "max_grad_norm": 0.5}
#       )

system = jdem.System.create(
    state.shape,
    minimizer=jdem.fire,
    minimizer_kw={"dt": 1e-2},
)
print("Minimizer:", system.minimizer.type_name)


# %%
# The :py:func:`~jaxdem.system.System.minimize` Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM provides a convenience method
# :py:meth:`~jaxdem.system.System.minimize` that runs a
# ``while_loop`` until both free-body force and torque norms meet tolerance.
# Custom targets use their position and world-angular gradient norms;
# those residuals can differ from the State's physical forces and torques.
# Step limits, nonfinite values and search overflow are unsuccessful exits.
# With the physical objective, FIRE and damped Newtonian evaluate energy once
# after relaxation. Custom targets are evaluated during optimization.
# Relaxation iterations are not elapsed physical time. A minimum can still
# have positive overlap energy under compression; small energy alone does
# not establish mechanical equilibrium or rigidity.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
)
system = jdem.System.create(
    state.shape,
    minimizer=jdem.fire,
    minimizer_kw={"dt": 0.05},
    mat_table=jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=1.0, poisson=0.25, density=1.0)]
    ),
)

# Unit masses and stiffness keep this small relaxation inexpensive.
result = system.minimize(state, system, max_steps=500, force_tol=1e-6)
state, system, steps, pe = result
print(f"Converged: {result.converged}, steps: {steps}, PE = {pe:.6e}")
print(f"Force residual: {result.info.force_max}, torque residual: {result.info.torque_max}")
if not bool(result.converged):
    raise RuntimeError(f"Relaxation failed with reason {result.reason}.")
# The initially overlapping pair can separate freely, so its minimum energy is zero.
np.testing.assert_allclose(pe, 0.0, atol=1e-8)


# %%
# The Integration Loop
# ~~~~~~~~~~~~~~~~~~~~~~
# For reference, the full per-step sequence executed by
# :py:meth:`~jaxdem.system.System.step` is:
#
# 1. :py:meth:`~jaxdem.domains.Domain.apply` — enforce boundary conditions
# 2. :py:meth:`~jaxdem.integrators.LinearIntegrator.step_before_force`
# 3. :py:meth:`~jaxdem.integrators.RotationIntegrator.step_before_force`
# 4. *Collider + force manager* — compute forces and torques
# 5. :py:meth:`~jaxdem.integrators.RotationIntegrator.step_after_force`
# 6. :py:meth:`~jaxdem.integrators.LinearIntegrator.step_after_force`
#
# The :py:meth:`~jaxdem.integrators.Integrator.step_before_force` / :py:meth:`~jaxdem.integrators.Integrator.step_after_force` split lets multi-stage
# schemes (such as Velocity Verlet) position their updates around the force
# evaluation correctly.
#
# Fixed Particles
# ~~~~~~~~~~~~~~~~~
# Particles with ``state.fixed = True`` keep their prescribed linear and
# angular velocities; forces do not accelerate them. Zero velocity keeps
# them stationary. See :doc:`../auto_examples/state_guide` for this field.
