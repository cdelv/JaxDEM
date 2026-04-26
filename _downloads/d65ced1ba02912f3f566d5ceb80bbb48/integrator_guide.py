r"""Integrators and Minimizers.
----------------------------------------

:py:class:`~jaxdem.integrators.LinearIntegrator` and
:py:class:`~jaxdem.integrators.RotationIntegrator` advance the simulation
state in time. A :py:class:`~jaxdem.system.System` holds one of each — one
for translational degrees of freedom (position, velocity) and one for
rotational degrees of freedom (orientation, angular velocity).

:py:class:`~jaxdem.minimizers.LinearMinimizer` and
:py:class:`~jaxdem.minimizers.RotationMinimizer` are special integrators
that drive the system towards a potential-energy minimum instead of
performing physical time integration. Because they subclass
:py:class:`~jaxdem.integrators.Integrator`, they can be plugged into the
same slots on :py:class:`~jaxdem.system.System`.

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
# 4. :py:meth:`~jaxdem.integrators.LinearIntegrator.step_after_force`
# 5. :py:meth:`~jaxdem.integrators.RotationIntegrator.step_after_force`
#
# This split lets you mix and match: you could pair a Velocity Verlet linear
# integrator with a SPIRAL rotation integrator, or disable rotation
# entirely while keeping translation active.

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
# Integrators are selected by their registered name when calling
# :py:meth:`~jaxdem.system.System.create`. Some integrators accept
# additional keyword arguments through ``linear_integrator_kw`` or
# ``rotation_integrator_kw``. Consult the API reference for the specific
# parameters of each integrator.


# %%
# Available Linear Integrators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The following linear integrators are registered:

print("Linear integrators:", list(jdem.LinearIntegrator._registry.keys()))

# Available Rotation Integrators
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The following rotation integrators are registered:

print("Rotation integrators:", list(jdem.RotationIntegrator._registry.keys()))

# %%
# See the API documentation of each class for constructor parameters and
# algorithmic details.

# %%
# Deactivating an Integrator
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Passing an **empty string** ``""`` as the integrator type selects the
# base no-op integrator, which leaves the corresponding degrees of freedom
# untouched. This is useful when you want to freeze translation or rotation.

# Freeze rotation — only translate:
system = jdem.System.create(
    state.shape,
    linear_integrator_type="verlet",
    rotation_integrator_type="",
)
print("Rotation integrator (deactivated):", type(system.rotation_integrator).__name__)

# %%
# Freeze translation — only rotate:
system = jdem.System.create(
    state.shape,
    linear_integrator_type="",
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
    linear_integrator_type="lineargradientdescent",
    rotation_integrator_type="",
    linear_integrator_kw={"learning_rate": 1e-4},
)
print("GD learning rate:", system.linear_integrator.learning_rate)


# %%
# Minimizers
# ~~~~~~~~~~~~
# Minimizers are integrators that descend the potential energy landscape
# instead of advancing physical time. They are registered in **both** the
# minimizer and integrator registries, so you can select them in
# :py:meth:`~jaxdem.system.System.create` the same way you select any
# integrator.

system = jdem.System.create(
    state.shape,
    linear_integrator_type="linearfire",
    rotation_integrator_type="rotationfire",
)
print("Linear minimizer:", type(system.linear_integrator).__name__)
print("Rotation minimizer:", type(system.rotation_integrator).__name__)


# %%
# The :py:func:`~jaxdem.minimizers.routines.minimize` Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM provides a convenience function
# :py:func:`~jaxdem.minimizers.routines.minimize` that runs a
# ``while_loop`` until the potential energy converges or a maximum step
# count is reached.

from jaxdem.minimizers import minimize

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
)
system = jdem.System.create(
    state.shape,
    linear_integrator_type="linearfire",
    rotation_integrator_type="",
)

state, system, steps, pe = minimize(
    state, system, max_steps=500, pe_tol=1e-12, pe_diff_tol=1e-12
)
print(f"Converged in {steps} steps, PE = {pe:.6e}")


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
# 5. :py:meth:`~jaxdem.integrators.LinearIntegrator.step_after_force`
# 6. :py:meth:`~jaxdem.integrators.RotationIntegrator.step_after_force`
#
# The :py:meth:`~jaxdem.integrators.Integrator.step_before_force` / :py:meth:`~jaxdem.integrators.Integrator.step_after_force` split lets multi-stage
# schemes (such as Velocity Verlet) position their updates around the force
# evaluation correctly.
#
# Fixed Particles
# ~~~~~~~~~~~~~~~~~
# Particles with ``state.fixed = True`` are **immobile**: the integrator
# multiplies velocity updates by ``(1 - fixed)`` so their velocity stays
# constant. See :doc:`../auto_examples/state_guide` for how to set this field.
