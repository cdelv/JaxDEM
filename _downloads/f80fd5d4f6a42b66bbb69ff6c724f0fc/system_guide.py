# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""The Simulation System
----------------------------------------

Now that we know how to use and modify the simulation state
(:py:class:`~jaxdem.state.State`), we move to the simulation
configuration in :py:class:`~jaxdem.system.System`.

A :py:class:`~jaxdem.system.System` holds the domain, integrator settings,
force model, and evolving global values such as time and the PRNG key.
Both :py:class:`~jaxdem.state.State` and :py:class:`~jaxdem.system.System` are JAX pytrees: array values such as the time step
:math:`\Delta t` and box dimensions can change inside compiled functions.
Component classes, callable configuration, and other static metadata select
a compiled specialization; change those during host-side setup.
"""

# %%
# System Creation
# ~~~~~~~~~~~~~~~~~~~~~
# By default, :py:meth:`~jaxdem.system.System.create` fills unspecified
# attributes (e.g., :py:attr:`~jaxdem.system.System.domain`,
# :py:attr:`~jaxdem.system.System.force_model`, :math:`\Delta t`) with default values.

import jax
import jax.numpy as jnp

import jaxdem as jdem

# %%
# The system's dimension must match the state's dimension. Some
# components (e.g., domains) transform arrays of shape :math:`(N, d)`
# and require :math:`d` to agree with the system.

state = jdem.State.create(pos=jnp.zeros((1, 2)))
system = jdem.System.create(state.shape)
# %%
# Creation assembles the configuration. Initialization is a separate explicit
# step through :py:meth:`~jaxdem.system.System.initialize` that evaluates
# the starting forces and prepares search state before the
# first integration kick.
state, system = jdem.System.initialize(state, system)
state, system = system.step(state, system)  # one step

# %%
# Instead of :py:attr:`state.shape <jaxdem.state.State.shape>`, you can pass the state itself with
# ``state=state``. JaxDEM then infers the shape and forwards the state
# to colliders whose :py:meth:`Create <jaxdem.colliders.cell_list.DynamicCellList.Create>` method needs one
# (cell lists, neighbor
# lists).

system = jdem.System.create(state=state)

# %%
# A note on static methods
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Every operation on :py:class:`~jaxdem.state.State` and
# :py:class:`~jaxdem.system.System` (:py:meth:`~jaxdem.system.System.step`,
# :py:meth:`~jaxdem.system.System.trajectory_rollout`,
# :py:meth:`~jaxdem.state.State.merge`,
# :py:meth:`~jaxdem.state.State.stack`, etc.) is a **static method**. That means
# ``system.step(state, system)`` and ``jdem.System.step(state, system)`` are
# equivalent. Static methods make it easy to use these operations
# inside `jax.jit <https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>`_,
# `jax.vmap <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>`_, and other JAX transforms.

# %%
# Configuring the System
# ~~~~~~~~~~~~~~~~~~~~~~~~
# You can configure submodules when you create the system with keyword arguments.

system = jdem.System.create(state.shape, domain_type="periodic")
print("periodic domain:", system.domain)

# %%
# You can also pass constructor arguments to submodules with ``*_kw`` dictionaries.

system = jdem.System.create(
    state.shape,
    domain_type="periodic",
    domain_kw={"box_size": 10.0 * jnp.ones(2), "anchor": jnp.zeros(2)},
)
print("periodic domain (10x10):", system.domain)

# %%
# Passing Module Objects Directly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Internally, :py:meth:`~jaxdem.system.System.create` builds each submodule and
# runs sanity checks. You can also build a component yourself and pass the
# instance directly. Every component slot accepts a pre-built instance
# (:py:attr:`~jaxdem.system.System.domain`, :py:attr:`~jaxdem.system.System.collider`,
# :py:attr:`~jaxdem.system.System.linear_integrator`, :py:attr:`~jaxdem.system.System.rotation_integrator`,
# :py:attr:`~jaxdem.system.System.force_model`, :py:attr:`~jaxdem.system.System.force_manager`,
# :py:attr:`~jaxdem.system.System.bonded_force_model`, :py:attr:`~jaxdem.system.System.mat_table`).
# The instance overrides the corresponding ``*_type`` / ``*_kw`` arguments.

domain = jdem.Domain.create("free", dim=2)
collider = jdem.Collider.create("naive")
system = jdem.System.create(state.shape, domain=domain, collider=collider)
print("free default domain:", system.domain)
print("directly assigned collider:", type(system.collider).__name__)

# %%
# This works for instances of your own custom components too (see the
# :doc:`custom modules guide <../auto_examples/custom_modules_guide>`).
#
# Post-hoc replacement (``system.domain = domain``) also works, because
# :py:class:`~jaxdem.system.System` is a mutable dataclass. Passing the
# instances to :py:meth:`~jaxdem.system.System.create` is better, because
# the factory validates the components and wires them together. To swap
# the dynamics setup, use
# `dataclasses.replace <https://docs.python.org/3/library/dataclasses.html#dataclasses.replace>`_ to change
# the integrators
# and keep everything else, including the domain's *current* box.

import jax.numpy as jnp

import dataclasses

from jaxdem.integrators import LinearIntegrator

system_dyn = dataclasses.replace(
    system,
    linear_integrator=LinearIntegrator.create("verlet"),
    dt=jnp.asarray(1e-3, dtype=float),
)
print("swapped integrator:", type(system_dyn.linear_integrator).__name__)

# %%
# Summary: three ways to build a system component
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The previous sections showed three equivalent ways to get a component
# into a :py:class:`~jaxdem.system.System`. From most to least common:
#
# 1. **Let** :py:meth:`System.create <jaxdem.system.System.create>` **build it**. Pass the registered name and
#    its constructor arguments::
#
#        system = jdem.System.create(state.shape, domain_type="periodic",
#                                    domain_kw={"box_size": box})
#
# 2. **Build it yourself, then pass the instance**. Use the component's
#    own factory (:py:meth:`Domain.create <jaxdem.factory.Factory.create>`, as in
#    ``jdem.Domain.create("periodic", box_size=box)``) or its
#    constructor, and hand the object to :py:meth:`System.create <jaxdem.system.System.create>` with the slot
#    name (``domain=domain``). Equivalent to 1. Useful when you want to
#    inspect or reuse the component.
#
# 3. **Assign it to an existing system**. ``system.domain = domain``.
#    Use this to swap components after creation. Prefer 1 or 2 when
#    first building the system, so the factory can validate the combination.
#
# Two things live outside this scheme. Writers are plain objects you
# construct directly (:py:class:`~jaxdem.writers.vtk_writer.VTKWriter`). Minimizers are Optax
# constructor *functions* (e.g. :py:func:`~jaxdem.minimizers.optimizers.fire`) passed via ``minimizer=`` /
# ``minimizer_kw=``. See the :doc:`integrator guide <../auto_examples/integrator_guide>`.

# %%
# Time stepping
# ~~~~~~~~~~~~~
# The system controls how the simulation advances in time.
# Use :py:meth:`~jaxdem.system.System.step` for a single step or multiple steps at once. Multi-step calls
# use `jax.lax.fori_loop <https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html>`_ internally for
# speed.

state = jdem.State.create(jnp.zeros((1, 2)))
state, system = jdem.System.initialize(state, system)
state, system = system.step(state, system)  # 1 step

# Multiple steps in a single call:
state, system = system.step(state, system, n=10)  # 10 steps

# %%
# Trajectory rollout
# ~~~~~~~~~~~~~~~~~~~~~
# If you want to store snapshots along the way, use
# :py:meth:`~jaxdem.system.System.trajectory_rollout`. It records ``n``
# snapshots separated by ``stride`` integration steps each, for a total
# of :math:`n \times \text{stride}` steps. It takes each snapshot *after*
# its integration steps, so it does not store the initial (step-0) state.
# To record it, save it yourself before the rollout, or pass per-frame
# ``strides`` with a leading ``0`` entry.

state = jdem.State.create(jnp.zeros((1, 2)), vel=jnp.array([[0.1, 0.0]]))
state, system = jdem.System.initialize(state, system)
start_time = system.time

state, system, trajectory = system.trajectory_rollout(
    state, system, n=10, stride=2  # total steps = 20
)

# %%
# The trajectory is a tuple of :py:class:`~jaxdem.state.State` and
# :py:class:`~jaxdem.system.System` objects with an extra leading
# axis of length ``n``.

traj_state, traj_system = trajectory
print("trajectory pos shape:", traj_state.pos.shape)  # (n, N, d)
print("Recorded frame times:", traj_system.time)
assert bool(jnp.isclose(system.time - start_time, 20 * system.dt))

# %%
# Inspecting Forces Without Evolving the System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.system.System.evaluate_forces` updates instantaneous forces
# and search caches without advancing time, integrator state, contact/plastic
# history, or queued loads. Use it for diagnostics after editing a snapshot.
# A later physical step can still consume the same queued loads. Custom force
# callbacks used this way must also be pure evaluations.

evaluated_state, evaluated_system = jdem.System.evaluate_forces(state, system)
print("Evaluation preserved time:", evaluated_system.time == system.time)
print("Evaluated force shape:", evaluated_state.force.shape)

# %%
# Batched simulations with vmap
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can run many independent simulations in parallel with
# `jax.vmap <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>`_.
# Make sure the initialization returns per-simulation
# :py:class:`~jaxdem.state.State`/:py:class:`~jaxdem.system.System` pairs.


def initialize(i):
    st = jdem.State.create(pos=jnp.array([[1.0, 1.0]]), rad=jnp.array([0.1]))
    sys = jdem.System.create(
        st.shape,
        domain_type="reflect",
        domain_kw={"box_size": (2 + i) * jnp.ones(2), "anchor": jnp.zeros(2)},
    )
    return jdem.System.initialize(st, sys)


# Create a small batch of three simulations.
state_b, system_b = jax.vmap(initialize)(jnp.arange(3))
print(system_b.domain)  # batched variable domain

# %%
# Advance each simulation by 10 steps.
# Use the class method (or a small wrapper) to avoid variable shadowing.

state_b, system_b = jax.vmap(lambda st, sys: jdem.System.step(st, sys, n=10))(
    state_b, system_b
)
print("batched pos shape:", state_b.pos.shape)  # (batch, N, d)

# %%
# Another way to create batch systems is :py:meth:`~jaxdem.system.System.stack`:

state = jdem.State.create(jnp.zeros((1, 2)))
system = jdem.System.create(
    state.shape,
)

system = system.stack([system, system, system])
print("Stacked time-step shape:", system.dt.shape)


# %%
# Deactivating Components
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Some modules can be **deactivated** when you create the system. For the
# integrators, pass ``None`` (preferred) to select the base no-op
# integrator. The empty string ``""`` is equivalent.
#
# .. list-table::
#    :header-rows: 1
#
#    * - Component
#      - Deactivation value
#      - Effect
#    * - ``linear_integrator_type``
#      - ``None`` (or ``""``)
#      - No position/velocity updates.
#    * - ``rotation_integrator_type``
#      - ``None`` (or ``""``)
#      - No orientation/angular-velocity updates.
#    * - ``collider_type``
#      - ``None`` (or ``""``)
#      - No non-bonded pair interactions; external and bonded forces remain active.
#    * - ``bonded_force_model_type``
#      - ``None`` (default)
#      - No bonded forces.
#    * - ``force_manager_kw`` -> ``gravity``
#      - ``None`` (default)
#      - No gravitational acceleration.
#    * - ``force_manager_kw`` -> ``force_functions``
#      - ``()`` (default)
#      - No custom external force and torque functions.
#
# Keep a valid domain and force-model configuration. For unbounded motion use
# ``domain_type="free"``; to omit non-bonded forces, deactivate the collider.
# Deactivating integration does not deactivate force evaluation or boundaries.

# No integration — a "frozen" system:
system_frozen = jdem.System.create(
    state.shape,
    linear_integrator_type=None,
    rotation_integrator_type=None,
)
print("Integrator:", type(system_frozen.linear_integrator).__name__)


# %%
# Random Number Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.system.System.create` accepts a ``seed`` (integer) or
# a ``key`` (`jax.random.PRNGKey <https://docs.jax.dev/en/latest/_autosummary/jax.random.PRNGKey.html>`_) to
# initialize the system's
# JAX PRNG state. An explicit ``key`` takes precedence over ``seed``.
# JaxDEM stores the key in :py:attr:`system.key <jaxdem.system.System.key>` for stochastic integrators or
# custom force functions.

system_rng = jdem.System.create(state.shape, seed=42)
print("PRNG key:", system_rng.key)
