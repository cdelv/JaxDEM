r"""
The Simulation System
----------------------------------------

Now that we know how to use and manipulate the simulation state
(:py:class:`~jaxdem.state.State`), it's time to delve into the simulation
configuration in :py:class:`~jaxdem.system.System`.

A :py:class:`~jaxdem.system.System` holds the "static" configuration of a
simulation, such as the domain, integrator settings, and force model. Although
we call it "static", many fields (e.g., the time step :math:`\Delta t`,
domain dimensions, boundary conditions) can be changed at runtime—even inside a
JIT-compiled function—because both :py:class:`~jaxdem.state.State` and
:py:class:`~jaxdem.system.System` are JAX pytrees.
"""

# %%
# System Creation
# ~~~~~~~~~~~~~~~~~~~~~
# By default, :py:meth:`~jaxdem.system.System.create` initializes unspecified
# attributes (e.g., domain, force_model, :math:`\Delta t`) with sensible defaults.

import dataclasses as _dc
import jax
import jax.numpy as jnp
import jaxdem as jdem

# %%
# The system's dimension must match the state's dimension. Some
# components (e.g., domains) transform arrays of shape :math:`(N, d)`
# and require :math:`d` to agree with the system.

state = jdem.State.create(pos=jnp.zeros((1, 2)))
system = jdem.System.create(state.shape)
state, system = system.step(state, system)  # one step

# %%
# A note on static methods
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Every operation on :py:class:`~jaxdem.state.State` and
# :py:class:`~jaxdem.system.System` (``step``, ``trajectory_rollout``,
# ``merge``, ``stack``, etc.) is a **static method**. That means
# ``system.step(state, system)`` and ``jdem.System.step(state, system)`` are
# equivalent. Static methods make it straightforward to use these operations
# inside :py:func:`jax.jit`, :py:func:`jax.vmap`, and other JAX transforms.

# %%
# Configuring the System
# ~~~~~~~~~~~~~~~~~~~~~~~~
# You can configure submodules when creating the system via keyword arguments.

system = jdem.System.create(state.shape, domain_type="periodic")
print("periodic domain:", system.domain)

# %%
# You can also pass constructor arguments to submodules via *_kw dictionaries.

system = jdem.System.create(
    state.shape,
    domain_type="periodic",
    domain_kw=dict(box_size=10.0 * jnp.ones(2), anchor=jnp.zeros(2)),
)
print("periodic domain (10x10):", system.domain)

# %%
# Manually swapping a submodule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Internally, :py:meth:`~jaxdem.system.System.create` builds each submodule and
# performs sanity checks. You can also create a submodule manually and replace
# it using:

domain = jdem.Domain.create("free", dim=2)
system.domain = domain
print("free default domain:", system.domain)

# %%
# Time stepping
# ~~~~~~~~~~~~~
# The system controls how the simulation advances in time.
# You can take a single step or multiple steps at once. Multi-step calls
# use :py:func:`jax.lax.fori_loop` internally for speed.

state = jdem.State.create(jnp.zeros((1, 2)))
state, system = system.step(state, system)  # 1 step

# Multiple steps in a single call:
state, system = system.step(state, system, n=10)  # 10 steps

# %%
# Trajectory rollout
# ~~~~~~~~~~~~~~~~~~~~~
# If you want to store snapshots along the way, use
# :py:meth:`~jaxdem.system.System.trajectory_rollout`. It records ``n``
# snapshots separated by ``stride`` integration steps each, for a total
# of :math:`n \times \text{stride}` steps.

state = jdem.State.create(jnp.zeros((1, 2)))

state, system, trajectory = system.trajectory_rollout(
    state, system, n=10, stride=2  # total steps = 20
)

# %%
# The trajectory is a ``Tuple[State, System]`` with an extra leading
# axis of length ``n``.

traj_state, traj_system = trajectory
print("trajectory pos shape:", traj_state.pos.shape)  # (n, N, d)

# %%
# Batched simulations with vmap
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can run many independent simulations in parallel with :py:func:`jax.vmap`.
# Make sure the initialization returns per-simulation State/System pairs.


def initialize(i):
    st = jdem.State.create(jnp.zeros((1, 2)))
    sys = jdem.System.create(
        st.shape,
        domain_type="reflect",
        domain_kw=dict(box_size=(2 + i) * jnp.ones(2), anchor=jnp.zeros(2)),
    )
    return st, sys


# Create a batch of 5 simulations
state_b, system_b = jax.vmap(initialize)(jnp.arange(5))
print(system_b.domain)  # batched variable domain

# %%
# Advance each simulation by 10 steps.
# Use the class method (or a small wrapper) to avoid variable shadowing.

state_b, system_b = jax.vmap(lambda st, sys: jdem.System.step(st, sys, n=10))(
    state_b, system_b
)
print("batched pos shape:", state_b.pos.shape)  # (batch, N, d)

# %%
# Another way to create batch systems is the stack method:

state = jdem.State.create(jnp.zeros((1, 2)))
system = jdem.System.create(
    state.shape,
)

system = system.stack([system, system, system])
print("stacked system:", system)


# %%
# Deactivating Components
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Some modules can be **deactivated** by passing an empty string ``""``
# (or ``None``, depending on the field) when creating the system. The
# base class is then used, which provides no-op behaviour.
#
# .. list-table::
#    :header-rows: 1
#
#    * - Component
#      - Deactivation value
#      - Effect
#    * - ``collider_type``
#      - ``""``
#      - No pairwise force computation; forces/torques are zeroed.
#    * - ``linear_integrator_type``
#      - ``""``
#      - No position/velocity updates.
#    * - ``rotation_integrator_type``
#      - ``""``
#      - No orientation/angular-velocity updates.
#    * - ``bonded_force_model_type``
#      - ``None`` (default)
#      - No bonded forces.
#    * - ``force_manager_kw`` → ``gravity``
#      - ``None`` (default)
#      - No gravitational acceleration.
#
# **Note:** the domain (``domain_type``) and force model
# (``force_model_type``) cannot be deactivated — a valid type must
# always be provided.

# No collisions, no integration — a "frozen" system:
system_frozen = jdem.System.create(
    state.shape,
    collider_type="",
    linear_integrator_type="",
    rotation_integrator_type="",
)
print("Collider:", type(system_frozen.collider).__name__)
print("Integrator:", type(system_frozen.linear_integrator).__name__)


# %%
# Random Number Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.system.System.create` accepts a ``seed`` (integer) or
# a ``key`` (:py:func:`jax.random.PRNGKey`) that initialises the system's
# JAX PRNG state. The key is stored in ``system.key`` and is available for
# stochastic integrators or custom force functions.

system_rng = jdem.System.create(state.shape, seed=42)
print("PRNG key:", system_rng.key)
