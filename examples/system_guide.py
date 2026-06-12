r"""The Simulation System
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
# Instead of ``state.shape``, you can pass the state itself with
# ``state=state``: the shape is inferred, and the state is forwarded
# automatically to colliders whose ``Create`` method needs one (cell
# lists, neighbor lists).

system = jdem.System.create(state=state)

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
# You can also pass constructor arguments to submodules via ``*_kw`` dictionaries.

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
# performs sanity checks. You can also build a component yourself and pass the
# instance directly. Every component slot accepts a pre-built instance
# (``domain``, ``collider``, ``linear_integrator``, ``rotation_integrator``,
# ``force_model``, ``force_manager``, ``bonded_force_model``, ``mat_table``),
# which overrides the corresponding ``*_type`` / ``*_kw`` arguments.

domain = jdem.Domain.create("free", dim=2)
collider = jdem.Collider.create("naive")
system = jdem.System.create(state.shape, domain=domain, collider=collider)
print("free default domain:", system.domain)
print("directly assigned collider:", type(system.collider).__name__)

# %%
# This works for instances of your own custom components too (see the
# custom modules guide).
#
# Post-hoc replacement (``system.domain = domain``) also works, since
# :py:class:`~jaxdem.system.System` is a mutable dataclass, but passing the
# instances to :py:meth:`~jaxdem.system.System.create` is preferred because
# the factory can validate and wire the components together. For swapping
# integrators specifically (e.g. moving from a minimization setup to a
# dynamics setup), prefer :py:meth:`~jaxdem.system.System.with_integrators`,
# which returns a copy with new integrators (and optionally a new ``dt``)
# while preserving everything else — including the domain's *current* box.

system_dyn = system.with_integrators(linear_integrator_type="verlet", dt=1e-3)
print("swapped integrator:", type(system_dyn.linear_integrator).__name__)

# %%
# Summary: three ways to build a system component
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The previous sections showed three equivalent ways to get a component
# into a :py:class:`~jaxdem.system.System`. From most to least common:
#
# 1. **Let** ``System.create`` **build it** — pass the registered name and
#    its constructor arguments::
#
#        system = jdem.System.create(state.shape, domain_type="periodic",
#                                    domain_kw={"box_size": box})
#
# 2. **Build it yourself, then pass the instance** — use the component's
#    own factory (``jdem.Domain.create("periodic", box_size=box)``) or its
#    constructor, and hand the object to ``System.create`` via the slot
#    name (``domain=domain``). Equivalent to 1; useful when you want to
#    inspect or reuse the component.
#
# 3. **Assign it to an existing system** — ``system.domain = domain``.
#    Use this for swapping components after creation; prefer 1 or 2 when
#    first building the system so the factory can validate the combination.
#
# Two things live outside this scheme: writers are plain objects you
# construct directly (``jdem.VTKWriter(...)``), and minimizers are optax
# constructor *functions* (e.g. ``jdem.fire``) passed via ``minimizer=`` /
# ``minimizer_kw=`` — see the integrator guide.

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
# of :math:`n \times \text{stride}` steps. Each snapshot is taken *after*
# its integration steps, so the initial (step-0) state is not stored; to
# record it, save it yourself before the rollout, or pass per-frame
# ``strides`` with a leading ``0`` entry.

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
        domain_kw={"box_size": (2 + i) * jnp.ones(2), "anchor": jnp.zeros(2)},
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
# Some modules can be **deactivated** when creating the system. For the
# integrators, pass ``None`` (preferred; the empty string ``""`` is
# equivalent) to select the base no-op integrator.
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
# **Note:** the domain (``domain_type``), collider (``collider_type``),
# and force model (``force_model_type``) cannot be deactivated — a valid
# type must always be provided.

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
# a ``key`` (:py:func:`jax.random.PRNGKey`) that initializes the system's
# JAX PRNG state. An explicit ``key`` takes precedence over ``seed``. The
# key is stored in ``system.key`` and is available for stochastic
# integrators or custom force functions.

system_rng = jdem.System.create(state.shape, seed=42)
print("PRNG key:", system_rng.key)
