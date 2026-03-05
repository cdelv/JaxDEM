r"""
The Force Manager
----------------------------------------

The :py:class:`~jaxdem.forces.force_manager.ForceManager` collects *all*
non-collider contributions (gravity, external forces, bonded forces, custom
force functions) and performs the final rigid-body aggregation that writes
``state.force`` and ``state.torque``.

For an overview of the available pairwise force models, how to combine them,
and how to set up species-wise interactions, see the
:doc:`Force Models <../auto_examples/force_model_guide>` guide.
"""

# %%
# The :py:class:`~jaxdem.forces.force_manager.ForceManager` lives on
# ``system.force_manager``. It handles:
#
# * **Gravity** — a constant acceleration applied to every particle.
# * **External forces and torques** — accumulated buffers that are applied
#   and cleared each step.
# * **Force functions** — user-supplied callables evaluated every step.
# * **Rigid-body aggregation** — summing per-sphere contributions into
#   per-clump forces/torques.

import jax
import jax.numpy as jnp
import jaxdem as jdem

# %%
# Gravity
# ~~~~~~~~
# Gravity is set via ``force_manager_kw`` at system creation time, or
# modified directly on the system:

state = jdem.State.create(pos=jnp.zeros((1, 2)))

system = jdem.System.create(
    state.shape,
    force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
)
print("Gravity:", system.force_manager.gravity)

# Change gravity at runtime:
system.force_manager.gravity = jnp.array([0.0, -1.0])
print("Updated gravity:", system.force_manager.gravity)


# %%
# External Forces
# ~~~~~~~~~~~~~~~~~
# You can push forces into the manager's buffers before a step. The buffers
# are cleared automatically after each ``apply`` call (which happens inside
# ``system.step``).
#
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.add_force` adds a
# force array to **all** particles. Use ``is_com=True`` to apply the force
# at the center of mass (no induced torque) or ``is_com=False`` (default)
# to apply it at the particle position (induces torque via lever arm on
# clumps).

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
)
system = jdem.System.create(state.shape)

# Apply a rightward force to all particles:
push = jnp.array([[1.0, 0.0], [0.5, 0.0]])
system = jdem.ForceManager.add_force(state, system, push)
print("Buffered external force:\n", system.force_manager.external_force)

# %%
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.add_force_at` targets
# a specific particle by its ``unique_id``:

system = jdem.ForceManager.add_force_at(
    state, system, force=jnp.array([[0.0, -5.0]]), idx=jnp.array([0])
)
print("After add_force_at:\n", system.force_manager.external_force)


# %%
# External Torques
# ~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.add_torque` and
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.add_torque_at` work
# the same way for torques:

system = jdem.ForceManager.add_torque(state, system, torque=jnp.array([[0.1], [0.2]]))
print("Buffered external torque:\n", system.force_manager.external_torque)


# %%
# Custom Force Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~
# For forces that depend on the current state and do not follow the
# pairwise interaction pattern, you can register a custom callable
# (usually a function) via ``force_manager_kw``.
# The callable must have the signature ``(pos, state, system) -> (force, torque)``.
# An optional energy function with signature
# ``(pos, state, system) -> energy`` can be paired with it.

from typing import Tuple


def harmonic_trap(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> Tuple[jax.Array, jax.Array]:
    """Pull every particle towards the origin."""
    k = 1.0
    return -k * pos, jnp.zeros_like(state.torque)


def harmonic_trap_energy(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> jax.Array:
    k = 1.0
    return 0.5 * k * jnp.sum(pos**2, axis=-1)


state = jdem.State.create(pos=jnp.array([[2.0, 0.0]]))
system = jdem.System.create(
    state.shape,
    force_manager_kw=dict(
        force_functions=[
            (harmonic_trap, harmonic_trap_energy),
        ],
    ),
    dt=1e-1,
)
print("Registered force functions:", len(system.force_manager.force_functions))

# %%
# The force functions are called automatically every step:
state, system = system.step(state, system, n=16)
print("Position after stepping (pulled to origin):", state.pos)


# %%
# Force Function Formats
# ~~~~~~~~~~~~~~~~~~~~~~~~
# ``force_functions`` accepts several shorthand formats:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Format
#      - Interpretation
#    * - ``func``
#      - Force at particle position, no energy
#    * - ``(func, bool)``
#      - ``bool`` selects COM (``True``) or particle (``False``)
#    * - ``(func, energy_func)``
#      - Force at particle position with energy
#    * - ``(func, energy_func, bool)``
#      - Full specification: force, energy, and COM flag


# %%
# How Forces Are Aggregated
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each integration step, the force pipeline runs as follows:
#
# 1. The **collider** computes pairwise contact forces and writes
#    ``state.force`` / ``state.torque``.
# 2. The **force manager** then:
#
#    a. Evaluates all registered ``force_functions``.
#    b. Adds external force / torque buffers.
#    c. Adds gravity (applied at the center of mass).
#    d. Computes torques induced by particle-position forces via
#       :math:`\boldsymbol{\tau} = \mathbf{r}_p \times \mathbf{F}`.
#    e. Performs rigid-body aggregation:
#       sums per-sphere contributions per clump with ``segment_sum``,
#       then broadcasts the result back to all constituent spheres.
#    f. Clears the external buffers.
#
# This means that ``state.force`` after ``system.step`` already contains the
# fully aggregated clump forces and torques.


# %%
# Computing Potential Energy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The force manager exposes
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.compute_potential_energy`
# which evaluates gravitational PE and any registered energy functions:

state = jdem.State.create(
    pos=jnp.array([[0.0, 1.0]]),
    mass=jnp.array([2.0]),
)
system = jdem.System.create(
    state.shape,
    force_manager_kw=dict(gravity=jnp.array([0.0, -9.81])),
)

pe = system.force_manager.compute_potential_energy(state, system)
print(f"Gravitational PE (mgh): {pe}")  # -m * g . r


# %%
# Infinite Wall Example
# ~~~~~~~~~~~~~~~~~~~~~~~
# Another use-case for the force manager is implementing an infinite wall
# as a custom force function:

from typing import Callable
from jaxdem.utils.linalg import unit


def make_wall(
    point: jax.Array, normal: jax.Array, stiffness: float = 1.0
) -> Tuple[Callable, Callable]:
    point = jnp.asarray(point, dtype=float)
    normal = unit(jnp.asarray(normal, dtype=float))
    stiffness = jnp.asarray(stiffness, dtype=float)

    def energy_fn(pos: jax.Array, state: jdem.State, system: jdem.System) -> jax.Array:
        dist = jnp.dot(pos - point, normal) - state.rad
        delta = jnp.minimum(0.0, dist)
        return 0.5 * stiffness * jnp.square(delta)

    def force_fn(
        pos: jax.Array, state: jdem.State, system: jdem.System
    ) -> Tuple[jax.Array, jax.Array]:
        dist = jnp.dot(pos - point, normal) - state.rad
        delta = jnp.minimum(0.0, dist)
        f = -stiffness * delta[..., None] * normal
        return f, jnp.zeros_like(state.torque)

    return force_fn, energy_fn


system = jdem.System.create(
    state.shape,
    force_manager_kw=dict(
        force_functions=[
            make_wall(point=[0.0, 0.5], normal=[0.0, 1.0], stiffness=100.0)
        ],
    ),
)
