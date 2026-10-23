# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""The Force Manager
----------------------------------------

The :py:class:`~jaxdem.forces.force_manager.ForceManager` collects *all*
non-collider contributions (gravity, external forces, bonded forces, custom
force functions) and aggregates them per rigid body, writing
:py:attr:`~jaxdem.state.State.force` and :py:attr:`~jaxdem.state.State.torque`.

For an overview of the available pairwise force models, how to combine them,
and how to set up species-wise interactions, see the
:doc:`Force Models <../auto_examples/force_model_guide>` guide.
"""

# %%
# The :py:class:`~jaxdem.forces.force_manager.ForceManager` lives on
# :py:attr:`~jaxdem.system.System.force_manager`. It handles:
#
# * **Gravity** — a constant acceleration applied to every particle.
# * **External forces and torques** — accumulated buffers that are applied
#   and cleared each step.
# * **Force functions** — user-supplied callables evaluated every step.
# * **Rigid-body aggregation** — summing per-sphere contributions into
#   per-clump forces/torques.

import jax
import jax.numpy as jnp

import numpy as np

import jaxdem as jdem

# %%
# Gravity
# ~~~~~~~~
# Set gravity via ``force_manager_kw`` at system creation time, or
# modify it directly on the system:

state = jdem.State.create(pos=jnp.zeros((1, 2)))

system = jdem.System.create(
    state.shape,
    force_manager_kw={"gravity": jnp.array([0.0, -9.81])},
)
print("Gravity:", system.force_manager.gravity)

# Change gravity at runtime:
system.force_manager.gravity = jnp.array([0.0, -1.0])
print("Updated gravity:", system.force_manager.gravity)


# %%
# External Forces
# ~~~~~~~~~~~~~~~~~
# You can push forces into the manager's buffers before a step.
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.apply` clears the buffers
# automatically after each call (:py:meth:`~jaxdem.system.System.step` calls
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.apply`).
#
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.add_force` adds a
# force array to **all** particles. Use ``is_com=True`` to apply the force
# at the center of mass (no induced torque) or ``is_com=False`` (default)
# to apply it at the particle position (induces torque via lever arm on
# clumps).

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [3.0, 0.0]]),
)
system = jdem.System.create(state.shape)

# Apply a rightward force to all particles:
push = jnp.array([[1.0, 0.0], [0.5, 0.0]])
system = jdem.ForceManager.add_force(state, system, push)
print("Buffered external force:\n", system.force_manager.external_force)

# %%
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.add_force_at` targets
# a specific particle by its array index:

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
state, system = jdem.System.initialize(state, system)
state, system = system.step(state, system)
assert bool(jnp.all(system.force_manager.external_force == 0))
assert bool(jnp.all(system.force_manager.external_torque == 0))
print("External buffers are empty after one physical step.")


# %%
# Custom Force Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~
# For forces that depend on the current state and do not follow the
# pairwise interaction pattern, you can register a custom callable
# (usually a function) via ``force_manager_kw``.
# The callable must have the signature ``(pos, state, system) -> (force, torque)``.
# You can pair an optional energy function with signature
# ``(pos, state, system) -> energy`` with it.


def harmonic_trap(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> tuple[jax.Array, jax.Array]:
    """Pull every particle toward the origin."""
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
    force_manager_kw={
        "force_functions": [
            (harmonic_trap, harmonic_trap_energy),
        ],
    },
    dt=0.02,
)
print("Registered force functions:", len(system.force_manager.force_functions))

# %%
# The system calls the force functions automatically every step:
state, system = jdem.System.initialize(state, system)
state, system = system.step(state, system, n=20)
# With mass=k=1 and initial x=2, v=0, x(t)=2*cos(t). This short
# check verifies the force-function wiring without a long dynamics run.
expected_x = 2.0 * np.cos(float(system.time))
np.testing.assert_allclose(state.pos[0, 0], expected_x, atol=3e-5)
print("Trap position and analytical expectation:", state.pos[0, 0], expected_x)


# %%
# Force Function Formats
# ~~~~~~~~~~~~~~~~~~~~~~~~
# :py:attr:`~jaxdem.forces.force_manager.ForceManager.force_functions` accepts
# several shorthand formats:
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
#    :py:attr:`~jaxdem.state.State.force` / :py:attr:`~jaxdem.state.State.torque`.
# 2. The **force manager** then:
#
#    a. Evaluates all registered
#       :py:attr:`~jaxdem.forces.force_manager.ForceManager.force_functions`.
#    b. Adds external force / torque buffers.
#    c. Adds gravity (applied at the center of mass).
#    d. Computes torques induced by particle-position forces via
#       :math:`\boldsymbol{\tau} = \mathbf{r}_p \times \mathbf{F}`.
#    e. Aggregates rigid bodies:
#       sums per-sphere contributions per clump with `jax.ops.segment_sum <https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html>`_,
#       then broadcasts the result back to all constituent spheres.
#    f. Clears the external buffers.
#
# This means that :py:attr:`~jaxdem.state.State.force` after
# :py:meth:`~jaxdem.system.System.step` already contains the
# fully aggregated clump forces and torques.


# %%
# Computing Potential Energy
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The force manager exposes
# :py:meth:`~jaxdem.forces.force_manager.ForceManager.compute_potential_energy`
# which evaluates gravitational PE and any registered energy functions.
# The gravity term follows the convention :math:`U = -m\,(\mathbf{g} \cdot \mathbf{r})`,
# so with ``gravity=(0, -9.81)`` a particle at height :math:`h > 0` has
# **positive** energy :math:`m g h`:

state = jdem.State.create(
    pos=jnp.array([[0.0, 1.0]]),
    mass=jnp.array([2.0]),
)
system = jdem.System.create(
    state.shape,
    force_manager_kw={"gravity": jnp.array([0.0, -9.81])},
)

pe = system.force_manager.compute_potential_energy(state, system)
np.testing.assert_allclose(pe, 19.62, rtol=1e-6)
print(f"Gravitational PE: {pe}")  # U = -m (g . r) = m * 9.81 * 1.0 = +19.62

# %%
# Note the asymmetry with the collider: the force manager returns a bare
# scalar, while
# :py:meth:`~jaxdem.colliders.Collider.compute_potential_energy` returns
# ``(state, system, pe)`` because colliders may rebuild caches. Unpack it as:
#
# .. code-block:: python
#
#     _, _, pe = system.collider.compute_potential_energy(state, system)


# %%
# Infinite Wall Example
# ~~~~~~~~~~~~~~~~~~~~~~~
# You can also use the force manager to implement an infinite wall
# as a custom force function:

from collections.abc import Callable

from jaxdem.utils.linalg import unit


def make_wall(
    point: jax.Array, normal: jax.Array, stiffness: float = 1.0
) -> tuple[Callable, Callable]:
    point = jnp.asarray(point, dtype=float)
    normal = unit(jnp.asarray(normal, dtype=float))
    stiffness = jnp.asarray(stiffness, dtype=float)

    def energy_fn(pos: jax.Array, state: jdem.State, system: jdem.System) -> jax.Array:
        dist = jnp.dot(pos - point, normal) - state.rad
        delta = jnp.minimum(0.0, dist)
        return 0.5 * stiffness * jnp.square(delta)

    def force_fn(
        pos: jax.Array, state: jdem.State, system: jdem.System
    ) -> tuple[jax.Array, jax.Array]:
        dist = jnp.dot(pos - point, normal) - state.rad
        delta = jnp.minimum(0.0, dist)
        f = (-stiffness * delta)[..., None] * normal
        return f, jnp.zeros_like(state.torque)

    return force_fn, energy_fn


system = jdem.System.create(
    state.shape,
    force_manager_kw={
        "force_functions": [
            make_wall(point=[0.0, 0.5], normal=[0.0, 1.0], stiffness=100.0)
        ],
    },
)

# The sphere overlaps this wall by 0.5, so stiffness 100 gives an upward
# force of 50. evaluate_forces inspects this load without advancing time.
state, system = jdem.System.evaluate_forces(state, system)
np.testing.assert_allclose(state.force, [[0.0, 50.0]], rtol=1e-6)
print("Wall force on the overlapping sphere:", state.force)
