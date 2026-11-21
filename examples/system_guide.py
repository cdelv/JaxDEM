# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""The Simulation System
-------------------------

:py:class:`~jaxdem.system.System` holds the global configuration of a
simulation (domain, collider, force model, integrators, material table) along
with evolving scalars such as simulation time, step count, and the PRNG key.

Like :py:class:`~jaxdem.state.State`, :py:class:`~jaxdem.system.System` is a
JAX pytree: numeric leaves (such as ``dt`` and ``box_size``) can vary inside
compiled functions, while component classes and static options select the
compiled specialization.
"""

# %%
# System Creation and Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pass ``state=state`` (or ``state.shape``) to
# :py:meth:`~jaxdem.system.System.create`. Passing ``state=state`` also
# forwards particle data to colliders that need it during setup (such as cell
# lists and neighbor lists).
#
# Always call :py:meth:`~jaxdem.system.System.initialize` once before stepping
# to compute initial forces and build search caches:

import dataclasses

import jax
import jax.numpy as jnp

import jaxdem as jdem

state = jdem.State.create(pos=jnp.zeros((1, 2)))
system = jdem.System.create(state=state)
state, system = jdem.System.initialize(state, system)
state, system = jdem.System.step(state, system)
print("Step count:", int(system.step_count), "| time:", float(system.time))

# %%
# Every operation on :py:class:`~jaxdem.state.State` and
# :py:class:`~jaxdem.system.System` (:py:meth:`~jaxdem.system.System.step`,
# :py:meth:`~jaxdem.system.System.trajectory_rollout`,
# :py:meth:`~jaxdem.state.State.merge`, :py:meth:`~jaxdem.state.State.stack`)
# is a **static method**, so ``system.step(state, system)`` and
# ``jdem.System.step(state, system)`` are equivalent and compose cleanly with
# ``jax.jit`` and ``jax.vmap``.

# %%
# Configuring Components
# ~~~~~~~~~~~~~~~~~~~~~~
# There are three ways to attach or swap a component:
#
# 1. **By registered name and ``*_kw``** (most common) — let
#    :py:meth:`~jaxdem.system.System.create` construct and validate it.
# 2. **By pre-built instance** — pass ``domain=...``, ``collider=...``,
#    ``force_model=...``, etc. directly (overrides ``*_type``/``*_kw`` and works
#    for custom subclasses from :doc:`../auto_examples/custom_modules_guide`).
# 3. **Post-creation swap** — use ``dataclasses.replace`` to change an
#    integrator or ``dt`` while preserving the rest of the system (including
#    current domain geometry).

# 1. Name + keyword dict:
system = jdem.System.create(
    state.shape,
    domain_type="periodic",
    domain_kw={"box_size": 10.0 * jnp.ones(2), "anchor": jnp.zeros(2)},
)

# 2. Pre-built instances:
domain = jdem.Domain.create("free", dim=2)
collider = jdem.Collider.create("naive")
system = jdem.System.create(state.shape, domain=domain, collider=collider)

# 3. Swap integrator while keeping everything else:
system_dyn = dataclasses.replace(
    system,
    linear_integrator=jdem.LinearIntegrator.create("verlet"),
    dt=jnp.asarray(1e-3, dtype=float),
)
print("Swapped integrator:", type(system_dyn.linear_integrator).__name__)

# %%
# Minimizers are Optax-style constructor functions (e.g.
# :py:func:`~jaxdem.minimizers.optimizers.fire`) passed via ``minimizer=`` /
# ``minimizer_kw=``; see :doc:`../auto_examples/integrator_guide`.

# %%
# Stepping, Rollouts, and Diagnostics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.system.System.step` runs a static number of steps ``n``
# inside ``jax.lax.fori_loop`` (use :py:meth:`~jaxdem.system.System.step_dynamic`
# when the step count itself is a traced JAX value).
#
# :py:meth:`~jaxdem.system.System.trajectory_rollout` records ``n`` snapshots
# separated by ``stride`` steps (or custom per-frame ``strides``, where a
# leading ``0`` records step 0). Pass a lightweight ``save_fn(state, system)``
# to record only the observables you need instead of full states:

state = jdem.State.create(jnp.zeros((1, 2)), vel=jnp.array([[0.1, 0.0]]))
state, system = jdem.System.initialize(state, system)

state, system, (traj_state, traj_system) = jdem.System.trajectory_rollout(
    state, system, n=10, stride=2  # 20 total steps
)
print("Full trajectory pos shape:", traj_state.pos.shape)

# Record only positions at irregular intervals (including step 0):
_, _, pos_trace = jdem.System.trajectory_rollout(
    state,
    system,
    strides=jnp.array([0, 1, 2, 5]),
    save_fn=lambda st, sys: st.pos,
)
print("Custom save_fn trace shape:", pos_trace.shape)

# %%
# Use :py:meth:`~jaxdem.system.System.evaluate_forces` to recompute forces and
# search caches on an edited snapshot without advancing time or history.
# Call :py:meth:`~jaxdem.system.System.validate` on host before long runs and
# :py:meth:`~jaxdem.system.System.check_overflow` afterward to raise if any
# bounded spatial-search buffer overflowed:

evaluated_state, evaluated_system = jdem.System.evaluate_forces(state, system)
print("Evaluation preserved time:", bool(evaluated_system.time == system.time))
system.validate(state)
system.check_overflow()

# %%
# You can also register pure callbacks with ``user_pre_step_actions`` and
# ``user_post_step_actions`` on :py:meth:`~jaxdem.system.System.create` (each
# taking ``(state, system)`` and returning ``(state, system)``) to drive
# prescribed motion or boundary strain every step.

# %%
# Batched Simulations
# ~~~~~~~~~~~~~~~~~~~
# Build a batch of independent ``(State, System)`` pairs with ``jax.vmap`` (or
# :py:meth:`~jaxdem.system.System.stack` / :py:meth:`~jaxdem.system.System.unstack`).
# Rank-3 ``(B, N, dim)`` states paired with rank-1 ``(B,)`` systems are handled
# natively by :py:meth:`~jaxdem.system.System.initialize`,
# :py:meth:`~jaxdem.system.System.step`, and
# :py:meth:`~jaxdem.system.System.trajectory_rollout` without an explicit
# outer ``jax.vmap`` (use ``jax.vmap`` explicitly for higher-rank grids or
# custom per-replica logic):


def make_sim(i: jax.Array) -> tuple[jdem.State, jdem.System]:
    st = jdem.State.create(pos=jnp.array([[1.0, 1.0]]), rad=jnp.array([0.1]))
    sys = jdem.System.create(
        st.shape,
        domain_type="reflect",
        domain_kw={"box_size": (2.0 + i) * jnp.ones(2), "anchor": jnp.zeros(2)},
    )
    return jdem.System.initialize(st, sys)


state_b, system_b = jax.vmap(make_sim)(jnp.arange(3))
state_b, system_b = jdem.System.step(state_b, system_b, n=10)
print("Batched pos shape:", state_b.pos.shape, "| box sizes:", system_b.domain.box_size[:, 0])

stacked_sys = jdem.System.stack([system, system, system])
print("Stacked dt shape:", stacked_sys.dt.shape, "| unstacked:", len(jdem.System.unstack(stacked_sys)))

# %%
# Deactivating Components
# ~~~~~~~~~~~~~~~~~~~~~~~
# Pass ``None`` (preferred; ``""`` is equivalent for integrators and colliders)
# to disable optional modules:
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
#      - No custom external force/torque callbacks.

system_frozen = jdem.System.create(
    state.shape,
    linear_integrator_type=None,
    rotation_integrator_type=None,
)
print("Frozen integrator:", type(system_frozen.linear_integrator).__name__)

# %%
# Random Number Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Pass an integer ``seed`` or an explicit JAX PRNG ``key`` (which takes
# precedence) to :py:meth:`~jaxdem.system.System.create`. JaxDEM stores it in
# :py:attr:`system.key <jaxdem.system.System.key>` for stochastic integrators
# and custom callbacks:

system_rng = jdem.System.create(state.shape, seed=42)
print("PRNG key:", system_rng.key)
