# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Custom Module Registration and System Usage
-----------------------------------------------

Extend JaxDEM by subclassing any component base class
(:py:class:`~jaxdem.forces.ForceModel`, :py:class:`~jaxdem.domains.Domain`,
:py:class:`~jaxdem.colliders.Collider`,
:py:class:`~jaxdem.integrators.LinearIntegrator`,
:py:class:`~jaxdem.integrators.RotationIntegrator`), decorating it as a JAX
pytree dataclass, and registering it with ``@<Base>.register("key")``.

Registered classes work seamlessly via ``*_type`` / ``*_kw`` in
:py:meth:`~jaxdem.system.System.create`, or you can pass pre-built instances
directly (see :doc:`../auto_examples/system_guide`).
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

import jaxdem as jdem

# %%
# Custom Force Model (Finite Range, Static Fields, and ``Create``)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every :py:class:`~jaxdem.forces.ForceModel` implements
# :py:meth:`~jaxdem.forces.ForceModel.force` and
# :py:meth:`~jaxdem.forces.ForceModel.energy`. Two extra patterns are useful
# in real force laws:
#
# * **Finite-range search bounds:** Override
#   :py:meth:`~jaxdem.forces.ForceModel.search_radii` so cell-list and
#   neighbor-list colliders know the interaction reach (nonzero forces must
#   fit inside ``search_radii[i] + search_radii[j]``).
# * **Static vs dynamic fields & ``Create``:** Array/float leaves can vary
#   without recompiling; mark compile-time options static with
#   ``jax.tree.static()``. Define a ``Create`` static/classmethod to validate
#   or preprocess keyword arguments passed via ``force_model_kw``.


# * **Vectorized neighbor indices:** Accelerated colliders may pass a 1-D array
#   of neighbor indices ``j``, so always compute norms along ``axis=-1`` (or use
#   :py:func:`~jaxdem.utils.linalg.norm`).


@jdem.ForceModel.register("soft_cutoff_spring")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SoftCutoffSpring(jdem.ForceModel):
    k: jax.Array = field(default_factory=lambda: jnp.asarray(1.0))
    cutoff_scale: float = jax.tree.static(default=1.2)

    @staticmethod
    def Create(k: float = 1.0, cutoff_scale: float = 1.2) -> "SoftCutoffSpring":
        if cutoff_scale <= 0.0:
            raise ValueError("cutoff_scale must be positive.")
        return SoftCutoffSpring(k=jnp.asarray(k, dtype=float), cutoff_scale=float(cutoff_scale))

    def search_radii(self, state: jdem.State, system: jdem.System) -> jax.Array:
        return jnp.maximum(state._rad, self.cutoff_scale * state.rad)

    @staticmethod
    @jax.jit(inline=True)
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        rij = system.domain.displacement(pos[i], pos[j], system)
        r = jnp.linalg.norm(rij, axis=-1)
        safe_r = jnp.where(r == 0.0, 1.0, r)
        rc = system.force_model.cutoff_scale * (state.rad[i] + state.rad[j])
        delta = jnp.maximum(0.0, rc - r) * (i != j)
        f = (system.force_model.k * delta / safe_r)[..., None] * rij
        return f, jnp.zeros_like(state.torque[j]), history

    @staticmethod
    @jax.jit(inline=True)
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: jdem.State,
        system: jdem.System,
        history: jax.Array,
    ) -> jax.Array:
        rij = system.domain.displacement(pos[i], pos[j], system)
        r = jnp.linalg.norm(rij, axis=-1)
        rc = system.force_model.cutoff_scale * (state.rad[i] + state.rad[j])
        delta = jnp.maximum(0.0, rc - r) * (i != j)
        return 0.5 * system.force_model.k * delta * delta


state = jdem.State.create(pos=jnp.array([[0.0, 0.0], [1.1, 0.0]]), rad=jnp.full(2, 0.5))
system = jdem.System.create(
    state=state,
    force_model_type="soft_cutoff_spring",
    force_model_kw={"k": 2.0, "cutoff_scale": 1.2},
    collider_type="cell_list",
)
state, system = jdem.System.initialize(state, system)
state, system = jdem.System.evaluate_forces(state, system)
print("Custom force model:", type(system.force_model).__name__, "| force[0]:", state.force[0])

# %%
# Custom Domain
# ~~~~~~~~~~~~~
# Subclass :py:class:`~jaxdem.domains.Domain` and override
# :py:meth:`~jaxdem.domains.Domain.apply` (and/or ``displacement``, ``shift``).
# This example shifts coordinates so their geometric mean sits at the origin
# whenever :py:meth:`~jaxdem.domains.Domain.apply` runs:


@jdem.Domain.register("centered")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CenteredDomain(jdem.Domain):
    @staticmethod
    @jax.jit(inline=True)
    def apply(state: jdem.State, system: jdem.System) -> tuple[jdem.State, jdem.System]:
        state.pos_c -= jnp.mean(state.pos, axis=-2)
        return state, system


state = jdem.State.create(pos=jnp.array([[2.0, 0.0], [4.0, 0.0]]))
system = jdem.System.create(state.shape, domain_type="centered", force_model_type="soft_cutoff_spring")
state, system = system.domain.apply(state, system)
print("Mean position after CenteredDomain.apply:", jnp.mean(state.pos, axis=0))

# %%
# Custom Collider
# ~~~~~~~~~~~~~~~
# Subclass :py:class:`~jaxdem.colliders.Collider` and implement
# :py:meth:`~jaxdem.colliders.Collider.compute_force`,
# :py:meth:`~jaxdem.colliders.Collider.compute_potential_energy`, and
# :py:meth:`~jaxdem.colliders.Collider.create_neighbor_list`:


@jdem.Collider.register("nocontact")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NoContactCollider(jdem.Collider):
    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
    def compute_force(
        state: jdem.State,
        system: jdem.System,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jdem.State, jdem.System]:
        state.force *= 0
        state.torque *= 0
        return state, system

    @staticmethod
    @jax.jit(inline=True)
    def compute_potential_energy(
        state: jdem.State, system: jdem.System
    ) -> tuple[jdem.State, jdem.System, jax.Array]:
        return state, system, jnp.asarray(0.0)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_neighbor_list(
        state: jdem.State, system: jdem.System, cutoff: float, max_neighbors: int
    ) -> tuple[jdem.State, jdem.System, jax.Array, jax.Array]:
        del cutoff
        return state, system, -jnp.ones((state.N, max_neighbors), dtype=int), jnp.asarray(False)


system = jdem.System.create(state.shape, collider_type="nocontact")
print("Custom collider:", type(system.collider).__name__)

# %%
# Custom Integrators
# ~~~~~~~~~~~~~~~~~~
# Subclass :py:class:`~jaxdem.integrators.LinearIntegrator` or
# :py:class:`~jaxdem.integrators.RotationIntegrator` and override
# :py:meth:`~jaxdem.integrators.Integrator.step_before_force` and/or
# :py:meth:`~jaxdem.integrators.Integrator.step_after_force` (see the full
# per-step order in :doc:`../auto_examples/integrator_guide`). Respect
# ``state.fixed`` so fixed particles keep their prescribed velocity while
# ignoring forces:


@jdem.LinearIntegrator.register("dampedeuler")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DampedEuler(jdem.LinearIntegrator):
    damping: float = 0.2

    @staticmethod
    @jax.jit(inline=True)
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> tuple[jdem.State, jdem.System]:
        accel = state.force / state.mass[..., None]
        active = (~state.fixed)[..., None]
        damp = 1.0 - system.dt * system.linear_integrator.damping
        state.vel = jnp.where(active, damp * state.vel + system.dt * accel, state.vel)
        state.pos_c += system.dt * state.vel
        return state, system


state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.1, 0.0]]),
    rad=jnp.full(2, 0.5),
    vel=jnp.array([[0.1, 0.0], [0.0, 0.0]]),
    fixed=jnp.array([True, False]),
)
system = jdem.System.create(
    state=state,
    force_model_type="soft_cutoff_spring",
    linear_integrator_type="dampedeuler",
    linear_integrator_kw={"damping": 0.5},
    rotation_integrator_type=None,
)
state, system = jdem.System.initialize(state, system)
state, system = jdem.System.step(state, system, n=10)
print("DampedEuler velocities (particle 0 fixed):\n", state.vel)

# %%
# Registration and Checkpointing Notes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * **Key normalization:** Registry keys ignore case, underscores, spaces, and
#   hyphens (``"SoftCutoffSpring"`` == ``"soft_cutoff_spring"``).
# * **Direct instances:** You can also pass ``force_model=SoftCutoffSpring.Create(k=0.5)``
#   directly to :py:meth:`~jaxdem.system.System.create` without using ``*_type``.
# * **Checkpointing:** Registrations are process-local. To reload checkpoints
#   that use a custom registered class, define it in an importable module and
#   import that module before calling
#   :py:meth:`~jaxdem.writers.checkpoints.CheckpointLoader.load` (see
#   :doc:`../auto_examples/checkpoint_guide`).
