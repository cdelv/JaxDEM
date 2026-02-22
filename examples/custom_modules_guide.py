r"""
Custom Module Registration and System Usage
-------------------------------------------------

JaxDEM components are created through factory registries. This makes it easy
to add custom modules and use them with :py:meth:`~jaxdem.system.System.create`.

To register a new module, the pattern is always the same:

1. Inherit from the corresponding base class
   (:py:class:`~jaxdem.forces.ForceModel`,
   :py:class:`~jaxdem.domains.Domain`,
   :py:class:`~jaxdem.colliders.Collider`,
   :py:class:`~jaxdem.integrators.LinearIntegrator`, etc.).
2. Implement the required abstract/interface methods for that base class.
3. Register the class with ``@<Base>.register("your_key")``.

This guide shows how to:

- define and register custom modules,
- instantiate them via ``*_type`` and ``*_kw`` arguments,
- and pass pre-built module objects directly to a ``System``.
"""

# %%
# Setup
# ~~~~~
# We will create several custom modules:
#
# - a custom :py:class:`~jaxdem.forces.ForceModel`
# - a custom :py:class:`~jaxdem.domains.Domain`
# - a custom :py:class:`~jaxdem.colliders.Collider`
# - custom linear and rotation integrators

from dataclasses import dataclass
from functools import partial
from typing import Tuple, cast

import jax
import jax.numpy as jnp
import jaxdem as jdem

# %%
# Custom Force Model
# ~~~~~~~~~~~~~~~~~~
# This force law applies a simple linear attraction between particle pairs:
#
# .. math::
#
#    \mathbf{F}_{ij} = -k\,(\mathbf{r}_i - \mathbf{r}_j)
#
# It does not require any material-table properties.
#
# Registration reminder:
# - inherit from :py:class:`~jaxdem.forces.ForceModel`
# - implement ``force`` and ``energy``


@jdem.ForceModel.register("pairattractor")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PairAttractor(jdem.ForceModel):
    k: float = 0.1

    @staticmethod
    @partial(jax.jit, inline=True)
    def force(
        i: int, j: int, pos: jax.Array, state: jdem.State, system: jdem.System
    ) -> Tuple[jax.Array, jax.Array]:
        rij = system.domain.displacement(pos[i], pos[j], system)
        mask = jnp.asarray(i != j, dtype=rij.dtype)[..., None]
        model = cast(PairAttractor, system.force_model)
        force = -model.k * rij * mask
        torque = jnp.zeros_like(state.torque[j])
        return force, torque

    @staticmethod
    @partial(jax.jit, inline=True)
    def energy(
        i: int, j: int, pos: jax.Array, state: jdem.State, system: jdem.System
    ) -> jax.Array:
        rij = system.domain.displacement(pos[i], pos[j], system)
        model = cast(PairAttractor, system.force_model)
        return 0.5 * model.k * jnp.sum(rij * rij, axis=-1) * (i != j)


print(
    "ForceModel registry contains pairattractor:",
    "pairattractor" in jdem.ForceModel._registry,
)
print("Registered ForceModels:", list(jdem.ForceModel._registry.keys()))

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [2.0, 0.0]]),
    rad=jnp.array([0.4, 0.4]),
)
system = jdem.System.create(
    state.shape,
    force_model_type="pairattractor",
    force_model_kw=dict(k=0.2),
)
state, system = system.step(state, system, n=5)
print("Custom force model:", type(system.force_model).__name__)
print("Positions after 5 steps:\n", state.pos)


# %%
# Custom Domain
# ~~~~~~~~~~~~~
# This domain recenters particles every step so the center of mass stays at
# the origin. It reuses the default ``Domain.Create`` for ``box_size``/``anchor``.
#
# Registration reminder:
# - inherit from :py:class:`~jaxdem.domains.Domain`
# - implement the relevant interface methods (here: ``apply``)


@jdem.Domain.register("centered")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CenteredDomain(jdem.Domain):
    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    def apply(state: jdem.State, system: jdem.System) -> Tuple[jdem.State, jdem.System]:
        center = jnp.mean(state.pos, axis=-2)
        state.pos_c -= center
        return state, system


print("Domain registry contains centered:", "centered" in jdem.Domain._registry)
print("Registered Domains:", list(jdem.Domain._registry.keys()))

state = jdem.State.create(
    pos=jnp.array([[2.0, 0.0], [4.0, 0.0]]),
    vel=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
)
system = jdem.System.create(
    state.shape,
    domain_type="centered",
    force_model_type="pairattractor",
)
state, system = system.step(state, system, n=3)
print("Custom domain:", type(system.domain).__name__)
print("Mean position after centering:", jnp.mean(state.pos, axis=0))


# %%
# Custom Collider
# ~~~~~~~~~~~~~~~
# This collider disables all pair contacts by forcing zero force and torque.
# In reallity, this is the same as passing no collider to the system object,
# but it serves as a simple example of a custom collider.
#
# Registration reminder:
# - inherit from :py:class:`~jaxdem.colliders.Collider`
# - implement interface methods (at minimum ``compute_force``, and for full
#   compatibility also ``compute_potential_energy`` and ``create_neighbor_list``)


@jdem.Collider.register("nocontact")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NoContactCollider(jdem.Collider):
    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    def compute_force(
        state: jdem.State, system: jdem.System
    ) -> Tuple[jdem.State, jdem.System]:
        state.force *= 0
        state.torque *= 0
        return state, system

    @staticmethod
    @partial(jax.jit, inline=True)
    def compute_potential_energy(state: jdem.State, system: jdem.System) -> jax.Array:
        return jnp.zeros_like(state.mass)

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",))
    def create_neighbor_list(
        state: jdem.State,
        system: jdem.System,
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple[jdem.State, jdem.System, jax.Array, jax.Array]:
        del cutoff
        nl = -jnp.ones((state.N, max_neighbors), dtype=int)
        overflow = jnp.asarray(False)
        return state, system, nl, overflow


print("Collider registry contains nocontact:", "nocontact" in jdem.Collider._registry)
print("Registered Colliders:", list(jdem.Collider._registry.keys()))

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
    vel=jnp.array([[0.5, 0.0], [-0.5, 0.0]]),
    rad=jnp.array([0.6, 0.6]),
)
system = jdem.System.create(
    state.shape,
    collider_type="nocontact",
    force_model_type="pairattractor",
)
state, system = system.step(state, system, n=2)
print("Custom collider:", type(system.collider).__name__)
print("Forces with nocontact collider:\n", state.force)


# %%
# Custom Integrators
# ~~~~~~~~~~~~~~~~~~
# Here we register:
#
# - ``DampedEuler`` for linear motion,
# - ``FrozenRotation`` for angular motion.
#
# Registration reminder:
# - inherit from :py:class:`~jaxdem.integrators.LinearIntegrator` or
#   :py:class:`~jaxdem.integrators.RotationIntegrator`
# - implement the needed step methods (here: ``step_after_force``)


@jdem.LinearIntegrator.register("dampedeuler")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DampedEuler(jdem.LinearIntegrator):
    damping: float = 0.2

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> Tuple[jdem.State, jdem.System]:
        accel = state.force / state.mass[..., None]
        active = (1 - state.fixed)[..., None]
        integrator = cast(DampedEuler, system.linear_integrator)
        damp = 1.0 - system.dt * integrator.damping

        state.vel = damp * state.vel + system.dt * accel * active
        state.vel *= active
        state.pos_c += system.dt * state.vel
        return state, system


@jdem.RotationIntegrator.register("frozenrotation")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class FrozenRotation(jdem.RotationIntegrator):
    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> Tuple[jdem.State, jdem.System]:
        state.ang_vel *= 0
        return state, system


print(
    "LinearIntegrator registry contains dampedeuler:",
    "dampedeuler" in jdem.LinearIntegrator._registry,
)
print("Registered LinearIntegrators:", list(jdem.LinearIntegrator._registry.keys()))
print(
    "RotationIntegrator registry contains frozenrotation:",
    "frozenrotation" in jdem.RotationIntegrator._registry,
)
print("Registered RotationIntegrators:", list(jdem.RotationIntegrator._registry.keys()))

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [2.0, 0.0]]),
    rad=jnp.array([0.4, 0.4]),
)
system = jdem.System.create(
    state.shape,
    force_model_type="pairattractor",
    linear_integrator_type="dampedeuler",
    linear_integrator_kw=dict(damping=0.5),
    rotation_integrator_type="frozenrotation",
)

state, system = system.step(state, system, n=10)
print("Custom linear integrator:", type(system.linear_integrator).__name__)
print("Custom rotation integrator:", type(system.rotation_integrator).__name__)
print("Velocity after damping:\n", state.vel)


# %%
# Passing Module Objects Directly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``System.create`` is convenient for factory-based construction. You can also
# build a base system and swap modules directly with pre-built objects.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([0.4, 0.4]),
)
system = jdem.System.create(state.shape)

system.force_model = PairAttractor(k=0.05)
system.domain = CenteredDomain.Create(dim=state.dim)
system.collider = NoContactCollider()
system.linear_integrator = DampedEuler(damping=0.8)
system.rotation_integrator = FrozenRotation()

state, system = system.step(state, system, n=3)
print("Directly assigned force model:", type(system.force_model).__name__)
print("Directly assigned domain:", type(system.domain).__name__)
print("Directly assigned collider:", type(system.collider).__name__)
print("Directly assigned linear integrator:", type(system.linear_integrator).__name__)
print(
    "Directly assigned rotation integrator:", type(system.rotation_integrator).__name__
)


# %%
# Notes on Registration
# ~~~~~~~~~~~~~~~~~~~~~
# - Registration keys are case-insensitive.
# - Registrations are process-local: define/register your custom classes before
#   calling ``System.create`` with the corresponding ``*_type``.
# - All custom modules should be JAX pytrees; using
#   ``@jax.tree_util.register_dataclass`` on dataclasses is the recommended path.
# - The "proof" that registration worked is that the key appears in the relevant
#   registry dictionary.

print(
    "Registered custom force model key exists:",
    "pairattractor" in jdem.ForceModel._registry,
)
print("Registered custom domain key exists:", "centered" in jdem.Domain._registry)
print("Registered custom collider key exists:", "nocontact" in jdem.Collider._registry)
print(
    "Registered custom linear integrator key exists:",
    "dampedeuler" in jdem.LinearIntegrator._registry,
)
print(
    "Registered custom rotation integrator key exists:",
    "frozenrotation" in jdem.RotationIntegrator._registry,
)
