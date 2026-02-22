r"""
The Simulation Domain
----------------------------------------

The :py:class:`~jaxdem.domains.Domain` defines the spatial boundaries and
boundary conditions of a simulation. It controls two things:

1. **Displacement** — how the relative displacement vector between two
   particles is computed (important for periodic boundary conditions).
2. **Boundary enforcement** — how particles are constrained to remain inside
   the simulation box (reflection, wrapping, or nothing at all).

JaxDEM supports four domain types:

* ``"free"`` — unbounded space, no boundary effects.
* ``"periodic"`` — periodic (minimum-image) boundary conditions.
* ``"reflect"`` — reflective walls with impulse-based collision for general
  rigid bodies (spheres and clumps).
* ``"reflectsphere"`` — a faster reflective domain optimised for
  sphere-only simulations.

Let's explore each one.
"""

# %%
# Domain Creation
# ~~~~~~~~~~~~~~~~~
# A domain is usually created through
# :py:meth:`~jaxdem.system.System.create` by passing ``domain_type`` and,
# optionally, ``domain_kw``:

import jax
import jax.numpy as jnp
import jaxdem as jdem

state = jdem.State.create(pos=jnp.zeros((1, 2)))

system = jdem.System.create(
    state.shape,
    domain_type="periodic",
    domain_kw=dict(
        box_size=10.0 * jnp.ones(2),
        anchor=jnp.zeros(2),
    ),
)
print("Domain type:", type(system.domain).__name__)
print("Box size:", system.domain.box_size)
print("Anchor:", system.domain.anchor)

# %%
# You can also build a domain independently and assign it to the system:

domain = jdem.Domain.create("free", dim=2)
system.domain = domain
print("Swapped to:", type(system.domain).__name__)


# %%
# Common Attributes
# ~~~~~~~~~~~~~~~~~~
# Every domain has two core attributes:
#
# * ``box_size`` — the length of the simulation box along each axis, shape
#   ``(dim,)``.
# * ``anchor`` — the minimum-corner coordinate of the box, shape ``(dim,)``.
#
# Together they define an axis-aligned box
# :math:`[\text{anchor},\;\text{anchor} + \text{box\_size}]`.
#
# If you do not supply them, they default to ``ones(dim)`` and ``zeros(dim)``
# respectively.

domain = jdem.Domain.create("reflect", dim=3)
print("box_size:", domain.box_size)
print("anchor:", domain.anchor)


# %%
# Free Domain
# ~~~~~~~~~~~~
# :py:class:`~jaxdem.domains.free.FreeDomain` imposes no boundaries.
# Particles move freely in an unbounded space. The ``box_size`` and
# ``anchor`` are automatically updated each step to tightly encompass all
# particles (some internal algorithms, like spatial hashing in the cell lists colliders,
# need a finite bounding box).

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [3.0, 4.0]]),
    rad=jnp.array([0.5, 0.5]),
)
system = jdem.System.create(state.shape, domain_type="free")

# After a step, the domain auto-fits to the particles:
state, system = system.step(state, system)
print("Free domain box_size:", system.domain.box_size)
print("Free domain anchor:", system.domain.anchor)


# %%
# Periodic Domain
# ~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.domains.periodic.PeriodicDomain` uses the
# **minimum-image convention**: the displacement between two particles is the
# shortest vector connecting them, potentially across a periodic boundary.
#
# The ``periodic`` property returns ``True`` for this domain, which
# lets colliders and other components adapt their behaviour automatically.
#
# Periodic boundary conditions do **not** modify positions during the time
# step (``apply`` is a no-op). To wrap positions back into the primary box
# — for example before saving — use the ``shift`` method.

state = jdem.State.create(
    pos=jnp.array([[0.1, 0.1], [9.9, 9.9]]),
)
system = jdem.System.create(
    state.shape,
    domain_type="periodic",
    domain_kw=dict(box_size=10.0 * jnp.ones(2), anchor=jnp.zeros(2)),
)
print("Is periodic?", system.domain.periodic)

# %%
# The minimum-image displacement between two particles near opposite edges
# wraps around the box:

rij = system.domain.displacement(state.pos[0], state.pos[1], system)
naive = state.pos[0] - state.pos[1]
print("Naive displacement:", naive)
print("Minimum-image displacement:", rij)

# %%
# Call ``shift`` to wrap positions back into the primary image:

state, system = system.domain.shift(state, system)
print("Wrapped positions:\n", state.pos)


# %%
# Reflective Domain
# ~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.domains.reflect.ReflectDomain` reflects particles off
# the walls of the simulation box. It uses impulse-based collision mechanics,
# correctly handling both linear and angular velocity for spheres and clumps.
#
# An optional ``restitution_coefficient`` (default ``1.0``, i.e. perfectly
# elastic) controls how much kinetic energy is retained after a wall
# collision.

state = jdem.State.create(
    pos=jnp.array([[0.5, 0.5]]),
    vel=jnp.array([[-1.0, 0.0]]),
    rad=jnp.array([0.4]),
)
system = jdem.System.create(
    state.shape,
    domain_type="reflect",
    domain_kw=dict(
        box_size=10.0 * jnp.ones(2),
        anchor=jnp.zeros(2),
        restitution_coefficient=1.0,
    ),
)
print("Restitution:", system.domain.restitution_coefficient)

# After stepping, the particle bounces off the left wall:
state, system = system.step(state, system, n=3)
print("Position after bounce:", state.pos)
print("Velocity after bounce:", state.vel)


# %%
# Reflect-Sphere Domain
# ~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.domains.reflect_sphere.ReflectSphereDomain` is a
# lightweight variant of the reflective domain that skips the full
# impulse calculation. It simply mirrors positions and reverses the
# velocity component normal to the boundary.
#
# Use this when your simulation contains **only spheres** (no clumps) for
# better performance.

state = jdem.State.create(
    pos=jnp.array([[0.5, 0.5]]),
    vel=jnp.array([[-1.0, 0.0]]),
    rad=jnp.array([0.4]),
)
system = jdem.System.create(
    state.shape,
    domain_type="reflectsphere",
    domain_kw=dict(box_size=10.0 * jnp.ones(2), anchor=jnp.zeros(2)),
)

state, system = system.step(state, system, n=3)
print("Sphere-reflect position:", state.pos)
print("Sphere-reflect velocity:", state.vel)


# %%
# ``apply`` vs ``shift``
# ~~~~~~~~~~~~~~~~~~~~~~~
# Domains expose two boundary-enforcement methods:
#
# * ``apply(state, system)`` — called automatically at every integration step.
#   For reflective domains this performs position correction and impulse
#   updates. For periodic domains it is a no-op.
# * ``shift(state, system)`` — an explicit call you make when you want
#   positions mapped back into the primary box. For periodic domains this
#   wraps coordinates; for free/reflective domains it is a no-op.
#
# In practice you rarely call ``apply`` yourself — the integrator does it.
# You typically call ``shift`` right before saving output or computing
# observables that need positions inside the box.
