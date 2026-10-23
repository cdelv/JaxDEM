r"""The Simulation Domain
----------------------------------------

The :py:class:`~jaxdem.domains.Domain` defines the spatial boundaries and
boundary conditions of a simulation. It controls two things:

1. **Displacement** — how the domain computes the relative displacement
   vector between two particles (important for periodic boundary
   conditions).
2. **Boundary enforcement** — how the domain constrains particles to
   stay inside the simulation box (reflection, wrapping, or nothing at
   all).

JaxDEM supports five domain types:

* ``"free"`` (:py:class:`~jaxdem.domains.free.FreeDomain`) — unbounded space, no boundary effects.
* ``"periodic"`` (:py:class:`~jaxdem.domains.periodic.PeriodicDomain`) — periodic (minimum-image) boundary conditions.
* ``"leesedwards"`` (:py:class:`~jaxdem.domains.lees_edwards.LeesEdwardsDomain`) — shear-periodic Lees--Edwards boundaries.
* ``"reflect"`` (:py:class:`~jaxdem.domains.reflect.ReflectDomain`) — reflective walls with impulse-based collision for general
  rigid bodies (spheres and clumps).
* ``"reflectsphere"`` (:py:class:`~jaxdem.domains.reflect_sphere.ReflectSphereDomain`) — a faster reflective domain for
  sphere-only simulations.

This guide covers each one.
"""

# %%
# Domain Creation
# ~~~~~~~~~~~~~~~~~
# You usually create a domain through
# :py:meth:`~jaxdem.system.System.create`. Pass ``domain_type`` and,
# optionally, ``domain_kw``:

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxdem as jdem

state = jdem.State.create(pos=jnp.zeros((1, 2)))

system = jdem.System.create(
    state.shape,
    domain_type="periodic",
    domain_kw={
        "box_size": 10.0 * jnp.ones(2),
        "anchor": jnp.zeros(2),
    },
)
print("Domain type:", type(system.domain).__name__)
print("Box size:", system.domain.box_size)
print("Anchor:", system.domain.anchor)

# %%
# You can also build a domain independently with
# :py:meth:`Domain.create <jaxdem.factory.Factory.create>` and assign it to
# :py:attr:`system.domain <jaxdem.system.System.domain>`:

domain = jdem.Domain.create("free", dim=2)
system.domain = domain
print("Swapped to:", type(system.domain).__name__)


# %%
# Common Attributes
# ~~~~~~~~~~~~~~~~~~
# Every domain has two core attributes:
#
# * :py:attr:`box_size <jaxdem.domains.Domain.box_size>` — the length of the simulation box along each axis,
#   shape
#   ``(dim,)``.
# * :py:attr:`anchor <jaxdem.domains.Domain.anchor>` — the minimum-corner coordinate of the box, shape
#   ``(dim,)``.
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
# Particles move freely in an unbounded space. The domain updates the
# :py:attr:`box_size <jaxdem.domains.Domain.box_size>` and :py:attr:`anchor <jaxdem.domains.Domain.anchor>`
# each step to tightly enclose all particles.
# Some internal algorithms, like spatial hashing in the cell-list colliders,
# need a finite bounding box.
state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [3.0, 4.0]]),
    rad=jnp.array([0.5, 0.5]),
)
system = jdem.System.create(state.shape, domain_type="free")

# After a step, the domain auto-fits to the particles:
state, system = jdem.System.initialize(state, system)
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
# The :py:attr:`periodic <jaxdem.domains.periodic.PeriodicDomain.periodic>` property returns ``True`` for this
# domain, which
# lets colliders and other components adapt their behavior automatically.
#
# Periodic boundary conditions do **not** modify positions during the time
# step (:py:meth:`~jaxdem.domains.Domain.apply` is a no-op). To wrap positions back into the primary box
# — for example before saving — use the :py:meth:`~jaxdem.domains.Domain.shift` method.

state = jdem.State.create(
    pos=jnp.array([[0.1, 0.1], [9.9, 10.2]]),
)
system = jdem.System.create(
    state.shape,
    domain_type="periodic",
    domain_kw={"box_size": 10.0 * jnp.ones(2), "anchor": jnp.zeros(2)},
)
print("Is periodic?", system.domain.periodic)

# %%
# The minimum-image displacement between two particles near opposite edges
# wraps around the box through :py:meth:`~jaxdem.domains.periodic.PeriodicDomain.displacement`:

rij = system.domain.displacement(state.pos[0], state.pos[1], system)
naive = state.pos[0] - state.pos[1]
print("Naive displacement:", naive)
print("Minimum-image displacement:", rij)
periodic_naive = np.asarray(naive)
periodic_displacement = np.asarray(rij)

# %%
# Call :py:meth:`shift <jaxdem.domains.periodic.PeriodicDomain.shift>` to wrap positions back into the primary
# image:

state, system = system.domain.shift(state, system)
print("Wrapped positions:\n", state.pos)


# %%
# Lees--Edwards Shear-Periodic Domain
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.domains.lees_edwards.LeesEdwardsDomain` is periodic in
# every direction. When an image crosses the shear-gradient axis
# :py:attr:`beta <jaxdem.domains.lees_edwards.LeesEdwardsDomain.beta>`, it
# is offset along the shear-flow axis :py:attr:`alpha <jaxdem.domains.lees_edwards.LeesEdwardsDomain.alpha>`
# by ``gamma * box_size[beta]``.
# :py:attr:`gamma <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma>` is the current shear strain. The
# defaults are flow along x
# (``alpha=0``) and gradient along y (``beta=1``); the axes must be distinct.

system = jdem.System.create(
    state.shape,
    domain_type="leesedwards",
    domain_kw={
        "box_size": 10.0 * jnp.ones(2),
        "anchor": jnp.zeros(2),
        "gamma": 0.5,
        "alpha": 0,
        "beta": 1,
    },
)
rij = system.domain.displacement(jnp.array([1.0, 0.1]), jnp.array([6.0, 9.9]), system)
print("Lees-Edwards displacement:", rij)
sheared_displacement = np.asarray(rij)
sheared_naive = np.array([1.0, 0.1]) - np.array([6.0, 9.9])

# %%
# The domain reads :py:attr:`gamma <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma>` (strain) and
# :py:attr:`gamma_dot <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma_dot>` (shear rate), but does
# not advance them. Update the strain in :py:attr:`~jaxdem.system.System.user_pre_step_actions` so forces
# use the strain at the new time. :py:class:`Cundall–Strack <jaxdem.forces.cundall_strack.CundallStrackForce>`
# uses :py:attr:`gamma_dot <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma_dot>` to account
# for the velocity of the selected periodic image. Ordinary stepping keeps
# particle coordinates unwrapped; no call to
# :py:meth:`domain.shift <jaxdem.domains.periodic.PeriodicDomain.shift>` is needed.
#
# .. code-block:: python
#
#    from dataclasses import replace
#
#    def advance_shear(state, system):
#        domain = replace(
#            system.domain,
#            gamma=system.domain.gamma + system.domain.gamma_dot * system.dt,
#        )
#        return state, replace(system, domain=domain)
#
#    system = jdem.System.create(
#        state.shape,
#        domain_type="leesedwards",
#        domain_kw={"box_size": 10.0 * jnp.ones(2), "gamma_dot": shear_rate},
#        user_pre_step_actions=advance_shear,
#    )
#    state, system = system.initialize(state, system)
#
# For a time-dependent rate, update
# :py:attr:`gamma_dot <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma_dot>` alongside
# :py:attr:`gamma <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma>` and set
# both to their initial values before :py:meth:`initialize <jaxdem.system.System.initialize>`. For example,
# oscillatory
# shear uses ``gamma = amplitude * jnp.sin(omega * system.time)`` and
# ``gamma_dot = amplitude * omega * jnp.cos(omega * system.time)``.
#
# For discrete affine increments (for example, quasistatic shear), call
# :py:meth:`~jaxdem.domains.lees_edwards.LeesEdwardsDomain.shear`:
# ``state, system = system.domain.shear(state, system, dgamma)``. This updates
# :py:attr:`gamma <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma>` and advances pair histories such as
# Cundall--Strack exactly once,
# with no time passing: velocities, ``dt``,
# :py:attr:`gamma_dot <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma_dot>` and dashpots play no
# part. Do not represent the same strain increment through both this call and
# the continuous protocol. Box rescaling works the same way through
# :py:meth:`~jaxdem.domains.Domain.resize`:
# ``system.domain.resize(state, system, new_box_size)``, which every domain
# supports and which :py:func:`~jaxdem.utils.packing_utils.scale_to_packing_fraction` uses.
#
# :py:meth:`shift <jaxdem.domains.periodic.PeriodicDomain.shift>` removes the positional shear offset for each
# crossed gradient image and then
# wraps all coordinates into the primary box. Spatial-search caches include
# :py:attr:`gamma <jaxdem.domains.lees_edwards.LeesEdwardsDomain.gamma>`,
# :py:attr:`alpha <jaxdem.domains.lees_edwards.LeesEdwardsDomain.alpha>`, and
# :py:attr:`beta <jaxdem.domains.lees_edwards.LeesEdwardsDomain.beta>` in their geometry snapshot, so cached
# colliders rebuild when that geometry changes.


# %%
# Reflective Domain
# ~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.domains.reflect.ReflectDomain` reflects particles off
# the walls of the simulation box. It uses impulse-based collision mechanics,
# correctly handling both linear and angular velocity for spheres and clumps.
#
# An optional :py:attr:`~jaxdem.domains.reflect.ReflectDomain.restitution_coefficient` (default ``1.0``, i.e.
# perfectly
# elastic) scales the normal rebound speed relative to the approach speed.
# For a head-on impact on a stationary wall, the retained kinetic-energy
# fraction is its square.

state = jdem.State.create(
    pos=jnp.array([[0.5, 0.5]]),
    vel=jnp.array([[-1.0, 0.0]]),
    rad=jnp.array([0.4]),
)
system = jdem.System.create(
    state.shape,
    domain_type="reflect",
    dt=0.01,
    domain_kw={
        "box_size": 10.0 * jnp.ones(2),
        "anchor": jnp.zeros(2),
        "restitution_coefficient": 1.0,
    },
)
print("Restitution:", system.domain.restitution_coefficient)
reflect_approach_velocity = float(state.vel[0, 0])

# The center starts 0.1 above the contact position x = radius. At speed 1,
# it reaches the wall after 0.1 time units. Twenty dt=0.01 steps cover the
# approach and rebound while keeping this demonstration short:
state, system = jdem.System.initialize(state, system)
state, system = system.step(state, system, n=20)
print("Position after bounce:", state.pos)
print("Velocity after bounce:", state.vel)
reflect_rebound_velocity = float(state.vel[0, 0])
np.testing.assert_allclose(state.vel, [[1.0, 0.0]], atol=1e-6)
assert float(state.pos[0, 0]) >= float(state.rad[0])


# %%
# Reflect-Sphere Domain
# ~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.domains.reflect_sphere.ReflectSphereDomain` is a
# lightweight variant of the reflective domain that skips the full
# impulse calculation. It mirrors positions and reverses the
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
    dt=0.01,
    domain_kw={"box_size": 10.0 * jnp.ones(2), "anchor": jnp.zeros(2)},
)

state, system = jdem.System.initialize(state, system)
reflect_sphere_approach_velocity = float(state.vel[0, 0])
state, system = system.step(state, system, n=20)
print("Sphere-reflect position:", state.pos)
print("Sphere-reflect velocity:", state.vel)
reflect_sphere_rebound_velocity = float(state.vel[0, 0])
np.testing.assert_allclose(state.vel, [[1.0, 0.0]], atol=1e-6)


# %%
# :py:meth:`~jaxdem.domains.Domain.apply` vs :py:meth:`~jaxdem.domains.Domain.shift`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Domains expose two boundary-enforcement methods:
#
# * :py:meth:`~jaxdem.domains.Domain.apply` — called automatically at every integration step.
#   For reflective domains this performs position correction and impulse
#   updates. For periodic domains it is a no-op.
# * :py:meth:`~jaxdem.domains.Domain.shift` — an explicit call you make when you want
#   positions mapped back into the primary box. For periodic domains this
#   wraps coordinates; for free/reflective domains it is a no-op.
#
# In practice you rarely call :py:meth:`~jaxdem.domains.Domain.apply` yourself — the integrator does it.
# You typically call :py:meth:`~jaxdem.domains.Domain.shift` right before saving output or computing
# observables that need positions inside the box.

# %%
# Compare displacement and rebound diagnostics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The first two panels compare vector components, rather than drawing a
# particle layout. They use the exact displacement queries above. Periodic
# wrapping and the shear offset change which image supplies the separation.
# The last panel reports measured velocities from the two existing head-on
# rebounds. With restitution one, the normal speed is retained and its sign
# reverses at the wall.
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
components = np.arange(2)
for ax, raw, minimum_image, title in (
    (axes[0], periodic_naive, periodic_displacement, "Periodic separation"),
    (axes[1], sheared_naive, sheared_displacement, "Lees–Edwards separation"),
):
    raw_bars = ax.bar(
        components - 0.18, raw, width=0.36, label="Direct subtraction", color="#7b8794"
    )
    image_bars = ax.bar(
        components + 0.18,
        minimum_image,
        width=0.36,
        label="Domain displacement",
        color="#258685",
    )
    ax.bar_label(raw_bars, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(image_bars, fmt="%.2f", padding=2, fontsize=8)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(
        title=title,
        xticks=components,
        xticklabels=["x component", "y component"],
        ylabel="Displacement component",
    )
    ax.legend(fontsize=8)
axes[2].bar(
    components - 0.18,
    [reflect_approach_velocity, reflect_sphere_approach_velocity],
    width=0.36,
    label="Approach",
    color="#7b8794",
)
axes[2].bar(
    components + 0.18,
    [reflect_rebound_velocity, reflect_sphere_rebound_velocity],
    width=0.36,
    label="Rebound",
    color="#d07a28",
)
axes[2].axhline(0, color="0.4", lw=0.8)
axes[2].set(
    title="Elastic head-on reflection",
    xticks=components,
    xticklabels=["Reflect", "Reflect-sphere"],
    ylabel="Normal velocity component",
)
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(axis="y", alpha=0.18)
plt.show()
