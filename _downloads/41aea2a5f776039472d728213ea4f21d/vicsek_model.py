# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Vicsek alignment of rigid dumbbells
=======================================

Build custom active-matter integrators by subclassing
:class:`~jaxdem.integrators.LinearIntegrator` and running 16 two-sphere rigid
dumbbells in a periodic 2D box.

Each step queries neighbors via
:meth:`~jaxdem.colliders.Collider.create_neighbor_list`, averages neighbor
velocities and harmonic contact forces per clump via ``jax.ops.segment_sum``,
and prescribes constant speed ``v0`` under two classic Vicsek noise rules:

* **Intrinsic (angular) noise:** rotates the normalized heading by
  :math:`\xi_I \sim \mathcal{U}[-\pi\eta, \pi\eta]`.
* **Extrinsic (vectorial) noise:** adds :math:`\eta\hat{\boldsymbol{\xi}}_I`
  before normalizing.

Notice ``max_neighbors: int = jax.tree.static()`` on the integrator dataclasses:
buffer sizes must be static compile-time metadata in JAX pytrees.
"""

from dataclasses import dataclass
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem
from jaxdem.integrators import free_mask
from jaxdem.utils.linalg import unit

# %%
# 1) Custom Vicsek Integrators (Intrinsic vs Extrinsic Noise)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _vicsek_alignment(
    state: jdem.State,
    system: jdem.System,
    neighbor_radius: jax.Array,
    max_neighbors: int,
) -> tuple[jdem.State, jdem.System, jax.Array, jax.Array]:
    """Query neighbors and compute clump-averaged alignment vector and force."""
    state, system, nl, overflow = system.collider.create_neighbor_list(
        state, system, neighbor_radius, max_neighbors
    )
    system.search_overflow |= overflow

    safe_nl = jnp.maximum(nl, 0)
    valid = nl != -1
    sum_v = jnp.sum(state.vel[safe_nl] * valid[..., None], axis=1) + state.vel
    avg_v = sum_v / (jnp.sum(valid, axis=1) + 1)[..., None]

    # Reduce per-sphere vectors to one shared vector per clump.
    counts = jnp.bincount(state.clump_id, length=state.N)
    counts_safe = jnp.where(counts > 0, counts, 1).astype(avg_v.dtype)
    clump_vals = (
        jax.ops.segment_sum(
            jnp.concatenate((avg_v, state.force), axis=-1),
            state.clump_id,
            num_segments=state.N,
        )
        / counts_safe[..., None]
    )[state.clump_id]
    return state, system, clump_vals[..., state.dim :], clump_vals[..., : state.dim]


@jax.jit(inline=True)
def _apply_velocity(
    state: jdem.State, system: jdem.System, v_des: jax.Array
) -> tuple[jdem.State, jdem.System]:
    state.vel = jnp.where(free_mask(state), v_des, state.vel)
    state.pos_c = state.pos_c + system.dt * state.vel
    return state, system


@jdem.LinearIntegrator.register("vicsek_intrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekIntrinsic(jdem.LinearIntegrator):
    """2D Vicsek integrator with intrinsic (angular) noise per clump."""

    neighbor_radius: jax.Array
    eta: jax.Array
    v0: jax.Array
    max_neighbors: int = jax.tree.static()

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="VicsekIntrinsic.step_after_force")
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> tuple[jdem.State, jdem.System]:
        cfg = system.linear_integrator
        state, system, force, avg_v = _vicsek_alignment(
            state, system, cfg.neighbor_radius, cfg.max_neighbors
        )
        base = force + avg_v
        base_dir = jnp.where(
            jnp.linalg.norm(base, axis=-1, keepdims=True) > 0.0,
            unit(base),
            unit(state.vel),
        )
        system.key, noise_key = jax.random.split(system.key)
        dtheta = (
            (2.0 * jax.random.uniform(noise_key, shape=(state.N,)) - 1.0)
            * (jnp.pi * cfg.eta)
        )[state.clump_id]
        c, s = jnp.cos(dtheta), jnp.sin(dtheta)
        heading = jnp.stack(
            (c * base_dir[:, 0] - s * base_dir[:, 1], s * base_dir[:, 0] + c * base_dir[:, 1]),
            axis=-1,
        )
        return _apply_velocity(state, system, cfg.v0 * heading)


@jdem.LinearIntegrator.register("vicsek_extrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekExtrinsic(jdem.LinearIntegrator):
    """2D Vicsek integrator with extrinsic (vectorial) noise per clump."""

    neighbor_radius: jax.Array
    eta: jax.Array
    v0: jax.Array
    max_neighbors: int = jax.tree.static()

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="VicsekExtrinsic.step_after_force")
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> tuple[jdem.State, jdem.System]:
        cfg = system.linear_integrator
        state, system, force, avg_v = _vicsek_alignment(
            state, system, cfg.neighbor_radius, cfg.max_neighbors
        )
        system.key, noise_key = jax.random.split(system.key)
        angles = jax.random.uniform(noise_key, shape=(state.N,), minval=0.0, maxval=2.0 * jnp.pi)
        noise_dir = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)[state.clump_id]
        return _apply_velocity(state, system, cfg.v0 * unit(force + avg_v + cfg.eta * noise_dir))


# %%
# 2) Build 16 Rigid Dumbbells on a Periodic Grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# See :doc:`particle_construction` for general high-level packing builders.
# Because Vicsek integrators prescribe velocities directly (ignoring mass and
# inertia), we skip ``compute_clump_properties`` and disable rigid rotation:


def build_dumbbells(
    n_clumps: int,
    phi: float,
    dt: float,
    neighbor_radius: float,
    eta: float,
    v0: float,
    integrator_type: str,
    seed: int = 0,
) -> tuple[jdem.State, jdem.System]:
    n_spheres = 2 * n_clumps
    n_axis = int(np.sqrt(n_clumps))
    side = jnp.sqrt(n_clumps * jnp.pi * 1.5**2 / phi)
    axis = (jnp.arange(n_axis) + 0.5) * side / n_axis
    x, y = jnp.meshgrid(axis, axis, indexing="ij")
    centers = jnp.stack((x.ravel(), y.ravel()), axis=-1)

    angles = jax.random.uniform(jax.random.key(seed), (n_clumps,), minval=0.0, maxval=2.0 * jnp.pi)
    headings = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)

    state = jdem.State.create(
        pos=jnp.repeat(centers, 2, axis=0),
        pos_p=jnp.tile(jnp.array([[-0.5, 0.0], [0.5, 0.0]]), (n_clumps, 1)),
        vel=v0 * jnp.repeat(headings, 2, axis=0),
        rad=jnp.ones(n_spheres),
        mass=jnp.ones(n_spheres),
        clump_id=jnp.repeat(jnp.arange(n_clumps), 2),
    )
    mat_table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)],
        matcher=jdem.MaterialMatchmaker.create("harmonic"),
    )
    system = jdem.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type=integrator_type,
        linear_integrator_kw={
            "neighbor_radius": jnp.asarray(neighbor_radius, dtype=float),
            "eta": jnp.asarray(eta, dtype=float),
            "v0": jnp.asarray(v0, dtype=float),
            "max_neighbors": n_spheres - 1,
        },
        rotation_integrator_type=None,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, side)},
        force_model_type="spring",
        mat_table=mat_table,
        collider_type="naive",
        seed=seed,
    )
    return jdem.System.initialize(state, system)


# %%
# 3) Run Both Noise Variants and Plot Collective Polarization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Polarization :math:`P = \frac{1}{M v_0}\left|\sum_I \mathbf{v}_I\right|`
# rises from disordered random headings toward 1 as clumps align:

strides = jnp.asarray([0] + [1] * 47)
results = {}
for variant in ("vicsek_intrinsic", "vicsek_extrinsic"):
    st0, sys0 = build_dumbbells(
        n_clumps=16, phi=0.65, dt=0.1, neighbor_radius=3.5, eta=0.2, v0=1.0, integrator_type=variant
    )
    _, sys_f, (traj_s, traj_sys) = jdem.System.trajectory_rollout(st0, sys0, strides=strides)
    sys_f.check_overflow()
    pol = np.asarray(jnp.linalg.norm(jnp.mean(traj_s.vel, axis=-2), axis=-1))
    results[variant] = (np.asarray(traj_sys.time), pol, traj_s, traj_sys)
    print(f"{variant:16s}: polarization {pol[0]:.3f} -> {pol[-1]:.3f}")

scene_states, scene_systems = results["vicsek_intrinsic"][2], results["vicsek_intrinsic"][3]

fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
for variant, label in (
    ("vicsek_intrinsic", "Intrinsic (angular) noise"),
    ("vicsek_extrinsic", "Extrinsic (vectorial) noise"),
):
    t, pol, _, _ = results[variant]
    ax.plot(t, pol, label=label)
ax.set(
    xlabel="Time",
    ylabel="Polarization P",
    ylim=(0.0, 1.05),
    title="Collective alignment of 16 rigid dumbbells (eta = 0.2)",
)
ax.legend(fontsize="small")
ax.grid(alpha=0.2)
plt.show()
