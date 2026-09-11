# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jdem

SYSTEM_BOX_SIZE = 200.0


class SkipBenchmark(Exception):
    """Exception that signals the benchmark runner to skip a benchmark."""

    pass


def create_spheres_state(N: int = 500_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    key_pos, key_rad = jax.random.split(jax.random.PRNGKey(0))
    pos_c = jax.random.uniform(key_pos, (N, dim), minval=0.0, maxval=SYSTEM_BOX_SIZE)
    pos_p = jnp.zeros_like(pos_c)
    rad = jax.random.uniform(key_rad, (N,), minval=0.08, maxval=0.12)
    clump_id = jnp.arange(N, dtype=int)
    bond_id = jnp.arange(N, dtype=int)
    state_kwargs = {
        "pos": pos_c,
        "pos_p": pos_p,
        "rad": rad,
        "clump_id": clump_id,
        "bond_id": bond_id,
    }
    state_kwargs.update(kwargs)
    return jdem.State.create(**state_kwargs)


def create_clumps_state(N: int = 500_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    key_center = jax.random.PRNGKey(1)
    clump_pattern = np.array([3, 4, 5, 6], dtype=np.int32)
    pattern_total = int(np.sum(clump_pattern))
    pattern_repeats = max(1, (N + pattern_total - 1) // pattern_total)
    clump_sizes = np.tile(clump_pattern, pattern_repeats)
    cumulative_sizes = np.cumsum(clump_sizes)
    n_clumps = int(np.searchsorted(cumulative_sizes, N, side="left")) + 1
    clump_sizes = clump_sizes[:n_clumps]
    overflow = int(np.sum(clump_sizes)) - N
    if overflow > 0:
        clump_sizes[-1] -= overflow

    clump_id = jnp.repeat(jnp.arange(n_clumps, dtype=int), jnp.asarray(clump_sizes))
    centers = jax.random.uniform(
        key_center, (n_clumps, dim), minval=0.0, maxval=SYSTEM_BOX_SIZE
    )
    pos_c = centers[clump_id]

    # A deterministic regular polygon around each COM gives centered,
    # non-overlapping members and reproducible principal axes.
    member_index = jnp.arange(N) - jnp.repeat(
        jnp.asarray(np.r_[0, cumulative_sizes[: n_clumps - 1]], dtype=int),
        jnp.asarray(clump_sizes),
    )
    member_count = jnp.asarray(clump_sizes)[clump_id]
    theta = 2.0 * jnp.pi * member_index / member_count
    offset_radius = jnp.where(member_count == 1, 0.0, 0.3)
    pos_p = jnp.zeros((N, dim), dtype=float)
    pos_p = pos_p.at[:, 0].set(offset_radius * jnp.cos(theta))
    if dim > 1:
        pos_p = pos_p.at[:, 1].set(offset_radius * jnp.sin(theta))

    rad = jnp.full((N,), 0.1)
    mass = jnp.ones((N,))
    member_mass = 1.0 / member_count
    ball_volume = jnp.exp(
        0.5 * dim * jnp.log(jnp.pi)
        + dim * jnp.log(0.1)
        - jax.scipy.special.gammaln(0.5 * dim + 1.0)
    )
    volume = member_count * ball_volume
    if dim == 2:
        inertia_body = jnp.zeros((N, 1))
        inertia_body = inertia_body.at[:, 0].set(offset_radius**2 + 0.5 * 0.1**2)
    else:
        intrinsic = 0.4 * 0.1**2
        inertia_body = jnp.stack(
            [
                jax.ops.segment_sum(
                    member_mass
                    * (jnp.sum(pos_p**2, axis=-1) - pos_p[:, axis] ** 2 + intrinsic),
                    clump_id,
                    n_clumps,
                )[clump_id]
                for axis in range(3)
            ],
            axis=-1,
        )
    bond_id = jnp.arange(N, dtype=int)
    state_kwargs = {
        "pos": pos_c,
        "pos_p": pos_p,
        "rad": rad,
        "volume": volume,
        "mass": mass,
        "inertia": inertia_body,
        "clump_id": clump_id,
        "bond_id": bond_id,
    }
    state_kwargs.update(kwargs)
    return jdem.State.create(**state_kwargs)


def create_deformable_state(
    N: int = 500_000, dim: int = 3, **kwargs: Any
) -> jdem.State:
    key_center, key_offset, key_scale, key_rad = jax.random.split(
        jax.random.PRNGKey(2), 4
    )
    nodes_per_particle = 6
    n_particles = max(1, (N + nodes_per_particle - 1) // nodes_per_particle)

    # State.bond_id stores adjacency indices, not per-body labels. Use a ring
    # within every deformable particle, including a shorter final particle.
    bond_id_np = np.full((N, 2), -1, dtype=np.int32)
    for start in range(0, N, nodes_per_particle):
        stop = min(start + nodes_per_particle, N)
        size = stop - start
        if size == 2:
            bond_id_np[start, 0] = start + 1
            bond_id_np[start + 1, 0] = start
        elif size > 2:
            ids = np.arange(start, stop)
            bond_id_np[ids, 0] = np.roll(ids, 1)
            bond_id_np[ids, 1] = np.roll(ids, -1)
    bond_id = jnp.asarray(bond_id_np)
    centers = jax.random.uniform(
        key_center, (n_particles, dim), minval=0.0, maxval=SYSTEM_BOX_SIZE
    )
    raw_offset = jax.random.normal(key_offset, (N, dim))
    unit_offset = raw_offset / (
        jnp.linalg.norm(raw_offset, axis=1, keepdims=True) + 1e-8
    )
    offset_scale = jax.random.uniform(key_scale, (N, 1), minval=0.1, maxval=0.5)
    particle_id = jnp.arange(N, dtype=int) // nodes_per_particle
    pos_c = centers[particle_id] + unit_offset * offset_scale

    pos_p = jnp.zeros_like(pos_c)
    rad = jax.random.uniform(key_rad, (N,), minval=0.06, maxval=0.11)
    clump_id = jnp.arange(N, dtype=int)
    state_kwargs = {
        "pos": pos_c,
        "pos_p": pos_p,
        "rad": rad,
        "clump_id": clump_id,
        "bond_id": bond_id,
    }
    state_kwargs.update(kwargs)
    return jdem.State.create(**state_kwargs)


def create_mixed_state(N: int = 500_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    if N < 3:
        raise SkipBenchmark("Mixed benchmark requires at least 3 particles.")

    n_spheres = N // 3
    n_clumps = N // 3
    n_deformable = N - n_spheres - n_clumps

    spheres = create_spheres_state(N=n_spheres, dim=dim)
    clumps = create_clumps_state(N=n_clumps, dim=dim)
    deformable = create_deformable_state(N=n_deformable, dim=dim)
    mixed_state = jdem.State.merge(spheres, [clumps, deformable])

    if not kwargs:
        return mixed_state
    # Keep all fields produced by State.merge. Benchmark-specific overrides
    # should not rebuild a partial State and discard body properties.
    import dataclasses

    if "pos" in kwargs:
        kwargs["pos_c"] = kwargs.pop("pos")
    return dataclasses.replace(mixed_state, **kwargs)


def get_state_factory(system_type: str) -> Callable[..., jdem.State]:
    factories: dict[str, Callable[..., jdem.State]] = {
        "spheres": create_spheres_state,
        "clumps": create_clumps_state,
        "deformable": create_deformable_state,
        "mixed": create_mixed_state,
    }
    return factories[system_type]
