# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from typing import Any, Callable

import jax
import jax.numpy as jnp

import jaxdem as jdem

SYSTEM_BOX_SIZE = 200.0


class SkipBenchmark(Exception):
    """Exception raised to skip a benchmark."""

    pass


def create_spheres_state(N: int = 1_000_000, dim: int = 3, **kwargs: Any) -> jdem.State:
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


def create_clumps_state(N: int = 1_000_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    key_center, key_offset, key_scale, key_rad = jax.random.split(
        jax.random.PRNGKey(1), 4
    )
    clump_pattern = jnp.array([3, 4, 5, 6], dtype=int)
    pattern_total = int(jnp.sum(clump_pattern))
    pattern_repeats = max(1, (N + pattern_total - 1) // pattern_total)
    clump_sizes = jnp.tile(clump_pattern, pattern_repeats)
    cumulative_sizes = jnp.cumsum(clump_sizes)
    n_clumps = int(jnp.searchsorted(cumulative_sizes, N, side="left")) + 1
    clump_sizes = clump_sizes[:n_clumps]
    overflow = int(jnp.sum(clump_sizes)) - N
    if overflow > 0:
        clump_sizes = clump_sizes.at[-1].add(-overflow)

    clump_id = jnp.repeat(jnp.arange(n_clumps, dtype=int), clump_sizes)[:N]
    centers = jax.random.uniform(
        key_center, (n_clumps, dim), minval=0.0, maxval=SYSTEM_BOX_SIZE
    )
    pos_c = centers[clump_id]

    raw_offset = jax.random.normal(key_offset, (N, dim))
    unit_offset = raw_offset / (
        jnp.linalg.norm(raw_offset, axis=1, keepdims=True) + 1e-8
    )
    offset_scale = jax.random.uniform(key_scale, (N, 1), minval=0.05, maxval=0.45)
    pos_p = unit_offset * offset_scale

    rad = jax.random.uniform(key_rad, (N,), minval=0.06, maxval=0.14)
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


def create_deformable_state(
    N: int = 1_000_000, dim: int = 3, **kwargs: Any
) -> jdem.State:
    key_center, key_offset, key_scale, key_rad = jax.random.split(
        jax.random.PRNGKey(2), 4
    )
    nodes_per_particle = 6
    n_particles = max(1, (N + nodes_per_particle - 1) // nodes_per_particle)

    bond_id = jnp.arange(N, dtype=int) // nodes_per_particle
    centers = jax.random.uniform(
        key_center, (n_particles, dim), minval=0.0, maxval=SYSTEM_BOX_SIZE
    )
    raw_offset = jax.random.normal(key_offset, (N, dim))
    unit_offset = raw_offset / (
        jnp.linalg.norm(raw_offset, axis=1, keepdims=True) + 1e-8
    )
    offset_scale = jax.random.uniform(key_scale, (N, 1), minval=0.1, maxval=0.5)
    pos_c = centers[bond_id] + unit_offset * offset_scale

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


def create_mixed_state(N: int = 1_000_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    if N < 3:
        raise SkipBenchmark("Mixed benchmark requires at least 3 particles.")

    n_spheres = N // 3
    n_clumps = N // 3
    n_deformable = N - n_spheres - n_clumps

    spheres = create_spheres_state(N=n_spheres, dim=dim)
    clumps = create_clumps_state(N=n_clumps, dim=dim)
    deformable = create_deformable_state(N=n_deformable, dim=dim)
    mixed_state = jdem.State.merge(spheres, [clumps, deformable])

    state_kwargs = {
        "pos": mixed_state.pos_c,
        "pos_p": mixed_state.pos_p,
        "rad": mixed_state.rad,
        "clump_id": mixed_state.clump_id,
        "bond_id": mixed_state.bond_id,
    }
    state_kwargs.update(kwargs)
    return jdem.State.create(**state_kwargs)


def get_state_factory(system_type: str) -> Callable[..., jdem.State]:
    factories: dict[str, Callable[..., jdem.State]] = {
        "spheres": create_spheres_state,
        "clumps": create_clumps_state,
        "deformable": create_deformable_state,
        "mixed": create_mixed_state,
    }
    return factories[system_type]
