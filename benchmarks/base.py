# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from typing import Any, Callable

import jax
import jax.numpy as jnp

import jaxdem as jdem


class SkipBenchmark(Exception):
    """Exception raised to skip a benchmark."""

    pass


def create_spheres_state(N: int = 100_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    key = jax.random.PRNGKey(0)
    pos = jax.random.uniform(key, (N, dim)) * 10.0
    rad = jnp.ones(N) * 0.1

    return jdem.State.create(pos=pos, rad=rad, **kwargs)


def create_clumps_state(N: int = 100_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    clump_id = jnp.repeat(jnp.arange(N // 10), 10)
    return create_spheres_state(N=N, dim=dim, clump_id=clump_id, **kwargs)


def create_deformable_state(
    N: int = 100_000, dim: int = 3, **kwargs: Any
) -> jdem.State:
    bond_id = jnp.repeat(jnp.arange(N // 2), 2)
    return create_spheres_state(N=N, dim=dim, bond_id=bond_id, **kwargs)


def create_mixed_state(N: int = 100_000, dim: int = 3, **kwargs: Any) -> jdem.State:
    clump_id = jnp.concatenate(
        [jnp.arange(N // 2), jnp.repeat(jnp.arange(N // 20), 10) + N // 2]
    )
    return create_spheres_state(N=N, dim=dim, clump_id=clump_id, **kwargs)


def get_state_factory(system_type: str) -> Callable[..., jdem.State]:
    factories: dict[str, Callable[..., jdem.State]] = {
        "spheres": create_spheres_state,
        "clumps": create_clumps_state,
        "deformable": create_deformable_state,
        "mixed": create_mixed_state,
    }
    return factories[system_type]
