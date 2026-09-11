"""Built-in frictional environments initialize persistent contact history."""

import jax
import jax.numpy as jnp

from jaxdem.colliders import NeighborList
from jaxdem.rl.environments.swarm_roller_3d import SwarmRoller3D
from jaxdem.rl.environments.three_gears import ThreeGears


def test_jitted_rl_resets_construct_static_capacity_neighbor_lists() -> None:
    cases = (
        ThreeGears.Create(num_gears=1, box_size=12.0, max_steps=2),
        SwarmRoller3D.Create(
            N=2,
            num_objectives=2,
            box_size=4.0,
            box_padding=2.0,
            max_steps=2,
            n_lidar_rays=4,
        ),
    )
    for env in cases:
        reset = env.reset(env, jax.random.key(0))
        assert isinstance(reset.system.collider, NeighborList)
        assert reset.system.collider.max_neighbors <= reset.state.N
        assert reset.system.collider.history.shape[-1] > 0
        assert not bool(reset.system.collider.overflow)
        assert not bool(reset.system.search_overflow)
        assert bool(jnp.all(jnp.isfinite(reset.state.force)))
