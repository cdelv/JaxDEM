# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Batch Sharding Across GPUs
-----------------------------

This guide creates periodic sphere systems,
then splits a batch across devices.
"""

# %%
# Create One Setup
# ~~~~~~~~~~~~~~~~~
# We first create a function to create the system we want to study:

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import jaxdem as jdem

box_length = (64 * (4 * jnp.pi / 3) / 0.5) ** (1 / 3)

def create_setup(key):
    state = jdem.utils.grid_state(n_per_axis=(4, 4, 4), spacing=3.0, key=key)
    state.pos_c %= box_length
    state.vel = 0.5 * jdem.utils.unit(state.vel)
    system = jdem.System.create(
        state=state,
        key=key,
        dt=1e-2,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(3, box_length)},
        collider_type="neighbor_list",
        collider_kw={
            "cutoff": 2.0,
            "skin": 0.2,
            "max_neighbors": 32,
            "secondary_collider_type": "cell_list",
            "secondary_collider_kw": {"cell_size": 2.2, "search_range": 1},
        },
    )
    return state, system


state, system = create_setup(jax.random.key(0))
print("One setup:", state.pos.shape)

# %%
# Create a Batch with vmap
# ~~~~~~~~~~~~~~~~~~~~~~~~
# ``vmap`` calls the setup function for each random key. All systems have the
# same array shapes and static configuration. Each system has its own
# velocities and dynamic data:

batch_size = 16
keys = jax.random.split(jax.random.key(42), batch_size)
batch_state, batch_system = jax.vmap(create_setup)(keys)
print("Batch positions:", batch_state.pos.shape)

# %%
# Shard the Batch Axis
# ~~~~~~~~~~~~~~~~~~~~~
# Now, to distribute the computation a mesh names the device axis. ``P("batch")`` splits the leading array axis
# across that mesh. On two GPUs, each GPU holds eight complete simulations.
# Each simulation must fit on one GPU. The batch size must be divisible by
# the device count.

mesh = Mesh(jax.local_devices(), ("batch",))
sharding = NamedSharding(mesh, P("batch"))
batch_state, batch_system = jax.device_put((batch_state, batch_system), sharding)

for shard in batch_state.pos_c.addressable_shards:
    print(f"{shard.device}: position shape {shard.data.shape}")

# %%
# Create the Batch Directly on the Devices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For a larger setup, give ``jit`` the output sharding. This replaces batch
# creation followed by transfer. Only the small array of keys needs placement
# before the call. JAX creates the output arrays with the requested sharding.

create_batch = jax.jit(jax.vmap(create_setup), out_shardings=sharding)
batch_state, batch_system = create_batch(jax.device_put(keys, sharding))
jax.block_until_ready((batch_state, batch_system))

# %%
# Take One Step
# ~~~~~~~~~~~~~~
# The existing System methods handle the batch axis and keep positions and velocities on their shards:

batch_state, batch_system = jdem.System.initialize(batch_state, batch_system)
batch_state, batch_system = jdem.System.step(batch_state, batch_system, n=1)
jax.block_until_ready((batch_state, batch_system))
print("Step counts:", batch_system.step_count)
