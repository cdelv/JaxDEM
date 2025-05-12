# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from typing import Dict
from dataclasses import dataclass, field, fields
from functools import partial

import jaxdem as jdem
from .Env import Env, EnvState

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
@Env.register("single_navigator")
class SingleNavigator(Env):
    """
    TO DO: Explore broadcasting the env parameters

    SingleNavigator: a minimal single‐agent environment in a reflective box.

    On reset, the agent’s position and velocity, as well as the goal target,
    are sampled uniformly within the interior of a box of size `L`.

    **Action space**
      - Continuous acceleration vector of shape `(dim,)`.

    **Observation space**
      - Vector of length `dim * 5`, comprised of:
        1. Agent position `pos` (flattened)
        2. Agent velocity `vel` (flattened)
        3. Box dimensions `box_size` (flattened)
        4. Domain anchor `anchor` (flattened)
        5. Goal target position `target` (flattened)

    **Reward**
      - Negative Euclidean distance between the agent and the target: r = -|pos - target|.

    **Termination**
      - Episodes never terminate early.

    **Parameters**
      - `dim` (int): Spatial dimension (2 or 3).
      - `L` (float): Side length of the cubic box.
      - `rad` (float): Agent radius.
      - `max_vel` (float): Maximum initial velocity of the agent.
    """
    dim: int = field(default=2, metadata={"static": True})
    L: float = field(default=20.0, metadata={"static": True})
    rad: float = field(default=1.0, metadata={"static": True})
    max_vel: float = field(default=1.0, metadata={"static": True})
    damping: float = field(default=0.15, metadata={"static": True})

    action_space: int = field(default=2, metadata={"static": True})
    observation_space: int = field(default=10, metadata={"static": True})

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, getattr(self, f.name))

        object.__setattr__(self, "action_space", self.dim)
        object.__setattr__(self, "observation_space", self.dim * 5)

    @staticmethod
    @jax.jit
    def reset(key: jnp.ndarray, env_params: Dict = {}) -> "EnvState":
        N   = 1
        dim = SingleNavigator.dim
        L   = SingleNavigator.L
        rad = SingleNavigator.rad
        max_vel = SingleNavigator.max_vel

        key, subkey = jax.random.split(key)
        pos = jax.random.uniform(subkey,  shape=(N, dim), minval=rad, maxval=L-rad)
        key, subkey = jax.random.split(key)
        vel = jax.random.uniform(subkey,  shape=(N, dim), minval=-max_vel, maxval=max_vel)
        key, subkey = jax.random.split(key)
        target = jax.random.uniform(subkey, shape=(N, dim), minval=rad, maxval=L-rad)

        state = jdem.State.create(dim=dim, pos=pos, vel=vel)
        system = jdem.System(
            dt         = 0.01,
            domain     = jdem.Domain.create('reflect', dim=dim, box_size=L*jnp.ones(dim)),
            simulator  = jdem.Simulator.create('naive'),
            integrator = jdem.Integrator.create('euler'),
            force_model= jdem.ForceModel.create('spring'),
        )

        return EnvState(state, system, target)

    @staticmethod
    @jax.jit
    def step(env_state: "EnvState", action: jnp.ndarray) -> "EnvState":
        state, system, env_params = env_state

        state, system = system.simulator.compute_force(state, system)
        state.accel += action - SingleNavigator.damping * state.vel

        state, system = system.integrator.step(state, system)
        state = system.domain.shift(state, system)

        return EnvState(state, system, env_params)

    @staticmethod
    @jax.jit
    def observation(env_state: "EnvState") -> jnp.ndarray:
        state, system, target = env_state
        p = state.pos.flatten()
        v = state.vel.flatten()
        b = system.domain.box_size.flatten()
        a = system.domain.anchor.flatten()
        return jnp.concatenate([p, v, b, a, target.flatten()])

    @staticmethod
    @jax.jit
    def reward(env_state: "EnvState") -> jnp.ndarray:
        state, system, target = env_state
        distance = jnp.linalg.norm(system.domain.displacement(state.pos, target, system))

        return SingleNavigator.L/2 - distance