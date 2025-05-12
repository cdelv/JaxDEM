# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, Any
from functools import partial
from dataclasses import dataclass, field, MISSING

from ..Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..State import State
    from ..System import System

class EnvState(NamedTuple):
    """
    Holds the full state of an environment at a given timestep.

    Attributes:
        state: jdem.State
        system: jdem.System
        env_params: jnp.ndarray
            A JAX array or Dict of any environment specific information.
        rng: jnp.ndarray
            The JAX PRNGKey.
    """
    state: "State"
    system: "System"
    env_params: jnp.ndarray | Dict[str, Any]
    #env:   "Env"

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class Env(Factory, ABC):
    """
    Abstract base class for JaxDEM reinforcement‐learning environments.

    Subclasses **must** override:
      - `reset`
      - `observation`
      - `reward`

    Provides:
      - A default `step` that applies actions as accelerations,
      - No‐op `done` and `info` methods which you may override.

    Attributes
    ----------
    action_space : int
        The dimension of the action vector.  All policies should output
        arrays of shape `(action_space,)`.
    observation_space : int
        The dimension of the observation vector.  The `observation` method
        must return arrays of shape `(observation_space,)`.
    """
    action_space: int = field(default=MISSING, metadata={"static": True})
    observation_space: int = field(default=MISSING, metadata={"static": True})

    @staticmethod
    @abstractmethod
    @jax.jit
    def reset(env: "Env", rng: jnp.ndarray, env_params: Dict = {}) -> "EnvState":
        """
        Initialize a new episode by sampling initial conditions and building
        the physics system.

        Args:
            rng: jnp.ndarray
                A JAX PRNGKey.
            env_params: Dict, optional
                Dictionary of environment configuration options.

        Returns:
            EnvState
                A namedtuple containing:
                  - state: jdem.State
                  - system: jdem.System
                  - env_params: jnp.ndarray
                      A JAX array or Dict of any environment specific information.
                  - rng: jnp.ndarray
                      The JAX PRNGKey.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def step(env_state: "EnvState", action: jnp.ndarray) -> "EnvState":
        """
        Advance the simulation one timestep under the given action.

        Args:
            env_state: EnvState
                The current environment state.
            action: jnp.ndarray
                External control input.

        Returns:
            EnvState
                A new EnvState reflecting the post‐step state and PRNGKey.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def observation(env_state: "EnvState") -> jnp.ndarray:
        """
        returns a flat observation vector from the current state.

        Args:
            env_state: EnvState

        Returns:
            jnp.ndarray
                A 1D array of features suitable as input to a policy/value network.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def reward(env_state: "EnvState") -> jnp.ndarray:
        """
        Compute the scalar reward for the current state.

        Args:
            env_state: EnvState

        Returns:
            jnp.ndarray
                A scalar (shape `()`) representing the immediate reward.
        """
        raise NotImplementedError

    @staticmethod
    @partial(jax.jit, inline=True)
    def done(env_state: "EnvState") -> bool:
        """
        Check whether the current state is terminal.

        By default, always returns False.  Subclasses may override to
        signal episode termination (e.g. out‐of‐bounds, goal reached).
        """
        return False

    @staticmethod
    @partial(jax.jit, inline=True)
    def info(env_state: "EnvState") -> Dict:
        """
        Return auxiliary diagnostic information.

        By default, returns an empty dict.  Subclasses may override to
        provide environment specific information.
        """
        return {}