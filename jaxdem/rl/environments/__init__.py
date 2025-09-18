# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Reinforcement-learning environment interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Tuple, Type

import jax
from jax.typing import ArrayLike

from ...factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ...state import State
    from ...system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Environment(Factory, ABC):
    """Defines the high-level interface for RL environments."""

    state: "State"
    system: "System"
    env_params: Dict[str, Any]
    max_num_agents: int = field(default=0, metadata={"static": True})
    action_space_size: int = field(default=0, metadata={"static": True})
    action_space_shape: Tuple[int, ...] = field(default=(), metadata={"static": True})
    observation_space_size: int = field(default=0, metadata={"static": True})
    _base_env_cls: ClassVar[Type["Environment"]]

    @staticmethod
    @abstractmethod
    @jax.jit
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """Initialise the environment to a valid start state."""
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def reset_if_done(env: "Environment", done: jax.Array, key: ArrayLike) -> "Environment":
        base_cls = getattr(env.__class__, "_base_env_cls", env.__class__)
        return jax.lax.cond(
            done,
            lambda _: base_cls.reset(env, key),
            lambda _: env,
            operand=None,
        )

    @staticmethod
    @abstractmethod
    @jax.jit
    def step(env: "Environment", action: jax.Array) -> "Environment":
        """Advance the simulation by one time step using the given actions."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """Return the per-agent observation vector."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def reward(env: "Environment") -> jax.Array:
        """Return the per-agent immediate reward."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def done(env: "Environment") -> jax.Array:
        """Return whether the episode has terminated."""
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def info(env: "Environment") -> Dict[str, Any]:
        """Optional diagnostic information (empty by default)."""
        return dict()


from .multi_navigator import MultiNavigator  # noqa: E402,F401
from .single_navigator import SingleNavigator  # noqa: E402,F401

__all__ = ["Environment", "MultiNavigator", "SingleNavigator"]
