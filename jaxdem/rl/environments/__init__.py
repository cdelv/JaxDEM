# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Reinforcement-learning environment interface."""

from __future__ import annotations

import jax
from jax.typing import ArrayLike

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Tuple, Type

from ...factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ...state import State
    from ...system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class Environment(Factory, ABC):
    """
    Defines the interface for reinforcement-learning environments.

    - Let **A** be the number of agents (A ≥ 1). Single-agent environments still use A=1.
    - Observations and actions are flattened per agent to fixed sizes. Use ``action_space_shape``
      to reshape inside the environment if needed.

    **Required shapes**

    - Observation: ``(A, observation_space_size)``
    - Action (input to :meth:`step`): ``(A, action_space_size)``
    - Reward: ``(A,)``
    - Done: scalar boolean for the whole environment

    TODO:

    - Truncated data field: per-agent termination flag
    - Render method

    Example
    -------
    To define a custom environment, inherit from :class:`Environment` and implement the abstract methods:

    >>> @Environment.register("MyCustomEnv")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True, frozen=True)
    >>> class MyCustomEnv(Environment):
        ...
    """

    state: "State"
    """
    Simulation state.
    """

    system: "System"
    """
    Simulation system configuration.
    """

    env_params: Dict[str, Any]
    """
    Environment-specific parameters.
    """

    max_num_agents: int = field(default=0, metadata={"static": True})
    """
    Maximum number of active agents in the environment.
    """

    action_space_size: int = field(default=0, metadata={"static": True})
    """
    Flattened action size per agent. Actions passed to :meth:`step` have shape ``(A, action_space_size)``.
    """

    action_space_shape: Tuple[int, ...] = field(default=(), metadata={"static": True})
    """
    Original per-agent action shape (useful for reshaping inside the environment).
    """

    observation_space_size: int = field(default=0, metadata={"static": True})
    """
    Flattened observation size per agent. :meth:`observation` returns shape ``(A, observation_space_size)``.
    """

    _base_env_cls: ClassVar[Type["Environment"]]

    @staticmethod
    @abstractmethod
    @jax.jit
    def reset(env: "Environment", key: ArrayLike) -> "Environment":
        """
        Initialize the environment to a valid start state.

        Parameters
        ----------
        env: Environment
            Instance of the environment.

        key : jax.random.PRNGKey
            JAX random number generator key.

        Returns
        -------
        Environment
            Freshly initialized environment.
        """
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def reset_if_done(
        env: "Environment", done: jax.Array, key: ArrayLike
    ) -> "Environment":
        """
        Conditionally resets the environment if the environment has reached a terminal state.

        This method checks the `done` flag and, if `True`, calls the environment's
        `reset` method to reinitialize the state. Otherwise, it returns the current
        environment unchanged.

        Parameters
        ----------
        env : Environment
            The current environment instance.

        done : jax.Array
            A boolean flag indicating whether the environment has reached a terminal state.

        key : jax.random.PRNGKey
            JAX random number generator key used for reinitialization.

        Returns
        -------
        Environment
            Either the freshly reset environment (if `done` is True) or the unchanged
            environment (if `done` is False).
        """
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
        """
        Advance the simulation by one step using **per-agent** actions.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : jax.Array
            The vector of actions each agent in the environment should take.

        Returns
        -------
        Environment
            The updated environment state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """
        Returns the per-agent observation vector.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to the environment observation.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def reward(env: "Environment") -> jax.Array:
        """
        Returns the per-agent immediate rewards.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Vector corresponding to all the agent's rewards based on the current environment state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def done(env: "Environment") -> jax.Array:
        """
        Returns a boolean indicating whether the environment has ended.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool indicating when the environment ended
        """
        raise NotImplementedError

    @staticmethod
    @jax.jit
    def info(env: "Environment") -> Dict[str, Any]:
        """
        Return auxiliary diagnostic information.

        By default, returns an empty dict. Subclasses may override to
        provide environment specific information.

        Parameters
        ----------
        env : Environment
            The current state of the environment.

        Returns
        -------
        Dict
            A dictionary with additional information about the environment.
        """
        return dict()


from .multi_navigator import MultiNavigator
from .single_navigator import SingleNavigator

__all__ = ["Environment", "MultiNavigator", "SingleNavigator"]
