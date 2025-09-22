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
    Defines the interface for environments.

    - Let **A** = number of agents (A ≥ 1). **Single-agent envs must still use A=1.**
    - Observations and actions are **flattened per agent** to fixed sizes.
    and ``action_space_shape`` to reshape if needed.

    **Required shapes**
    - Observation: ``(A, observation_space_size)``
    - Action (input to :meth:`step`): ``(A, action_space_size)``
    - Reward: ``(A,)``
    - Done: scalar boolean for the **whole environment**

    TO DO: truncated data field: per agent termination flag
    TO DO: render method

    Example
    -------
    To define a custom integrator, inherit from :class:`Integrator` and implement its abstract methods:

    >>> @Integrator.register("myCustomIntegrator")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True, frozen=True)
    >>> class MyCustomIntegrator(Integrator):
        ...
    """

    state: "State"
    """
    Simulation state
    """

    system: "System"
    """
    Simulation system's configuration
    """

    env_params: Dict[str, Any]
    """
    Environment specific parameters
    """

    max_num_agents: int = field(default=0, metadata={"static": True})
    """
    Maximun number of active agents in the environment
    """

    action_space_size: int = field(default=0, metadata={"static": True})
    """
    Flattened action size per agent: actions passed to :meth:`step` are shape ``(A, action_space_size)``.
    """

    action_space_shape: Tuple[int, ...] = field(default=(), metadata={"static": True})
    """
    Original per-agent action shape (useful for reshaping inside the env).
    """

    observation_space_size: int = field(default=0, metadata={"static": True})
    """
    Flattened observation size per agent: :meth:`observation` returns shape ``(A, observation_space_size)
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
            Jax random numbers key

        Returns
        -------
        Tuple[Environment, ArrayLike]
            Freshly initialized environment and new random numbers key.
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

        action : System
            The vector of actions each agent on the environment should take.

        Returns
        -------
        Environment
            The updated envitonment state.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    @jax.jit
    def observation(env: "Environment") -> jax.Array:
        """
        Return the **per-agent** observation vector.

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
        Return the **per-agent** immediate rewards.

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
        Return a bool indicating when the environment ended.

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
            A dictionary with aditional information of the environment.
        """
        return dict()


from .multi_navigator import MultiNavigator
from .single_navigator import SingleNavigator

__all__ = ["Environment", "MultiNavigator", "SingleNavigator"]
