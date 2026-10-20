# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reinforcement-learning environment interface."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ...state import State
    from ...system import System


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Environment(Factory, ABC):
    """Defines the interface for reinforcement-learning environments.

    - Let **A** be the number of agents (A ≥ 1). Single-agent environments still use A=1.
    - The environment flattens observations and actions per agent to fixed sizes.
      Use ``action_space_shape`` to reshape inside the environment if needed.

    **Required shapes**

    - Observation: ``(A, observation_space_size)``
    - Action (input to :meth:`step`): ``(A, action_space_size)``
    - Reward: ``(A,)``
    - Done: scalar boolean for the whole environment

    Todo:
    - Truncated data field: per-agent termination flag
    - Render method

    Example:
    --------
    To define a custom environment, inherit from :class:`Environment`. Implement the abstract methods:

    >>> @Environment.register("MyCustomEnv")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomEnv(Environment):
        ...

    """

    state: State
    """
    Simulation state.
    """

    system: System
    """
    Simulation system configuration.
    """

    env_params: dict[str, Any]
    """
    Environment-specific parameters.
    """

    _base_env_cls: ClassVar[type[Environment]]

    @classmethod
    @jax.tree_util.Partial(jax.named_call, name="Environment.Create")  # type: ignore[no-untyped-call]
    def Create(cls, dim: int = 2) -> Environment:
        from ...state import State
        from ...system import System

        state = State.create(pos=jnp.zeros((1, dim)))
        system = System.create(state.shape)
        return cls(state=state, system=system, env_params={})

    @staticmethod
    @jax.jit
    def reset(env: Environment, key: ArrayLike) -> Environment:
        """Initialize the environment to a valid start state.

        Parameters
        ----------
        env: 'MyCustomEnv'
            The current environment.

        key : jax.random.PRNGKey
            JAX random number generator key.

        Returns
        -------
        Environment
            The initialized environment.

        """
        return env

    @staticmethod
    @jax.jit
    def reset_if_done(env: Environment, done: jax.Array, key: ArrayLike) -> Environment:
        """Reset the environment when ``done`` is True.

        When ``done`` is True, this method calls the environment's ``reset``
        method. When ``done`` is False, it returns the current environment
        unchanged.

        Parameters
        ----------
        env : Environment
            The current environment.

        done : jax.Array
            A boolean flag that is True at either termination or truncation.

        key : jax.random.PRNGKey
            JAX random number generator key used for the reset.

        Returns
        -------
        Environment
            The reset environment when ``done`` is True, otherwise the
            unchanged environment.

        """
        base_cls = getattr(env.__class__, "_base_env_cls", env.__class__)
        reset_env = base_cls.reset(env, key)
        return jax.tree.map(partial(jnp.where, done), reset_env, env)

    @staticmethod
    @jax.jit(inline=True)
    def step(env: Environment, action: jax.Array) -> Environment:
        """Advance one physics step using **per-agent** actions.

        Advance only the dynamics; preserve historical reward baselines.
        Use ``utils.advance_action`` to snapshot those baselines once before
        repeating physics steps. Observations and rewards read the live state.

        Parameters
        ----------
        env : Environment
            The current environment.

        action : jax.Array
            The per-agent action vectors.

        Returns
        -------
        Environment
            The updated environment state.

        """
        return env

    @staticmethod
    @jax.jit
    def checkpoint(env: Environment, action: jax.Array) -> Environment:
        """Snapshot historical reward quantities before an action interval.

        The default implementation is a no-op. Environments whose rewards compare
        states across time override this hook to store only the required baseline
        quantities from the current physical state. Derived current quantities
        need not be stored: observation and reward accessors can compute them
        from the live state without modifying the baseline.

        Parameters
        ----------
        env : Environment
            The environment at the start of the next action interval.

        action : jax.Array
            The per-agent action about to be applied. Overrides may use it
            when capturing action-dependent history. Wrappers must pass the
            same transformed action to this hook and to physics steps.

        Returns
        -------
        Environment
            The environment with its historical reward baseline captured.

        Notes
        -----
        ``utils.advance_action`` calls this hook exactly once before its physics
        loop, including when ``skip_frames=0``. Physics steps must preserve the
        baseline until reward and any truncation bootstrap have been consumed.
        There is no end-of-action checkpoint. Reset must initialize the baseline
        from the reset state without requiring a policy action.

        The hook does not advance physics or reset episodes. Boundary predicates,
        observations, and rewards must remain valid for the live physical state.
        For a difference reward, checkpointing a frozen environment captures its
        unchanged endpoint and gives zero change on the next evaluation interval.
        This is a reward-history snapshot, not a disk checkpoint operation.
        """
        return env

    @staticmethod
    @jax.jit
    def observation(env: Environment) -> jax.Array:
        """Return per-agent observations derived from the live environment.

        This accessor must not require an endpoint checkpoint or change the
        historical reward baseline.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Per-agent observations, shape ``(A, observation_space_size)``.

        """
        return jnp.zeros((env.max_num_agents, env.observation_space_size), dtype=float)

    @staticmethod
    @jax.jit
    def reward(env: Environment) -> jax.Array:
        """Return per-agent rewards for the live state and saved action baseline.

        Historical terms compare the live endpoint with quantities saved by
        ``checkpoint`` before the action. Read this accessor before reset or
        the next checkpoint overwrites that baseline. Accessors must not
        mutate history; repeated reads of the same state return the same reward.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            Per-agent rewards for the current state, shape ``(A,)``.

        """
        return jnp.zeros((env.max_num_agents,), dtype=float)

    @staticmethod
    @jax.jit
    def done(env: Environment) -> jax.Array:
        """Return whether the episode terminated or was truncated.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        jax.Array
            A bool that is True when the episode has ended.

        Notes
        -----
        Environment subclasses should override ``terminated`` and/or
        ``truncated``. The trainer reads those causes separately; overriding
        only ``done`` does not signal a boundary to its physics loop.
        Leave the final state available in ``step`` so the trainer can read
        its reward and bootstrap observation before resetting the episode.

        """
        base_cls = getattr(env.__class__, "_base_env_cls", env.__class__)
        return base_cls.terminated(env) | base_cls.truncated(env)

    @staticmethod
    @jax.jit
    def terminated(env: Environment) -> jax.Array:
        """Return a terminal episode boundary.

        Terminal transitions do not bootstrap their value target.
        """
        return jnp.asarray(False, dtype=bool)

    @staticmethod
    @jax.jit
    def truncated(env: Environment) -> jax.Array:
        """Return a time-limit or external episode boundary.

        Truncated transitions reset the episode and recurrent carry but retain
        value bootstrapping from the final pre-reset observation, unless the
        same transition is also terminal. The default is ``False``.
        """
        return jnp.asarray(False, dtype=bool)

    @staticmethod
    @jax.jit
    def agent_mask(env: Environment) -> jax.Array:
        """Return active-agent flags with shape ``(max_num_agents,)``.

        At least one agent must remain active in every environment transition.
        Observations and rewards, including inactive padding, must be finite.
        """
        return jnp.ones((env.max_num_agents,), dtype=bool)

    @staticmethod
    @jax.jit
    def info(env: Environment) -> dict[str, Any]:
        """Return auxiliary diagnostic information.

        The default is an empty dict. Subclasses can override this method to
        provide environment-specific information.

        Parameters
        ----------
        env : Environment
            The current environment.

        Returns
        -------
        Dict
            A dictionary with additional information about the environment.

        """
        return {}

    @property
    def num_envs(self) -> int:
        """Number of batched environments."""
        return self.state.batch_size

    @property
    def max_num_agents(self) -> int:
        """Maximum number of active agents in the environment."""
        return self.state.N

    @property
    def action_space_size(self) -> int:
        """Flattened action size per agent. Actions passed to :meth:`step` have shape ``(A, action_space_size)``."""
        return 1

    @property
    def action_space_shape(self) -> tuple[int]:
        """Original per-agent action shape (useful for reshaping inside the environment)."""
        return (1,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent. :meth:`observation` returns shape ``(A, observation_space_size)``."""
        return 1


Environment.register("")(Environment)


from .granulabot import Granulabot
from .multi_navigator import MultiNavigator
from .multi_roller import MultiRoller
from .single_navigator import SingleNavigator
from .single_roller import SingleRoller
from .swarm_navigator import SwarmNavigator
from .swarm_roller import SwarmRoller
from .three_gears import ThreeGears
from .two_gears import TwoGears

__all__ = [
    "Environment",
    "Granulabot",
    "MultiNavigator",
    "MultiRoller",
    "SingleNavigator",
    "SingleRoller",
    "SwarmNavigator",
    "SwarmRoller",
    "ThreeGears",
    "TwoGears",
]
