# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Composite force model that sums multiple force laws."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..state import State
    from ..system import System

from . import ForceModel


@ForceModel.register("lawcombiner")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LawCombiner(ForceModel):
    r"""A `ForceModel` implementation that sums a tuple of elementary force laws.

    The total force, torque, and potential energy of the interaction between
    particles :math:`i` and :math:`j` are the sums over the contained laws:

    .. math::
        F_{ij} = \sum_k F^{(k)}_{ij}, \qquad
        \tau_{ij} = \sum_k \tau^{(k)}_{ij}, \qquad
        E_{ij} = \sum_k E^{(k)}_{ij}

    Notes
    -----
    - Each sub-law is evaluated with a system whose ``force_model`` is the
      sub-law itself, so laws that read their own configuration from
      :attr:`jaxdem.System.force_model` (including nested combiners) work correctly.
    - An empty combiner (``laws=()``) returns zero force, torque, and energy.
      :meth:`ForceRouter.from_dict` uses it as the default no-interaction law.
    - :attr:`required_material_properties` is the union of the requirements of
      all contained laws.
    """

    laws: tuple[ForceModel, ...] = jax.tree.static(default=())
    """A static tuple of the elementary :class:`ForceModel` instances to sum."""

    @property
    def requires_history(self) -> bool:
        return any(law.requires_history for law in self.laws)

    @jax.jit(inline=True)
    def init_history(self, shape: tuple[int, ...]) -> Any:
        return tuple(
            law.init_history(shape) if law.requires_history else None
            for law in self.laws
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LawCombiner.force_and_history")
    def force_and_history(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: Any,
    ) -> tuple[jax.Array, jax.Array, Any]:
        f_shape = jnp.shape(j) + jnp.shape(state.force[i])
        t_shape = jnp.shape(j) + jnp.shape(state.torque[i])
        force = jnp.zeros(f_shape, dtype=state.force.dtype)
        torque = jnp.zeros(t_shape, dtype=state.torque.dtype)
        combiner = cast(LawCombiner, system.force_model)
        new_histories = []
        for law, h in zip(combiner.laws, history):
            sub_system = dataclasses.replace(system, force_model=law)
            if law.requires_history:
                f, t, nh = law.force_and_history(i, j, pos, state, sub_system, h)
            else:
                f, t = law.force(i, j, pos, state, sub_system)
                nh = h
            force += f
            torque += t
            new_histories.append(nh)
        return force, torque, tuple(new_histories)

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """A static tuple of strings specifying the material properties required by this force model.

        The sorted union of the material properties required by all contained
        laws. These properties must be present in the :attr:`System.mat_table`
        for the model to function correctly. This is used for validation.
        """
        return tuple(
            sorted({p for lw in self.laws for p in lw.required_material_properties})
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LawCombiner.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute the total force and torque acting on particle :math:`i` due to particle :math:`j` by summing all contained laws.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        pos : jax.Array
            Particle positions used to evaluate the interaction.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            A tuple ``(force, torque)`` with the sums of the forces and torques
            of all contained laws acting on particle :math:`i` due to particle :math:`j`.

        """
        f_shape = jnp.shape(j) + jnp.shape(state.force[i])
        t_shape = jnp.shape(j) + jnp.shape(state.torque[i])
        force = jnp.zeros(f_shape, dtype=state.force.dtype)
        torque = jnp.zeros(t_shape, dtype=state.torque.dtype)
        combiner = cast(LawCombiner, system.force_model)
        for law in combiner.laws:
            # Each sub-law sees a system whose force_model is itself, so laws
            # that read their own config from system.force_model (including
            # nested combiners) work correctly.
            sub_system = dataclasses.replace(system, force_model=law)
            f, t = law.force(i, j, pos, state, sub_system)
            force += f
            torque += t
        return force, torque

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LawCombiner.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        """Compute the total potential energy of the interaction between particle :math:`i` and particle :math:`j` by summing all contained laws.

        Parameters
        ----------
        i : int
            Index of the first particle.
        j : int
            Index of the second particle.
        pos : jax.Array
            Particle positions used to evaluate the interaction.
        state : State
            Current state of the simulation.
        system : System
            Simulation system configuration.

        Returns
        -------
        jax.Array
            Scalar JAX array representing the total potential energy of the
            interaction between particles :math:`i` and :math:`j`.

        """
        # Initialize accumulator with shape of j. If j is an array (e.g., neighbor list),
        # this accumulator will broadcast properly. If j is a scalar, it's a scalar.
        e = jnp.zeros(jnp.shape(j), dtype=float)
        combiner = cast(LawCombiner, system.force_model)
        for law in combiner.laws:
            sub_system = dataclasses.replace(system, force_model=law)
            e = e + law.energy(i, j, pos, state, sub_system)
        return e


__all__ = ["LawCombiner"]
