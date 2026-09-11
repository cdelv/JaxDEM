# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Composite force model that sums multiple force laws."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

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
    r"""A `ForceModel` that sums a tuple of elementary force laws.

    The total force, torque, and potential energy of the interaction between
    particles :math:`i` and :math:`j` are the sums over the contained laws:

    .. math::
        F_{ij} = \sum_k F^{(k)}_{ij}, \qquad
        \tau_{ij} = \sum_k \tau^{(k)}_{ij}, \qquad
        E_{ij} = \sum_k E^{(k)}_{ij}

    Notes
    -----
    - The combiner evaluates each sub-law with a system whose ``force_model``
      is the sub-law itself. Laws that read their own configuration from
      :attr:`jaxdem.System.force_model` (including nested combiners) work
      correctly.
    - An empty combiner (``laws=()``) returns zero force, torque, and energy.
      :meth:`ForceRouter.from_dict` uses it as the default no-interaction law.
    - :attr:`required_material_properties` is the union of the requirements of
      all contained laws.
    """

    laws: tuple[ForceModel, ...] = ()
    """Tuple of elementary :class:`ForceModel` instances to sum as pytree data."""

    @property
    def supports_analytical_energy_gradient(self) -> bool:
        """Whether every contained law supports analytical minimization."""
        return all(law.supports_analytical_energy_gradient for law in self.laws)

    def history_shape(self, dim: int) -> tuple[int, ...]:
        return (sum(law.history_shape(dim)[0] for law in self.laws),)

    def init_history(self, pair_shape: tuple[int, ...], dim: int) -> jax.Array:
        histories = [law.init_history(pair_shape, dim) for law in self.laws]
        if not histories:
            return ForceModel.init_history(self, pair_shape, dim)
        return jnp.concatenate(histories, axis=-1)

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Conservative per-primitive reach across every contained law."""
        radii = jnp.zeros_like(state.rad)
        for law in self.laws:
            sub_system = dataclasses.replace(system, force_model=law)
            radii = jnp.maximum(radii, law.search_radii(state, sub_system))
        return radii

    @staticmethod
    @jax.jit(static_argnames=("advance_history",))
    @partial(jax.named_call, name="LawCombiner.force")
    def force(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
        *,
        advance_history: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        f_shape = jnp.shape(j) + jnp.shape(state.force[i])
        t_shape = jnp.shape(j) + jnp.shape(state.torque[i])
        force = jnp.zeros(f_shape, dtype=state.force.dtype)
        torque = jnp.zeros(t_shape, dtype=state.torque.dtype)
        combiner = cast(LawCombiner, system.force_model)
        new_histories = []
        offset = 0
        for law in combiner.laws:
            width = law.history_shape(pos.shape[-1])[0]
            h = history[..., offset : offset + width]
            sub_system = dataclasses.replace(system, force_model=law)
            f, t, nh = law.force(
                i,
                j,
                pos,
                state,
                sub_system,
                h,
                advance_history=advance_history,
            )
            force += f
            torque += t
            new_histories.append(nh)
            offset += width
        return (
            force,
            torque,
            jnp.concatenate(new_histories, axis=-1) if new_histories else history,
        )

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """Names of the material properties this force model needs.

        The sorted union of the material properties required by all contained
        laws. Each name must be present in :attr:`System.mat_table`. Used for
        validation.
        """
        return tuple(
            sorted({p for lw in self.laws for p in lw.required_material_properties})
        )

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
            Scalar total potential energy of the interaction between
            particles :math:`i` and :math:`j`.

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
