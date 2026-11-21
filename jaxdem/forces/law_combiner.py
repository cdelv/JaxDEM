# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Composite force model that sums multiple force laws."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import dataclasses
import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from . import ForceModel

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


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
      :attr:`jaxdem.System.force_model <jaxdem.system.System.force_model>` (including nested combiners) work
      correctly.
    - An empty combiner (``laws=()``) returns zero force, torque, and energy.
      :meth:`ForceRouter.from_dict <jaxdem.forces.router.ForceRouter.from_dict>` uses it as the default no-interaction law.
    - :attr:`required_material_properties <jaxdem.forces.law_combiner.LawCombiner.required_material_properties>` is the union of the requirements of
      all contained laws.
    """

    laws: tuple[ForceModel, ...] = ()
    """Tuple of elementary :class:`ForceModel <jaxdem.forces.ForceModel>` instances to sum as pytree data."""

    @property
    def supports_analytical_energy_gradient(self) -> bool:
        """Whether every contained law supports analytical minimization."""
        return all(law.supports_analytical_energy_gradient for law in self.laws)

    @property
    def has_pair_torque(self) -> bool:
        return any(law.has_pair_torque for law in self.laws)

    @property
    def supports_sorted_order(self) -> bool:
        return all(law.supports_sorted_order for law in self.laws)

    @property
    def species_capacity(self) -> int | None:
        """Smallest router capacity reachable through the contained laws."""
        capacities = [
            capacity
            for law in self.laws
            if (capacity := law.species_capacity) is not None
        ]
        return min(capacities) if capacities else None

    def history_shape(self, dim: int) -> tuple[int, ...]:
        return (sum(math.prod(law.history_shape(dim)) for law in self.laws),)

    def init_history(self, pair_shape: tuple[int, ...], dim: int) -> jax.Array:
        histories = [
            law.init_history(pair_shape, dim).reshape(
                (*pair_shape, math.prod(law.history_shape(dim)))
            )
            for law in self.laws
        ]
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
    @jax.jit(static_argnames=("advance_history", "quasistatic"), inline=True)
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
        quasistatic: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        combiner = cast(LawCombiner, system.force_model)
        if not combiner.laws:
            f_shape = jnp.shape(j) + jnp.shape(state.force[i])
            t_shape = jnp.shape(j) + jnp.shape(state.torque[i])
            return (
                jnp.zeros(f_shape, dtype=state.force.dtype),
                jnp.zeros(t_shape, dtype=state.torque.dtype),
                history,
            )
        has_history = history.shape[-1] > 0
        force: jax.Array | None = None
        torque: jax.Array | None = None
        new_histories: list[jax.Array] = []
        offset = 0
        for law in combiner.laws:
            child_shape = law.history_shape(pos.shape[-1])
            width = math.prod(child_shape)
            h = history[..., offset : offset + width].reshape(
                (*history.shape[:-1], *child_shape)
            )
            sub_system = dataclasses.replace(system, force_model=law)
            f, t, nh = law.force(
                i,
                j,
                pos,
                state,
                sub_system,
                h,
                advance_history=advance_history,
                quasistatic=quasistatic,
            )
            force = f if force is None else force + f
            torque = t if torque is None else torque + t
            if has_history and width > 0:
                new_histories.append(nh.reshape((*history.shape[:-1], width)))
            offset += width
        assert force is not None and torque is not None
        if not has_history:
            new_history = history
        elif len(new_histories) == 1:
            new_history = new_histories[0]
        else:
            new_history = jnp.concatenate(new_histories, axis=-1)
        return force, torque, new_history

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        """Names of the material properties this force model needs.

        The sorted union of the material properties required by all contained
        laws. Each name must be present in :attr:`System.mat_table <jaxdem.system.System.mat_table>`. Used for
        validation.
        """
        return tuple(
            sorted({p for lw in self.laws for p in lw.required_material_properties})
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="LawCombiner.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        """Sum the pair energies of the contained laws.

        Each law receives its own slice of the concatenated pair history.

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
        history : jax.Array
            Concatenated pair history of the contained laws.

        Returns
        -------
        jax.Array
            Scalar total potential energy of the interaction between
            particles :math:`i` and :math:`j`.

        """
        e = jnp.zeros(jnp.shape(j), dtype=float)
        combiner = cast(LawCombiner, system.force_model)
        offset = 0
        for law in combiner.laws:
            child_shape = law.history_shape(pos.shape[-1])
            width = math.prod(child_shape)
            child_history = history[..., offset : offset + width].reshape(
                (*history.shape[:-1], *child_shape)
            )
            sub_system = dataclasses.replace(system, force_model=law)
            e = e + law.energy(i, j, pos, state, sub_system, child_history)
            offset += width
        return e


__all__ = ["LawCombiner"]
