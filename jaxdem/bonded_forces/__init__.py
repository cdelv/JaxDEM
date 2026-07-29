# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bonded-force interfaces independent of the collider."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar
from collections.abc import Sequence

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..forces.force_manager import ForceFunction, EnergyFunction
    from ..state import State
    from ..system import System

BondedT = TypeVar("BondedT", bound="BondedForceModel")


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class BondedForceModel(Factory, ABC):
    """Abstract interface for bonded interaction containers.

    Design intent
    -------------
    - :py:class:`~jaxdem.system.System` stores one concrete bonded container
      instance.
    - The container exposes bonded force/energy callables through
      ``force_and_energy_fns``.
    - The :py:class:`~jaxdem.forces.force_manager.ForceManager` gets and runs
      these callables to compute bonded contributions at each time step.
    - The bonded data stays accessible through
      :py:class:`~jaxdem.system.System`, so the force/energy callables can
      read what they need.
    """

    @property
    def force_and_energy_fns(self) -> tuple[ForceFunction, EnergyFunction, bool]:
        """Build the bonded force/energy callables for the force manager.

        Returns
        -------
        Tuple[ForceFunction, EnergyFunction, bool]
            ``(force_fn, energy_fn, is_com_force)`` where:

            - ``force_fn`` computes bonded force and torque contributions.
            - ``energy_fn`` computes bonded potential-energy contributions.
            - ``is_com_force`` tells where the force acts: ``True`` for the
              center of mass, ``False`` for the contact point. It has no
              effect on spheres.

        """
        raise NotImplementedError

    @jax.jit(inline=True)
    @abstractmethod
    def compute_potential_energy(
        self,
        pos: jax.Array,
        state: State,
        system: System,
    ) -> jax.Array:
        """Compute the total bonded potential energy of the system."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def merge(
        model1: BondedForceModel,
        model2: BondedForceModel | Sequence[BondedForceModel],
    ) -> BondedForceModel:
        """Merge two or more bonded-force models into one.

        The merge concatenates topology, reference, and coefficient arrays.
        It shifts vertex indices and body IDs automatically so references
        stay consistent. When one side has a term that the other does not,
        the merge pads missing coefficients with ``0`` and missing reference
        values with ``1``.

        Parameters
        ----------
        model1 : BondedForceModel
            Base model.
        model2 : BondedForceModel or Sequence[BondedForceModel]
            Model(s) to merge into *model1*.

        Returns
        -------
        BondedForceModel
            A new model containing all bodies from both sides.

        """
        ...

    @staticmethod
    @abstractmethod
    def add(model: BondedForceModel, **kwargs: Any) -> BondedForceModel:
        """Create a new body from raw arrays and merge it into an existing model.

        This is a convenience wrapper equivalent to calling the concrete
        ``Create`` constructor followed by :meth:`merge`.

        Parameters
        ----------
        model : BondedForceModel
            Existing model to extend.
        **kwargs
            Constructor arguments forwarded to the concrete ``Create`` method
            (e.g. ``vertices``, ``elements``, coefficients, …).

        Returns
        -------
        BondedForceModel
            The extended model.

        """
        ...

    @classmethod
    @partial(jax.named_call, name="BondedForceModel.stack")
    def stack(cls: type[BondedT], models: Sequence[BondedT]) -> BondedT:
        models = list(models)
        if not models:
            raise ValueError("BondedForceModel.stack() received an empty list")

        ref_tree = jax.tree.structure(models[0])
        for m in models[1:]:
            if str(jax.tree.structure(m)) != str(ref_tree):
                raise ValueError(
                    "BondedForceModel.stack() expects identical field structure across models."
                )

        return jax.tree.map(lambda *xs: jnp.stack(xs), *models)

    @classmethod
    @partial(jax.named_call, name="BondedForceModel.unstack")
    def unstack(cls: type[BondedT], model: BondedT) -> list[BondedT]:
        leaves = jax.tree.leaves(model)
        sized_leaves = [x for x in leaves if isinstance(x, jax.Array) and x.ndim >= 1]
        if not sized_leaves:
            raise ValueError(
                "BondedForceModel.unstack() expects a stacked model with a leading axis."
            )

        n = int(sized_leaves[0].shape[0])
        for x in sized_leaves[1:]:
            if int(x.shape[0]) != n:
                raise ValueError(
                    "BondedForceModel.unstack() found inconsistent leading axis sizes."
                )

        return [jax.tree.map(lambda x, i=i: x[i], model) for i in range(n)]


from .deformable_particle import DeformableParticleModel
from .plastic_deformable_particle import PlasticDeformableParticleModel
from .plastic_bending_deformable_particle import (
    PlasticBendingDeformableParticleModel,
)
from .plastic_perimeter_deformable_particle import (
    PlasticPerimeterDeformableParticleModel,
)

__all__ = [
    "BondedForceModel",
    "DeformableParticleModel",
    "PlasticBendingDeformableParticleModel",
    "PlasticDeformableParticleModel",
    "PlasticPerimeterDeformableParticleModel",
]
