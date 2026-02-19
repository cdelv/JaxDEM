# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bonded-force interfaces independent of the collider."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Sequence, Tuple, TypeVar

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..forces.force_manager import ForceFunction, EnergyFunction

BondedT = TypeVar("BondedT", bound="BonndedForceModel")


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class BonndedForceModel(Factory, ABC):
    """
    Abstract interface for bonded interaction containers.

    This class is intended as the general bonded-force abstraction in JaxDEM.

    Design intent
    -------------
    - A concrete bonded container instance is stored on
      :py:class:`~jaxdem.system.System`.
    - The container exposes bonded force/energy callables through
      :meth:`create_force_and_energy_fns`.
    - The :py:class:`~jaxdem.forces.force_manager.ForceManager` obtains and
      executes these callables to compute bonded contributions at each time step.
    - Bonded data remains accessible through :py:class:`~jaxdem.system.System`
      (via the container itself), so the force/energy callables can read what they
      need.
    """

    @property
    def force_and_energy_fns(self) -> Tuple[ForceFunction, EnergyFunction, bool]:
        """
        Build bonded force/energy callables consumed by the force manager.

        Returns
        -------
        Tuple[ForceFunction, EnergyFunction, bool]
            ``(force_fn, energy_fn, is_com_force)`` where:

            - ``force_fn`` computes bonded force and torque contributions.
            - ``energy_fn`` computes bonded potential-energy contributions.
            - ``is_com_force`` indicates where force is applied:
              ``True`` for center-of-mass application, ``False`` for
              contact-point application. This has no effect on spheres.
        """
        raise NotImplementedError

    @classmethod
    @partial(jax.named_call, name="BonndedForceModel.stack")
    def stack(cls: type[BondedT], models: Sequence[BondedT]) -> BondedT:
        models = list(models)
        if not models:
            raise ValueError("BonndedForceModel.stack() received an empty list")

        ref_tree = jax.tree_util.tree_structure(models[0])
        for m in models[1:]:
            if str(jax.tree_util.tree_structure(m)) != str(ref_tree):
                raise ValueError(
                    "BonndedForceModel.stack() expects identical field structure across models."
                )

        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *models)

    @classmethod
    @partial(jax.named_call, name="BonndedForceModel.unstack")
    def unstack(cls: type[BondedT], model: BondedT) -> list[BondedT]:
        leaves = jax.tree_util.tree_leaves(model)
        sized_leaves = [x for x in leaves if isinstance(x, jax.Array) and x.ndim >= 1]
        if not sized_leaves:
            raise ValueError(
                "BonndedForceModel.unstack() expects a stacked model with a leading axis."
            )

        n = int(sized_leaves[0].shape[0])
        for x in sized_leaves[1:]:
            if int(x.shape[0]) != n:
                raise ValueError(
                    "BonndedForceModel.unstack() found inconsistent leading axis sizes."
                )

        return [jax.tree_util.tree_map(lambda x, i=i: x[i], model) for i in range(n)]


from .deformable_particle import DeformableParticleModel

__all__ = ["BonndedForceModel", "DeformableParticleModel"]
