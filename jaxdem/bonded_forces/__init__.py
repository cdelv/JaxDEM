# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bonded-force interfaces independent of the collider."""

from __future__ import annotations

import jax

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from ..factory import Factory

if TYPE_CHECKING:  # pragma: no cover
    from ..forces.force_manager import ForceFunction, EnergyFunction


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
    def force_and_energy_fns(self) -> [ForceFunction, EnergyFunction, bool]:
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


from .deformable_particle import DeformableParticleModel

__all__ = ["BonndedForceModel", "DeformableParticleModel"]
