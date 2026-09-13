# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, cast

from . import ForceModel
from ..utils.linalg import norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("lennardjones")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LennardJones(ForceModel):
    r"""Lennard-Jones (LJ) 12-6 interaction with a per-pair cutoff and energy shift.

    The model reads the material-pair parameter ``epsilon_eff[mi, mj]``.

    The model derives the length scale :math:`\sigma_{ij}` from the particle
    radii (as in `spring.py`):

    .. math::
        \sigma_{ij} = R_i + R_j

    Potential (for :math:`r < r_c = c \sigma_{ij}`, with ``cutoff_ratio`` :math:`c=2.5` by default):

    .. math::
        U(r) = 4 \epsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]
               - U(r_c)

    else:

    .. math::
        U(r) = 0

    Force (for :math:`r < r_c`):

    .. math::
        \mathbf{F} = 24 \epsilon \left(2 \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right)
                     \frac{1}{r^2}\, \mathbf{r}_{ij}
    """

    cutoff_ratio: jax.Array = field(default_factory=lambda: jnp.asarray(2.5))
    """Pair cutoff in units of ``rad[i] + rad[j]``; also sets collider search reach."""

    @staticmethod
    def Create(cutoff_ratio: float = 2.5) -> LennardJones:
        """Create an LJ law with a positive finite cutoff measured in pair sigma."""
        cutoff = jnp.asarray(cutoff_ratio, dtype=float)
        if cutoff.ndim != 0 or not bool(jnp.isfinite(cutoff) & (cutoff > 0)):
            raise ValueError("cutoff_ratio must be a positive finite scalar.")
        return LennardJones(cutoff_ratio=cutoff)

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Conservative search extent for this law's finite interaction range."""
        return jnp.maximum(state._rad, self.cutoff_ratio * state.rad)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="LennardJones.force")
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
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain._displacement(pos[i], pos[j], system)
        r2 = norm2(rij)
        r2 = jnp.where(r2 == 0, jnp.ones_like(r2), r2)

        sig2 = sig * sig
        inv_r2 = 1.0 / r2
        sr2 = sig2 * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        cutoff = cast(LennardJones, system.force_model).cutoff_ratio
        rc2 = cutoff * cutoff * sig2
        active = r2 < rc2
        not_self = j != i
        mask = active * not_self

        coeff = 24.0 * eps * inv_r2 * (2.0 * sr12 - sr6)
        f = (coeff * mask)[..., None] * rij

        t_shape = jnp.shape(j) + jnp.shape(state.torque[i])
        return f, jnp.zeros(t_shape, dtype=state.torque.dtype), history

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="LennardJones.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain._displacement(pos[i], pos[j], system)
        r2 = norm2(rij)
        r2 = jnp.where(r2 == 0, jnp.ones_like(r2), r2)

        sig2 = sig * sig
        inv_r2 = 1.0 / r2
        sr2 = sig2 * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        cutoff = cast(LennardJones, system.force_model).cutoff_ratio
        rc2 = cutoff * cutoff * sig2
        active = r2 < rc2
        not_self = j != i
        mask = active * not_self

        # Shift the potential to zero at the configured pair cutoff.
        inv_rc6 = (1.0 / cutoff) ** 6
        u_shift = 4.0 * eps * (inv_rc6 * inv_rc6 - inv_rc6)

        u = 4.0 * eps * (sr12 - sr6) - u_shift
        return u * mask

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        return ("epsilon_eff",)


__all__ = ["LennardJones"]
