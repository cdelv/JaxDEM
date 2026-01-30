from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple
from functools import partial

from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("wca")
@partial(jax.tree_util.register_dataclass, drop_fields=["required_material_properties"])
@dataclass(slots=True)
class WCA(ForceModel):
    r"""
    Weeks-Chandler-Andersen (WCA) purely repulsive Lennard-Jones interaction.

    Uses material-pair parameter:
      - epsilon_eff[mi, mj]

    The length scale :math:`\sigma_{ij}` is derived from particle radii (like `spring.py`):

    .. math::
        \sigma_{ij} = R_i + R_j

    Potential (for r < r_c = 2^(1/6) sigma):
      U(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] + eps
    else:
      U(r) = 0

    Force:
      F_vec = 24 eps (2 (sigma/r)^12 - (sigma/r)^6) * (1/r^2) * r_ij
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="WCA.force")
    def force(
        i: int, j: int, pos: jax.Array, state: "State", system: "System"
    ) -> Tuple[jax.Array, jax.Array]:
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain.displacement(pos[i], pos[j], system)
        r2 = jnp.sum(rij * rij, axis=-1)  # (...)
        r2 = jnp.where(r2 == 0, jnp.ones_like(r2), r2)

        sig2 = sig * sig
        inv_r2 = 1.0 / r2
        sr2 = sig2 * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        # cutoff: r_c = 2^(1/6) sigma  => r_c^2 = 2^(1/3) sigma^2
        rc2 = (2.0 ** (1.0 / 3.0)) * sig2
        active = r2 < rc2
        not_self = j != i
        mask = active & not_self

        coeff = 24.0 * eps * inv_r2 * (2.0 * sr12 - sr6)
        f = (coeff[..., None] * rij) * mask[..., None]

        return f, jnp.zeros_like(state.angVel[i])

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="WCA.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: "State", system: "System"
    ) -> jax.Array:
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain.displacement(pos[i], pos[j], system)
        r2 = jnp.sum(rij * rij, axis=-1)
        r2 = jnp.where(r2 == 0, jnp.ones_like(r2), r2)

        sig2 = sig * sig
        inv_r2 = 1.0 / r2
        sr2 = sig2 * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        rc2 = (2.0 ** (1.0 / 3.0)) * sig2
        active = r2 < rc2
        not_self = j != i
        mask = active & not_self

        u = 4.0 * eps * (sr12 - sr6) + eps
        return u * mask

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        return ("epsilon_eff",)