# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("wca_shifted")
@partial(jax.tree_util.register_dataclass, drop_fields=["required_material_properties"])
@dataclass(slots=True)
class WCAShifted(ForceModel):
    r"""
    Contact-start, force-shifted WCA/LJ repulsion.

    This model enforces that the interaction "begins" at contact:

    - cutoff at :math:`r_c = \sigma_{ij}` where :math:`\sigma_{ij} = R_i + R_j`
    - :math:`U(r_c) = 0`
    - :math:`F(r_c) = 0` (force-shifted; smooth turn-on at contact)

    Uses material-pair parameter:
      - epsilon_eff[mi, mj]
    """

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="WCAShifted.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> Tuple[jax.Array, jax.Array]:
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain.displacement(pos[i], pos[j], system)
        r2 = jnp.sum(rij * rij, axis=-1)
        r2 = jnp.where(r2 == 0, jnp.ones_like(r2), r2)

        sig2 = sig * sig
        inv_r2 = 1.0 / r2
        inv_r = jnp.sqrt(inv_r2)  # == 1/sqrt(r2)
        sr2 = sig2 * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        # cutoff at contact: r_c = sigma
        rc2 = sig2
        active = r2 < rc2
        not_self = j != i
        mask = active & not_self

        # Unit vector rhat = rij / r
        rhat = rij * inv_r[..., None]

        # LJ force magnitude along rhat: F(r) = 24 eps (2 sr12 - sr6) / r
        fmag = 24.0 * eps * inv_r * (2.0 * sr12 - sr6)
        # Force-shift so that F(rc) = 0. At rc = sigma: sr6=sr12=1 => F(rc) = 24 eps / sigma
        fmag_rc = 24.0 * eps / sig
        fmag_fs = fmag - fmag_rc

        f = (fmag_fs[..., None] * rhat) * mask[..., None]
        return f, jnp.zeros_like(state.ang_vel[i])

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="WCAShifted.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain.displacement(pos[i], pos[j], system)
        r2 = jnp.sum(rij * rij, axis=-1)
        r2 = jnp.where(r2 == 0, jnp.ones_like(r2), r2)

        sig2 = sig * sig
        inv_r2 = 1.0 / r2
        inv_r = jnp.sqrt(inv_r2)  # 1/r
        r = r2 * inv_r
        sr2 = sig2 * inv_r2
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        rc2 = sig2
        active = r2 < rc2
        not_self = j != i
        mask = active & not_self

        # Plain LJ energy (note: no WCA +eps shift). At r=sigma, U=0.
        u = 4.0 * eps * (sr12 - sr6)

        # Force-shifted energy so that U(rc)=0 and dU/dr(rc)=0.
        # With rc=sigma: U(rc)=0 and U'(rc)=-24 eps/sigma, so add (r-sigma)*24 eps/sigma.
        u_fs = u + (r - sig) * (24.0 * eps / sig)

        return u_fs * mask

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        return ("epsilon_eff",)


__all__ = ["WCAShifted"]
