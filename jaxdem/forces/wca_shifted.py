# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from . import ForceModel
from ..utils.linalg import norm, unit_and_norm

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("wca_shifted")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class WCAShifted(ForceModel):
    r"""Contact-start, force-shifted WCA/LJ repulsion.

    The interaction starts at contact:

    - cutoff at :math:`r_c = \sigma_{ij}` where :math:`\sigma_{ij} = R_i + R_j`
    - :math:`U(r_c) = 0`
    - :math:`F(r_c) = 0` (force-shifted; smooth turn-on at contact)

    The model reads the material-pair parameter ``epsilon_eff[mi, mj]``.
    """

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="WCAShifted.force")
    def force(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> tuple[jax.Array, jax.Array]:
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain._displacement(pos[i], pos[j], system)
        rhat, r = unit_and_norm(rij)
        safe_r = jnp.where(r == 0.0, 1.0, r)

        inv_r = 1.0 / safe_r
        sr = sig * inv_r
        sr2 = sr * sr
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        # cutoff at contact: r_c = sigma
        active = safe_r < sig
        not_self = j != i
        mask = active & not_self

        # LJ force magnitude along rhat: F(r) = 24 eps (2 sr12 - sr6) / r
        fmag = 24.0 * eps * inv_r * (2.0 * sr12 - sr6)
        # Force-shift so that F(rc) = 0. At rc = sigma: sr6=sr12=1 => F(rc) = 24 eps / sigma
        fmag_rc = 24.0 * eps / sig
        fmag_fs = fmag - fmag_rc

        f = (fmag_fs * mask)[..., None] * rhat
        t_shape = jnp.shape(j) + jnp.shape(state.torque[i])
        return f, jnp.zeros(t_shape, dtype=state.torque.dtype)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="WCAShifted.energy")
    def energy(
        i: int, j: int, pos: jax.Array, state: State, system: System
    ) -> jax.Array:
        mi, mj = state.mat_id[i], state.mat_id[j]
        eps = system.mat_table.epsilon_eff[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain._displacement(pos[i], pos[j], system)
        r = norm(rij)
        safe_r = jnp.where(r == 0.0, 1.0, r)

        inv_r = 1.0 / safe_r
        sr = sig * inv_r
        sr2 = sr * sr
        sr6 = sr2 * sr2 * sr2
        sr12 = sr6 * sr6

        active = safe_r < sig
        not_self = j != i
        mask = active & not_self

        # Plain LJ energy (note: no WCA +eps shift). At r=sigma, U=0.
        u = 4.0 * eps * (sr12 - sr6)

        # Force-shifted energy so that U(rc)=0 and dU/dr(rc)=0.
        # With rc=sigma: U(rc)=0 and U'(rc)=-24 eps/sigma, so add (r-sigma)*24 eps/sigma.
        u_fs = u + (safe_r - sig) * (24.0 * eps / sig)

        return u_fs * mask

    @property
    def required_material_properties(self) -> tuple[str, ...]:
        return ("epsilon_eff",)


__all__ = ["WCAShifted"]
