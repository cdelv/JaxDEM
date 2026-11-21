# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Contact-start, force-shifted Weeks-Chandler-Andersen repulsion."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from . import ForceModel
from ..utils.linalg import norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


def _wca_shifted_force_coeff(
    n2: jax.Array, sig: jax.Array, eps: jax.Array
) -> jax.Array:
    """Scalar multiplier ``c`` such that ``F_ij = c * r_ij`` for force-shifted WCA."""
    zero = n2 == 0.0
    safe_n2 = jnp.where(zero, 1.0, n2)
    inv_r = jnp.where(zero, 0.0, jax.lax.rsqrt(safe_n2))
    safe_inv_r = jnp.where(zero, 1.0, inv_r)
    safe_r = safe_n2 * safe_inv_r

    sr2 = (sig * sig) * (safe_inv_r * safe_inv_r)
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6

    fmag = 24.0 * eps * safe_inv_r * (2.0 * sr12 - sr6)
    fmag_rc = 24.0 * eps / jnp.where(sig > 0.0, sig, 1.0)
    return jnp.where(safe_r < sig, (fmag - fmag_rc) * inv_r, 0.0)


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

    @property
    def has_pair_torque(self) -> bool:
        return False

    @property
    def supports_sorted_order(self) -> bool:
        return True

    def search_radii(self, state: State, system: System) -> jax.Array:
        """Conservative search extent for this law's finite interaction range."""
        return jnp.maximum(state._rad, (1.0) * state.rad)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="WCAShifted.force")
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
        eps_table = system.mat_table.epsilon_eff
        if eps_table.shape == (1, 1):
            eps = eps_table[0, 0]
        else:
            mi, mj = state.mat_id[i], state.mat_id[j]
            eps = eps_table[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain._displacement(pos[i], pos[j], system)
        n2 = norm2(rij)
        coeff = _wca_shifted_force_coeff(n2, sig, eps) * (j != i)
        f = coeff[..., None] * rij
        torque = jnp.zeros(
            (*f.shape[:-1], state.torque.shape[-1]), dtype=state.torque.dtype
        )
        return f, torque, history

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="WCAShifted.energy")
    def energy(
        i: int,
        j: int,
        pos: jax.Array,
        state: State,
        system: System,
        history: jax.Array,
    ) -> jax.Array:
        eps_table = system.mat_table.epsilon_eff
        if eps_table.shape == (1, 1):
            eps = eps_table[0, 0]
        else:
            mi, mj = state.mat_id[i], state.mat_id[j]
            eps = eps_table[mi, mj]
        sig = state.rad[i] + state.rad[j]

        rij = system.domain._displacement(pos[i], pos[j], system)
        n2 = norm2(rij)
        zero = n2 == 0.0
        safe_n2 = jnp.where(zero, 1.0, n2)
        safe_r = jnp.sqrt(safe_n2)

        inv_r2 = 1.0 / safe_n2
        sr2 = (sig * sig) * inv_r2
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
