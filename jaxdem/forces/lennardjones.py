from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Tuple

from . import ForceModel

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@ForceModel.register("lennardjones")
@partial(jax.tree_util.register_dataclass, drop_fields=["required_material_properties"])
@dataclass(slots=True)
class LennardJones(ForceModel):
    r"""
    Lennard-Jones (LJ) 12-6 interaction with a per-pair cutoff and energy shift.

    Uses material-pair parameter:
      - epsilon_eff[mi, mj]

    The length scale :math:`\sigma_{ij}` is derived from particle radii (like `spring.py`):

    .. math::
        \sigma_{ij} = R_i + R_j

    Potential (for :math:`r < r_c = 2.5 \sigma_{ij}`):

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

    # Common LJ cutoff (in units of sigma). Kept as a python float constant.
    RC_FACTOR: ClassVar[float] = 2.5

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="LennardJones.force")
    def force(
        i: int, j: int, pos: jax.Array, state: "State", system: "System"
    ) -> Tuple[jax.Array, jax.Array]:
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

        rc2 = (LennardJones.RC_FACTOR * LennardJones.RC_FACTOR) * sig2
        active = r2 < rc2
        not_self = j != i
        mask = active & not_self

        coeff = 24.0 * eps * inv_r2 * (2.0 * sr12 - sr6)
        f = (coeff[..., None] * rij) * mask[..., None]

        return f, jnp.zeros_like(state.angVel[i])

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="LennardJones.energy")
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

        rc2 = (LennardJones.RC_FACTOR * LennardJones.RC_FACTOR) * sig2
        active = r2 < rc2
        not_self = j != i
        mask = active & not_self

        # Shift so U(rc) = 0. Since rc = RC_FACTOR * sigma, (sigma/rc) is constant.
        inv_rc6 = (1.0 / LennardJones.RC_FACTOR) ** 6
        u_shift = 4.0 * eps * (inv_rc6 * inv_rc6 - inv_rc6)

        u = 4.0 * eps * (sr12 - sr6) - u_shift
        return u * mask

    @property
    def required_material_properties(self) -> Tuple[str, ...]:
        return ("epsilon_eff",)


__all__ = ["LennardJones"]
