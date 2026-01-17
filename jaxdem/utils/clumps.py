import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from typing import Tuple, TYPE_CHECKING
from functools import partial

from . import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..materials import MaterialTable


@partial(jax.jit, static_argnames=("n", "dim"), inline=True)
def _generate_golden_lattice(n: int, dim: int = 2) -> jax.Array:
    if dim == 2:
        phi = 1.32471795724475
    else:
        phi = 1.22074408460576
    exponents = 1.0 + jax.lax.iota(int, size=dim)
    alphas = 1.0 / jnp.power(phi, exponents)
    ids = 1.0 + jax.lax.iota(int, size=n)
    return (0.5 + jnp.outer(ids, alphas)) % 1.0


@partial(jax.jit, static_argnames=("n_samples",))
def compute_clump_properties(
    state: "State", mat_table: "MaterialTable", n_samples: int = 50_000
) -> "State":
    dim = state.dim
    clump_ids = jnp.arange(state.N)
    counts = jnp.bincount(state.clump_ID, length=state.N)
    points_u = _generate_golden_lattice(n_samples, dim=state.dim)
    pos = state.pos

    def solve_monte_carlo(c_id: jax.Array) -> Tuple[jax.Array, ...]:
        is_in_clump = state.clump_ID == c_id

        # --- Bounding Box & Points ---
        inf = jnp.inf
        local_min = pos - state.rad[:, None]
        local_max = pos + state.rad[:, None]

        clump_min_b = jnp.min(jnp.where(is_in_clump[:, None], local_min, inf), axis=0)
        clump_max_b = jnp.max(jnp.where(is_in_clump[:, None], local_max, -inf), axis=0)

        box_vol = jnp.prod(clump_max_b - clump_min_b)
        points = clump_min_b + points_u * (clump_max_b - clump_min_b)

        # --- Filter Logic ---
        eff_rad = jnp.where(is_in_clump, state.rad, 0.0)
        eff_densities = jnp.where(is_in_clump, mat_table.density[state.mat_id], 0.0)

        diff = points[:, None, :] - pos[None, :, :]
        dists_sq = jnp.vecdot(diff, diff)
        inside_mask = dists_sq < jnp.square(eff_rad[None, :])

        densities_per_point = jnp.where(inside_mask, eff_densities[None, :], 0.0)
        rho = jnp.max(densities_per_point, axis=-1)

        # --- Mass & COM ---
        vol_per_sample = box_vol / n_samples
        total_mass = jnp.sum(rho) * vol_per_sample

        rho_r = points * rho[:, None]
        com = jnp.sum(rho_r, axis=0) * vol_per_sample / total_mass

        # --- Inertia & Orientation ---
        r_prime = points - com
        r_sq = jnp.sum(r_prime**2, axis=1)

        if dim == 3:
            term1 = jnp.sum(
                rho[:, None, None] * r_sq[:, None, None] * jnp.eye(3)[None, :, :],
                axis=0,
            )
            term2 = jnp.einsum("n,ni,nj->ij", rho, r_prime, r_prime)
            I_tensor = (term1 - term2) * vol_per_sample

            I_tensor = 0.5 * (I_tensor + I_tensor.T)
            eigvals, eigvecs = jnp.linalg.eigh(I_tensor)

            rot = Rotation.from_matrix(eigvecs)
            q_xyzw = rot.as_quat()
            q_update = jnp.concatenate([q_xyzw[3:4], q_xyzw[:3]])

            return total_mass, com, eigvals, q_update

        else:
            # 2D Case: Use Covariance Matrix to determine orientation
            Cov = jnp.einsum("n,ni,nj->ij", rho, r_prime, r_prime) * vol_per_sample
            eigvals_cov, eigvecs = jnp.linalg.eigh(Cov)

            # Convert 2D rotation matrix (eigvecs) to angle theta
            # Column 0 is the new X-axis
            theta = jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0])

            # Convert angle to Quaternion (rotation around Z)
            half_theta = theta / 2.0
            q_update = jnp.array([jnp.cos(half_theta), 0.0, 0.0, jnp.sin(half_theta)])

            # Scalar polar moment of inertia
            I_scalar = jnp.sum(rho * r_sq) * vol_per_sample
            I_res = I_scalar.reshape(1)

            return total_mass, com, I_res, q_update

    tm, cm, it, qt = jax.vmap(solve_monte_carlo)(clump_ids)
    is_clump = counts[state.clump_ID] > 1

    new_mass = jnp.where(is_clump, tm[state.clump_ID], state.mass)
    new_com = jnp.where(is_clump[:, None], cm[state.clump_ID], state.pos_c)
    new_inertia = jnp.where(is_clump[:, None], it[state.clump_ID], state.inertia)

    new_q_arr = jnp.where(
        is_clump[:, None],
        qt[state.clump_ID],
        jnp.concatenate([state.q.w, state.q.xyz], axis=-1),
    )

    state.mass = new_mass
    state.pos_c = new_com
    state.inertia = new_inertia
    state.q = Quaternion(new_q_arr[..., 0:1], new_q_arr[..., 1:])
    state.pos_p = state.q.rotate_back(state.q, pos - state.pos_c)

    return state
