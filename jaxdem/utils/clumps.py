import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from functools import partial
from jaxdem.utils import Quaternion


@partial(jax.jit, static_argnames=("n", "dim"), inline=True)
def _generate_golden_lattice(n, dim=2):
    if dim == 2:
        phi = 1.32471795724475
    else:
        phi = 1.22074408460576
    exponents = 1.0 + jax.lax.iota(int, size=dim)
    alphas = 1.0 / jnp.power(phi, exponents)
    ids = 1.0 + jax.lax.iota(int, size=n)
    return (0.5 + jnp.outer(ids, alphas)) % 1.0


@partial(jax.jit, static_argnames=("n_samples",))
def compute_clump_properties(state, mat_table, n_samples=50_000):
    dim = state.dim
    clump_ids = jnp.arange(state.N)
    counts = jnp.bincount(state.ID, length=state.N)
    points_u = _generate_golden_lattice(n_samples, dim=state.dim)
    pos = state.pos

    def solve_monte_carlo(c_id):
        is_in_clump = state.ID == c_id

        # --- Bounding Box & Points ---
        inf = jnp.inf
        local_min = pos - state.rad[:, None]
        local_max = pos + state.rad[:, None]

        # Masking optimization: only consider particles in the clump for the bbox
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

        # --- Inertia ---
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

            # Ensure valid rotation (determinant +1)
            det = jnp.linalg.det(eigvecs)
            eigvecs = eigvecs.at[:, 2].multiply(jnp.sign(det))

            # Create Rotation object from the matrix
            rot = Rotation.from_matrix(eigvecs)

            # SciPy returns [x, y, z, w] (scalar last)
            q_xyzw = rot.as_quat()
            q_update = jnp.concatenate([q_xyzw[3:4], q_xyzw[:3]])

            return total_mass, com, eigvals, q_update

        else:
            I_scalar = jnp.sum(rho * r_sq) * vol_per_sample
            I_res = I_scalar.reshape(1)
            q_update = jnp.array([1.0, 0.0, 0.0, 0.0])

            return total_mass, com, I_res, q_update

    tm, cm, it, qt = jax.vmap(solve_monte_carlo)(clump_ids)
    is_clump = counts[state.ID] > 1

    new_mass = jnp.where(is_clump, tm[state.ID], state.mass)
    new_com = jnp.where(is_clump[:, None], cm[state.ID], state.pos_c)
    new_inertia = jnp.where(is_clump[:, None], it[state.ID], state.inertia)

    new_q_arr = jnp.where(
        is_clump[:, None],
        qt[state.ID],
        jnp.concatenate([state.q.w, state.q.xyz], axis=-1),
    )

    state.mass = new_mass
    state.pos_c = new_com
    state.inertia = new_inertia
    state.q = Quaternion(new_q_arr[..., 0:1], new_q_arr[..., 1:])
    state.pos_p = pos - state.pos_c

    return state
