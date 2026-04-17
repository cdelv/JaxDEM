# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions for generating generalized Thomson problem meshes in 2d/3d."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

def _sample_uniform_surface_points(key, nv, axes):
    """
    Sample points uniformly on a hyper-ellipsoid surface.
    """
    inv_axes = 1.0 / axes
    accept_scale = jnp.min(axes)

    accepted_chunks = []
    accepted_count = 0
    current_key = key

    while accepted_count < nv:
        n_candidates = max(32, 2 * (nv - accepted_count))
        current_key, sphere_key, accept_key = jax.random.split(current_key, 3)
        unit_points = jax.random.normal(sphere_key, shape=(n_candidates, axes.shape[0]))
        unit_points /= jnp.linalg.norm(unit_points, axis=-1, keepdims=True)
        weights = jnp.linalg.norm(unit_points * inv_axes, axis=-1)
        accept_prob = accept_scale * weights
        accepted_mask = jax.random.uniform(accept_key, shape=(n_candidates,)) < accept_prob
        accepted = unit_points[accepted_mask]

        accepted_chunks.append(accepted)
        accepted_count += accepted.shape[0]

    return jnp.concatenate(accepted_chunks, axis=0)[:nv] * axes


def random_points_on_hyper_ellipsoid(key, nv, N, dim, aspect_ratio=None, use_uniform_sampling=True):
    """
    Generate nv uniform random points on a dim-dimensional
    unit hyper-ellipsoid surface with a set aspect ratio, repeated N times.
    If using uniform sampling, expensive rejection sampling is used
    to ensure the points are uniform across the surface.
    This gives the exact result for hyper-ellipsoids, but is 
    not necessary for hyper-spheres.
    Scaled so that the longest axis is unit-length.
    """
    if aspect_ratio is None:
        axes = jnp.ones((dim,), dtype=jnp.float32)
    else:
        axes = jnp.asarray(aspect_ratio, dtype=jnp.float32)
        if axes.ndim == 0:
            axes = jnp.full((dim,), axes, dtype=jnp.float32)
        elif axes.shape != (dim,):
            raise ValueError(
                f"Expected aspect_ratio to have shape ({dim},), got {axes.shape}."
            )
        if jnp.any(axes <= 0):
            raise ValueError("Hyper-ellipsoid axes must be strictly positive.")
        axes = axes / jnp.max(axes)

    if jnp.all(axes == axes[0]).item() or use_uniform_sampling:
        points = jax.random.normal(key, shape=(N, nv, dim))
        points /= jnp.linalg.norm(points, axis=-1, keepdims=True)
        points *= axes
        return (points[0] if N == 1 else points), axes

    points = jnp.stack(
        [_sample_uniform_surface_points(batch_key, nv, axes) for batch_key in jax.random.split(key, N)]
    )
    return (points[0] if N == 1 else points), axes


def riesz_energy(pos, alpha):
    """Riesz energy kernel.  alpha=1 reduces to the Thomson problem.  alpha=\infty reduces to the packing problem"""
    r_ij = pos[:, None, :] - pos[None, :, :]
    # squared distances (no gradient issue here)
    d_sq = jnp.sum(r_ij**2, axis=-1)
    # fill diagonal with 1.0 BEFORE sqrt, so grad(sqrt(1.0)) = 0.5, not inf
    n = pos.shape[0]
    d_sq = d_sq.at[jnp.diag_indices(n)].set(1.0)
    d_ij = jnp.sqrt(d_sq)
    e_ij = 1.0 / d_ij ** alpha
    # zero out the diagonal so self-interactions don't contribute
    e_ij = e_ij.at[jnp.diag_indices(n)].set(0.0)
    return jnp.sum(jnp.triu(e_ij, k=1))

def project_to_tangent(grad, pos, aspect_ratio):
    """Remove the normal component of the gradient (project onto tangent plane of surface)."""
    normal = pos / aspect_ratio ** 2
    normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return grad - jnp.sum(grad * normal, axis=-1, keepdims=True) * normal

def retract_to_surface(pos, aspect_ratio):
    """Project point back onto the ellipsoid/ellipse surface."""
    u = pos / aspect_ratio
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
    return u * aspect_ratio

def minimize_on_hyper_ellipsoid(pos, axes, alpha, lr=0.01, steps=1000):
    """Minimize Riesz energy for points constrained to a hyper-ellipsoid surface."""
    axes = jnp.asarray(axes, dtype=pos.dtype)
    energy_grad = jax.grad(riesz_energy)

    if steps == 0:
        return pos, riesz_energy(pos, alpha)

    def step(pos, _):
        g = energy_grad(pos, alpha)
        g_tangent = project_to_tangent(g, pos, axes)
        pos = pos - lr * g_tangent
        pos = retract_to_surface(pos, axes)
        return pos, riesz_energy(pos, alpha)

    pos, _ = jax.lax.scan(step, pos, None, length=steps)
    return pos, riesz_energy(pos, alpha)


def generate_thomson_mesh(
    nv,
    N,
    dim,
    alpha=1.0,
    lr=0.01,
    steps=1000,
    aspect_ratio=None,
    use_uniform_sampling=True,
    batch_size=None,
    seed=None,
):
    """Generate and minimize charges constrained to a hyper-ellipsoid surface."""
    key = jax.random.PRNGKey(np.random.randint(0, 1e9) if seed is None else seed)
    pos, axes = random_points_on_hyper_ellipsoid(
        key,
        nv=nv,
        N=N,
        dim=dim,
        aspect_ratio=aspect_ratio,
        use_uniform_sampling=use_uniform_sampling,
    )
    surface_dim = dim - 1
    scaled_lr = lr / nv ** ((alpha + 2) / surface_dim)
    if N == 1:
        pos, energy = minimize_on_hyper_ellipsoid(
            pos, axes, alpha, lr=scaled_lr, steps=steps
        )
    else:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive.")

        minimize_fn = lambda x: minimize_on_hyper_ellipsoid(
            x, axes, alpha, lr=scaled_lr, steps=steps
        )
        if batch_size is None:
            pos, energy = jax.vmap(minimize_fn)(pos)
        else:
            pos, energy = jax.lax.map(minimize_fn, pos, batch_size=batch_size)
    if np.any(np.isnan(energy)):
        raise ValueError(f'Minimization failed on {np.mean(np.isnan(energy)) * 100:.2f}% of runs. Try lowering lr!')
    return pos, energy
