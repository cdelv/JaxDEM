# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Defines the simulation State."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast, final

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .utils import linalg
from .utils.quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from .materials import MaterialTable

#: Diagonal regularization added to inertia computed from facet/mesh geometry,
#: as ``mass * INERTIA_REGULARIZATION`` (units of mass, so the bias is exactly
#: 1e-4 relative only for bodies of unit length scale). Degenerate geometry
#: (collinear or coplanar vertices) yields zero inertia about some axis, which
#: would make ``1/I`` in the rotation integrators infinite; this floor keeps
#: angular accelerations finite at the cost of a small bias in the affected
#: principal moment.
INERTIA_REGULARIZATION = 1e-4


@jax.jit(inline=True)
def _hypersphere_volume(rad: jax.Array, dim: int) -> jax.Array:
    """Volume of a ``dim``-dimensional ball of radius ``rad`` (area in 2D)."""
    return jnp.exp(
        0.5 * dim * jnp.log(jnp.pi)
        + dim * jnp.log(rad)
        - jax.scipy.special.gammaln(0.5 * dim + 1.0)
    )


@final
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class State:
    r"""Represents the complete simulation state for a system of N particles in 2D or 3D.

    Notes:
    ------
    `State` is designed to support various data layouts:

    - **Single snapshot:**
        `pos.shape = (N, dim)` for particle properties (e.g., `pos`, `vel`, `force`),
        and `(N,)` for scalar properties (e.g., `rad`, `mass`).
        In this case, `batch_size` is 1.

    - **Batched states:**
        `pos.shape = (B, N, dim)` for particle properties, and `(B, N)` for scalar properties.
        Here, `B` is the batch dimension (`batch_size = pos.shape[0]`).

    - **Trajectories of a single simulation:**
        `pos.shape = (T, N, dim)` for particle properties, and `(T, N)` for scalar properties.
        Here, `T` is the trajectory dimension.

    - **Trajectories of batched states:**
        `pos.shape = (T_1, T_2, ..., T_k, B, N, dim)` for particle properties,
        and `(T_1, T_2, ..., T_k, B, N)` for scalar properties.

        - The dimension immediately preceding `N` (i.e., `pos.shape[-3]`) is
          always interpreted as the **batch dimension (`B`)** — this is what
          :attr:`batch_size` returns.

        - All leading dimensions before it (`T_1, T_2, ... T_k`) are interpreted as **trajectory dimensions**
            and they are **flattened at save time** if there is more than 1 trajectory dimension.

    The class is `final` and cannot be subclassed.

    Example:
    --------
    Creating a simple 2D state for 4 particles:

    >>> import jaxdem as jdem
    >>> import jax.numpy as jnp
    >>> import jax
    >>>
    >>> positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> state = jdem.State.create(pos=positions)
    >>>
    >>> print(f"Number of particles (N): {state.N}")
    >>> print(f"Spatial dimension (dim): {state.dim}")
    >>> print(f"Positions: {state.pos}")

    Creating a batched state:

    >>> batched_state = jax.vmap(lambda _: jdem.State.create(pos=positions))(jnp.arange(10))
    >>>
    >>> print(f"Batch size: {batched_state.batch_size}")  # 10
    >>> print(f"Positions shape: {batched_state.pos.shape}")

    """

    pos_c: jax.Array
    """
    Array of particle center of mass positions. Shape is `(..., N, dim)`.

    For rigid clump members this is the **clump's** center of mass, identical
    on every member; the sphere's own center is the computed property ``pos``.
    """

    pos_p: jax.Array
    """
    Vector relative to the center of mass (pos_p = pos - pos_c) in the principal reference frame. This field should be constant. Shape is `(..., N, dim)`.
    """

    vel: jax.Array
    """
    Array of particle center of mass velocities. Shape is `(..., N, dim)`.

    For rigid clump members this is the clump COM velocity, identical on every member.
    """

    force: jax.Array
    """
    Array of particle forces. Shape is `(..., N, dim)`.

    For rigid clump members, every member stores the **total** aggregated clump force.
    """

    q: Quaternion
    """
    Quaternion representing the orientation of the particle.

    For rigid clump members this is the clump orientation, identical on every member.
    """

    ang_vel: jax.Array
    """
    Array of particle center of mass angular velocities. Shape is `(..., N, 1 | 3)` depending on 2D or 3D simulations.

    For rigid clump members this is the clump angular velocity, identical on every member.
    """

    torque: jax.Array
    """
    Array of particle torques. Shape is `(..., N, 1 | 3)` depending on 2D or 3D simulations.

    For rigid clump members, every member stores the **total** aggregated clump torque about the clump COM.
    """

    rad: jax.Array
    """
    Array of particle radii. Shape is `(..., N)`.
    """

    _rad: jax.Array
    """
    Array of particle search radii. Shape is `(..., N)`.
    """

    volume: jax.Array
    """
    Array of particle volumes (or areas if 2D). Shape is `(..., N)`.

    Per-sphere by default; when an explicit clump volume is passed to
    :py:meth:`State.add_clump`, it is replicated on every member.
    """

    mass: jax.Array
    """
    Array of particle masses. Shape is `(..., N)`.

    For rigid clump members, every member stores the **total** clump mass, not a per-member share.
    """

    inertia: jax.Array
    """
    Inertia tensor in the principal axis frame `(..., N, 1 | 3)` depending on 2D or 3D simulations.

    For rigid clump members, every member stores the **total** clump inertia tensor.
    """

    clump_id: jax.Array
    """
    Array of clump identifiers. Bodies with the same clump_id are treated as part of the same rigid body. Shape is `(..., N)`.
    IDs need to be between 0 and N.
    """

    bond_id: jax.Array
    """
    Array of connected neighbors for contact filtering. For each particle, it stores the unique_id values
    of the neighbor particles it is connected to. Interactions between connected particles are disabled.
    Shape is `(..., N, max_num_neighbors)`. Empty slots/non-connections are padded with -1.
    """

    unique_id: jax.Array
    """
    Array of unique particle identifiers. No ID should be repeated. Shape is `(..., N)`. IDs need to be between 0 and N.
    """

    mat_id: jax.Array
    """
    Array of material IDs for each particle. Shape is `(..., N)`.
    """

    species_id: jax.Array
    """
    Array of species IDs for each particle. Shape is `(..., N)`.
    """

    fixed: jax.Array
    """
    Boolean array indicating if a particle is fixed (immobile). Shape is `(..., N)`.

    Identical on every member of a rigid clump.
    """

    facet_id: jax.Array = field(default_factory=lambda: jnp.zeros((0,), dtype=int))
    """
    Array of facet identifiers. Shape is `(..., N)`.
    """

    facet_vertices: jax.Array = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=int)
    )
    """
    Array of vertex unique IDs for each facet. Shape is `(..., N, dim)`.
    """

    _pos_p_rot: jax.Array = field(default_factory=lambda: jnp.zeros((0, 0)))
    """
    Rotated pos_p (R(q) @ pos_p).
    """

    def __post_init__(self) -> None:
        # Bypass recalculation in dataclasses.replace if q or pos_p are not modified
        if (
            hasattr(self, "_pos_p_rot")
            and self._pos_p_rot is not None
            and hasattr(self._pos_p_rot, "shape")
            and hasattr(self.pos_p, "shape")
            and self._pos_p_rot.shape == self.pos_p.shape
        ):
            import sys

            frame: Any = sys._getframe(1)
            is_replace = False
            q_or_pos_p_changed = False
            while frame is not None:
                if (
                    frame.f_code.co_name == "replace"
                    and "dataclasses" in frame.f_code.co_filename
                ):
                    is_replace = True
                    changes = frame.f_locals.get("changes", {})
                    if "q" in changes or "pos_p" in changes:
                        q_or_pos_p_changed = True
                    break
                frame = frame.f_back
            if is_replace and not q_or_pos_p_changed:
                return

        try:
            computed_rot = self.q.rotate(self.q, self.pos_p)
            object.__setattr__(self, "_pos_p_rot", computed_rot)
        except (AttributeError, TypeError, IndexError, ValueError):
            pass

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in ("q", "pos_p"):
            try:
                q = self.q
                pos_p = self.pos_p
                computed_rot = q.rotate(q, pos_p)
                object.__setattr__(self, "_pos_p_rot", computed_rot)
            except (AttributeError, TypeError, IndexError, ValueError):
                pass

    @property
    def N(self) -> int:
        """Number of particles in the state."""
        return self.pos_c.shape[-2]

    @property
    def dim(self) -> int:
        """Spatial dimension of the simulation."""
        return self.pos_c.shape[-1]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the position array ``pos_c``, e.g. ``(N, dim)`` or ``(B, N, dim)``."""
        return self.pos_c.shape

    @property
    def batch_size(self) -> int:
        """Return the batch size of the state."""
        return 1 if self.pos_c.ndim < 3 else self.pos_c.shape[-3]

    @property
    @partial(jax.named_call, name="State.pos")
    def pos(self) -> jax.Array:
        """Returns the position of each sphere in the state.
        ``pos_c`` is the center of mass and ``pos_p`` is the vector relative to
        the center of mass in the principal reference frame, such that
        ``pos = pos_c + R(q) @ pos_p`` where ``R(q)`` rotates ``pos_p`` to the
        lab frame.
        """
        return self.pos_c + self._pos_p_rot

    @property
    @partial(jax.named_call, name="State.is_valid")
    def is_valid(self) -> bool:
        """Check if the internal representation of the State is consistent.

        Verifies that:

            - The spatial dimension (`dim`) is either 2 or 3.

            - All position-like arrays (`pos_c`, `pos_p`, `vel`, `force`) have the same shape.

            - All angular-like arrays (`ang_vel`, `torque`, `inertia`) have the same shape.

            - All scalar-per-particle arrays (`rad`, `mass`, `clump_id`, `bond_id`, `unique_id`, `mat_id`, `species_id`, `fixed`) have a shape consistent with `pos.shape[:-1]`.

        Returns
        -------
        bool
            `True` if the state is internally consistent, `False` otherwise.

        """
        if not ((self.dim in (2, 3)) or (self.N == 0 and self.dim == 0)):
            return False

        for name in (
            "pos_c",
            "pos_p",
            "_pos_p_rot",
            "vel",
            "force",
            "facet_vertices",
        ):
            arr = getattr(self, name)
            if self.pos_c.shape != arr.shape:
                return False

        ang_dim = 1 if self.dim == 2 else 3
        expected_ang_shape = (*self.pos_c.shape[:-1], ang_dim)
        for name in ("ang_vel", "torque", "inertia"):
            arr = getattr(self, name)
            if arr.shape != expected_ang_shape:
                return False

        for name in (
            "rad",
            "_rad",
            "mass",
            "volume",
            "clump_id",
            "unique_id",
            "mat_id",
            "species_id",
            "fixed",
            "facet_id",
        ):
            arr = getattr(self, name)
            if self.pos_c.shape[:-1] != arr.shape:
                return False

        if self.pos_c.shape[:-1] != self.bond_id.shape[:-1]:
            return False

        return True

    @staticmethod
    @partial(jax.named_call, name="State.create")
    def create(
        pos: ArrayLike | None = None,
        *,
        dim: int | None = None,
        pos_p: ArrayLike | None = None,
        vel: ArrayLike | None = None,
        force: ArrayLike | None = None,
        q: Quaternion | None | ArrayLike | None = None,
        ang_vel: ArrayLike | None = None,
        torque: ArrayLike | None = None,
        rad: ArrayLike | None = None,
        _rad: ArrayLike | None = None,
        volume: ArrayLike | None = None,
        mass: ArrayLike | None = None,
        inertia: ArrayLike | None = None,
        clump_id: ArrayLike | None = None,
        bond_id: ArrayLike | Sequence[Sequence[int]] | None = None,
        mat_id: ArrayLike | None = None,
        species_id: ArrayLike | None = None,
        fixed: ArrayLike | None = None,
        facet_id: ArrayLike | None = None,
        facet_vertices: ArrayLike | None = None,
        mat_table: MaterialTable | None = None,
    ) -> State:
        r"""Factory method to create a new :class:`State` instance.

        This method handles default values and ensures consistent array shapes
        for all state attributes.

        Parameters
        ----------
        pos : jax.typing.ArrayLike or None, optional
            Array of particle center of mass positions, equivalent to state.pos_c.
            Expected shape: `(..., N, dim)`.
            If `None`, an empty state is created. With `dim=None`, shape is
            `(0, 0)` (wildcard empty); with `dim=2|3`, shape is `(0, dim)`.
        dim : int or None, optional
            Spatial dimension used only when `pos is None` to create an empty
            state. Must be 2 or 3. If `None`, an empty state is created with
            wildcard dimension semantics (it can merge with 2D or 3D states).
        pos_p : jax.typing.ArrayLike
            Vector relative to the center of mass (pos_p = pos - pos_c) in the
            principal reference frame. This field should be constant. Shape is `(..., N, dim)`.
        vel : jax.typing.ArrayLike or None, optional
            Initial velocities of particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, dim)`.
        force : jax.typing.ArrayLike or None, optional
            Initial forces on particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, dim)`.
        q : Quaternion or array-like, optional
            Initial particle orientations. If `None`, defaults to identity quaternions.
            Accepted shapes: quaternion objects or arrays of shape `(..., N, 4)` with
            components ordered as `(w, x, y, z)`.
        ang_vel : jax.typing.ArrayLike or None, optional
            Initial angular velocities of particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, 1)` in 2D or `(..., N, 3)` in 3D.
        torque : jax.typing.ArrayLike or None, optional
            Initial torques on particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, 1)` in 2D or `(..., N, 3)` in 3D.
        rad : jax.typing.ArrayLike or None, optional
            Radii of particles. If `None`, defaults to ones.
            Expected shape: `(..., N)`.
        _rad : jax.typing.ArrayLike or None, optional
            Broad-phase search radii of particles, used by cell-list/neighbor
            colliders to size the contact-detection box. If `None`, defaults
            to `rad`. Expected shape: `(..., N)`.
        volume : jax.typing.ArrayLike or None, optional
            Volume of particles (or area in 2D). If `None`, defaults to hypersphere volumes of the radii.
            Expected shape: `(..., N)`.
        mass : jax.typing.ArrayLike or None, optional
            Masses of particles. If `None`, defaults to ones. Ignored when
            `mat_table` is provided.
            Expected shape: `(..., N)`.
        inertia : jax.typing.ArrayLike or None, optional
            Moments of inertia in the principal axes frame. If `None`, defaults to
            solid disks (2D) or spheres (3D).
            Expected shape: `(..., N, 1)` in 2D or `(..., N, 3)` in 3D.
        clump_id : jax.typing.ArrayLike or None, optional
            Unique identifiers for clumps. If `None`, defaults to
            :func:`jnp.arange`. Expected shape: `(..., N)`.
        bond_id : jax.typing.ArrayLike or None, optional
            List of connected unique_id values for each particle, storing the unique_ids of
            the particles it is connected to. Can be passed as a nested list
            (potentially with uneven lengths), or a 2D array.
            Connections are automatically symmetrized and padded with -1.
            If `None`, defaults to no connections (shape `(..., N, 1)` filled with -1).
        mat_id : jax.typing.ArrayLike or None, optional
            Material IDs for particles. If `None`, defaults to zeros.
            Expected shape: `(..., N)`.
        species_id : jax.typing.ArrayLike or None, optional
            Species IDs for particles. If `None`, defaults to zeros.
            Expected shape: `(..., N)`.
        fixed : jax.typing.ArrayLike or None, optional
            Boolean array indicating fixed particles. If `None`, defaults to all `False`.
            Expected shape: `(..., N)`.
        facet_id : jax.typing.ArrayLike or None, optional
            Facet identifiers; `-1` marks particles that are not facet
            vertices. If `None`, defaults to all `-1`.
            Expected shape: `(..., N)`.
        facet_vertices : jax.typing.ArrayLike or None, optional
            Vertex `unique_id` values for the facet each particle belongs to;
            `-1` marks non-facet particles. If `None`, defaults to all `-1`.
            Expected shape: `(..., N, dim)`.
        mat_table : MaterialTable or None, optional
            Optional material table providing per-material densities. When provided,
            the `mass` argument is ignored and particle masses are computed from
            `density` and particle volume.

        Returns
        -------
        State
            A new `State` instance with all attributes correctly initialized and shaped.

        Raises
        ------
        ValueError
            If the created `State` is not valid.

        Example
        -------
        Creating a 3D state for 5 particles:

        >>> import jaxdem as jdem
        >>> import jax.numpy as jnp
        >>>
        >>> my_pos = jnp.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [1.,1.,1.]])
        >>> my_rad = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])
        >>> my_mass = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        >>>
        >>> state_5_particles = jdem.State.create(pos=my_pos, rad=my_rad, mass=my_mass)
        >>> print(f"Shape of positions: {state_5_particles.pos.shape}")
        >>> print(f"Radii: {state_5_particles.rad}")

        """
        if pos is None:
            inferred_shape: tuple[int, ...] | None = None  # (..., N)
            inferred_dim: int | None = dim

            def _update_shape(shape: tuple[int, ...], name: str) -> None:
                nonlocal inferred_shape
                if inferred_shape is None:
                    inferred_shape = shape
                elif inferred_shape != shape:
                    raise ValueError(
                        f"Inconsistent particle shapes while inferring state from arguments: "
                        f"expected {inferred_shape}, got {shape} for `{name}`."
                    )

            def _update_dim(d: int, name: str) -> None:
                nonlocal inferred_dim
                if d not in (2, 3):
                    raise ValueError(
                        f"`{name}` implies invalid dim={d}. Expected 2 or 3."
                    )
                if inferred_dim is None:
                    inferred_dim = d
                elif inferred_dim != d:
                    raise ValueError(
                        f"Conflicting dimensions while inferring state from arguments: "
                        f"dim={inferred_dim} vs dim={d} from `{name}`."
                    )

            vector_fields = {
                "pos_p": pos_p,
                "vel": vel,
                "force": force,
                "facet_vertices": facet_vertices,
            }
            for name, arr in vector_fields.items():
                if arr is None:
                    continue
                a = jnp.asarray(arr)
                if a.ndim < 2:
                    raise ValueError(
                        f"`{name}` must have shape (..., N, dim). Got shape={a.shape}."
                    )
                _update_shape(tuple(a.shape[:-1]), name)
                _update_dim(int(a.shape[-1]), name)

            angular_fields = {
                "ang_vel": ang_vel,
                "torque": torque,
                "inertia": inertia,
            }
            for name, arr in angular_fields.items():
                if arr is None:
                    continue
                a = jnp.asarray(arr)
                if a.ndim < 2:
                    raise ValueError(
                        f"`{name}` must have shape (..., N, 1|3). Got shape={a.shape}."
                    )
                _update_shape(tuple(a.shape[:-1]), name)
                last = int(a.shape[-1])
                if last == 1:
                    _update_dim(2, name)
                elif last == 3:
                    _update_dim(3, name)
                else:
                    raise ValueError(
                        f"`{name}` last axis must be 1 (2D) or 3 (3D). Got {last}."
                    )

            scalar_fields = {
                "rad": rad,
                "_rad": _rad,
                "volume": volume,
                "mass": mass,
                "clump_id": clump_id,
                "mat_id": mat_id,
                "species_id": species_id,
                "fixed": fixed,
                "facet_id": facet_id,
            }
            for name, arr in scalar_fields.items():
                if arr is None:
                    continue
                a = jnp.asarray(arr)
                if a.ndim < 1:
                    raise ValueError(
                        f"`{name}` must have shape (..., N). Got shape={a.shape}."
                    )
                _update_shape(tuple(a.shape), name)

            if bond_id is not None:
                try:
                    a = jnp.asarray(bond_id)
                    if a.ndim >= 1:
                        _update_shape(tuple(a.shape[:-1]), "bond_id")
                except (ValueError, TypeError):
                    pass

            if q is not None:
                if isinstance(q, Quaternion):
                    if q.w.shape[-1] != 1 or q.xyz.shape[-1] != 3:
                        raise ValueError(
                            "Quaternion fields must have w.shape[-1]==1 and xyz.shape[-1]==3."
                        )
                    _update_shape(tuple(q.w.shape[:-1]), "q")
                    if q.xyz.shape[:-1] != q.w.shape[:-1]:
                        raise ValueError(
                            "Quaternion `w` and `xyz` leading shapes must match."
                        )
                else:
                    q_arr = jnp.asarray(q)
                    if q_arr.ndim < 2 or q_arr.shape[-1] != 4:
                        raise ValueError(
                            f"`q` array must have shape (..., N, 4). Got shape={q_arr.shape}."
                        )
                    _update_shape(tuple(q_arr.shape[:-1]), "q")

            if inferred_shape is None:
                if inferred_dim is None:
                    pos_c = jnp.zeros((0, 0), dtype=float)
                else:
                    if inferred_dim not in (2, 3):
                        raise ValueError(
                            "State.create(..., pos=None) requires dim in (2, 3) or None. "
                            f"Got dim={inferred_dim}."
                        )
                    pos_c = jnp.zeros((0, inferred_dim), dtype=float)
            else:
                if inferred_dim is None:
                    inferred_dim = 3
                if inferred_dim not in (2, 3):
                    raise ValueError(
                        f"Invalid inferred dim={inferred_dim}. Expected 2 or 3."
                    )
                pos_c = jnp.zeros((*inferred_shape, inferred_dim), dtype=float)
        else:
            pos_c = jnp.asarray(pos, dtype=float)

        N = pos_c.shape[-2]
        dim = pos_c.shape[-1]
        ang_dim = 1 if dim == 2 else 3
        ang_shape = (*pos_c.shape[:-1], ang_dim)

        pos_p = (
            jnp.zeros_like(pos_c, dtype=float)
            if pos_p is None
            else jnp.asarray(pos_p, dtype=float)
        )
        vel = (
            jnp.zeros_like(pos_c, dtype=float)
            if vel is None
            else jnp.asarray(vel, dtype=float)
        )
        force = (
            jnp.zeros_like(pos_c, dtype=float)
            if force is None
            else jnp.asarray(force, dtype=float)
        )

        if q is None:
            q_quat = Quaternion.create(
                jnp.ones((*pos_c.shape[:-1], 1), dtype=float),
                jnp.zeros((*pos_c.shape[:-1], 3), dtype=float),
            )
        elif not isinstance(q, Quaternion):
            # If it's ArrayLike, assume (..., 4) with [w, x, y, z]
            q_arr = jnp.asarray(q, dtype=float)
            q_quat = Quaternion.create(
                w=q_arr[..., 0:1],
                xyz=q_arr[..., 1:],
            )
        else:
            q_quat = q

        ang_vel = (
            jnp.zeros(ang_shape, dtype=float)
            if ang_vel is None
            else jnp.asarray(ang_vel, dtype=float)
        )
        torque = (
            jnp.zeros_like(ang_vel, dtype=float)
            if torque is None
            else jnp.asarray(torque, dtype=float)
        )

        rad = (
            jnp.ones(pos_c.shape[:-1], dtype=float)
            if rad is None
            else jnp.asarray(rad, dtype=float)
        )
        volume = (
            _hypersphere_volume(rad, dim)
            if volume is None
            else jnp.asarray(volume, dtype=float)
        )

        clump_id = (
            jnp.broadcast_to(jnp.arange(N, dtype=int), pos_c.shape[:-1])
            if clump_id is None
            else jnp.asarray(clump_id, dtype=int)
        )
        if bond_id is None:
            bond_id = jnp.full((*pos_c.shape[:-1], 1), -1, dtype=int)
        else:
            import numpy as np

            n_lead = len(pos_c.shape[:-1])  # number of (..., N) axes

            # Convert input (array, JAX array, or nested lists with uneven
            # lengths) to a (..., N, W) numpy int array padded with -1.
            # A single host transfer at most; no per-element device syncs.
            conn_np: Any
            try:
                conn_np = np.asarray(bond_id, dtype=int)
                if conn_np.ndim == n_lead:
                    # One uid per particle: (..., N) -> (..., N, 1)
                    conn_np = conn_np[..., None]
            except (ValueError, TypeError):
                # Nested python lists with uneven lengths (unbatched input).
                rows: list[list[int]] = []
                for item in list(cast(Any, bond_id))[:N]:
                    if hasattr(item, "__iter__") and not isinstance(item, str):
                        vals = np.asarray(item).ravel().tolist()
                    else:
                        vals = [item]
                    rows.append([int(v) for v in vals if v is not None])
                width = max(max((len(r) for r in rows), default=0), 1)
                conn_np = np.full((N, width), -1, dtype=int)
                for i, r in enumerate(rows):
                    conn_np[i, : len(r)] = r

            def _symmetrize(conn: Any) -> Any:
                # conn: (N, W) int array of neighbor uids, -1 padded.
                w = max(conn.shape[-1], 1)
                rows_idx = np.repeat(np.arange(N), conn.shape[-1])
                cols = conn.reshape(-1)
                valid_mask = (cols >= 0) & (cols < N)
                i_idx = rows_idx[valid_mask]
                j_idx = cols[valid_mask]
                if i_idx.size == 0:
                    return np.full((N, w), -1, dtype=int)
                pairs = np.unique(
                    np.stack(
                        (
                            np.concatenate((i_idx, j_idx)),
                            np.concatenate((j_idx, i_idx)),
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
                counts = np.bincount(pairs[:, 0], minlength=N)
                max_deg = max(int(counts.max()), w)
                out = np.full((N, max_deg), -1, dtype=int)
                starts = np.cumsum(counts) - counts
                slot = np.arange(pairs.shape[0]) - starts[pairs[:, 0]]
                out[pairs[:, 0], slot] = pairs[:, 1]
                return out

            if conn_np.ndim == 2:
                bond_np = _symmetrize(conn_np)
            else:
                # Batched: symmetrize each (N, W) slice independently.
                flat = conn_np.reshape((-1, *conn_np.shape[-2:]))
                outs = [_symmetrize(c) for c in flat]
                max_deg = max(o.shape[-1] for o in outs)
                outs = [
                    np.pad(
                        o,
                        ((0, 0), (0, max_deg - o.shape[-1])),
                        constant_values=-1,
                    )
                    for o in outs
                ]
                bond_np = np.stack(outs).reshape((*conn_np.shape[:-2], N, max_deg))
            bond_id = jnp.asarray(bond_np, dtype=int)

        mat_id = (
            jnp.zeros(pos_c.shape[:-1], dtype=int)
            if mat_id is None
            else jnp.asarray(mat_id, dtype=int)
        )
        species_id = (
            jnp.zeros(pos_c.shape[:-1], dtype=int)
            if species_id is None
            else jnp.asarray(species_id, dtype=int)
        )
        fixed = (
            jnp.zeros(pos_c.shape[:-1], dtype=bool)
            if fixed is None
            else jnp.asarray(fixed, dtype=bool)
        )
        facet_id = (
            jnp.full(pos_c.shape[:-1], -1, dtype=int)
            if facet_id is None
            else jnp.asarray(facet_id, dtype=int)
        )
        facet_vertices = (
            jnp.full(pos_c.shape, -1, dtype=int)
            if facet_vertices is None
            else jnp.asarray(facet_vertices, dtype=int)
        )

        if mat_table is not None:
            density = mat_table.density[mat_id]
            mass = density * volume
        else:
            mass = (
                jnp.ones(pos_c.shape[:-1], dtype=float)
                if mass is None
                else jnp.asarray(mass, dtype=float)
            )

        coeff = 0.5 if dim == 2 else 0.4
        inertia_scalar = coeff * mass * rad**2
        inertia = (
            inertia_scalar[..., None] * jnp.ones_like(ang_vel, dtype=float)
            if inertia is None
            else jnp.asarray(inertia, dtype=float)
        )

        if N > 0:
            # Relabel clump ids into [0, N) independently per batch: a global
            # unique over a batched (..., N) array can see more than N distinct
            # ids and would corrupt the labels.
            def _relabel(ids: jax.Array) -> jax.Array:
                return jnp.unique(ids, return_inverse=True, size=N)[1]

            flat_ids = clump_id.reshape((-1, N))
            clump_id = jax.vmap(_relabel)(flat_ids).reshape(clump_id.shape)

        pos_p_rot = q_quat.rotate(q_quat, pos_p)
        # Copy so rad and _rad never share a buffer: aliased leaves break
        # argument donation in jitted entry points (XLA rejects donating the
        # same buffer twice).
        _rad = jnp.copy(rad) if _rad is None else jnp.asarray(_rad, dtype=float)

        state = State(
            pos_c=pos_c,
            pos_p=pos_p,
            vel=vel,
            force=force,
            q=q_quat,
            ang_vel=ang_vel,
            torque=torque,
            rad=rad,
            _rad=_rad,
            volume=volume,
            mass=mass,
            inertia=inertia,
            clump_id=jnp.asarray(clump_id),
            bond_id=jnp.asarray(bond_id),
            unique_id=jnp.broadcast_to(jnp.arange(N, dtype=int), pos_c.shape[:-1]),
            facet_id=facet_id,
            facet_vertices=facet_vertices,
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
            _pos_p_rot=pos_p_rot,
        )

        if not state.is_valid:
            raise ValueError(f"State is not valid, state={state}")

        return state

    @staticmethod
    @partial(jax.named_call, name="State.merge")
    def merge(state1: State, state2: State | Sequence[State]) -> State:
        r"""Merges multiple :class:`State` instances into a single new :class:`State`.

        This method concatenates the particles from the provided state(s) onto `state1`.
        Particle clump_ids, bond_ids, and unique_ids are shifted to ensure
        uniqueness across the merged system.

        Parameters
        ----------
        state1 : State
            The first `State` instance. Its particles will appear first in the merged state.
        state2 : State or Sequence[State]
            The second `State` or a list/tuple of `State` instances to append.

        Returns
        -------
        State
            A new `State` instance containing all particles from both input states.

        Raises
        ------
        AssertionError
            If either input state is invalid, or if there is a mismatch in
            spatial dimension (`dim`) or batch size (`batch_size`).
        ValueError
            If the resulting merged state is somehow invalid.

        Example
        -------
        >>> import jaxdem as jdem
        >>> import jax.numpy as jnp
        >>>
        >>> state_a = jdem.State.create(pos=jnp.array([[0.0, 0.0], [1.0, 1.0]]), clump_id=jnp.array([0, 1]))
        >>> state_b = jdem.State.create(pos=jnp.array([[2.0, 2.0], [3.0, 3.0]]), clump_id=jnp.array([0, 1]))
        >>> merged_state = jdem.State.merge(state_a, [state_b, state_b, state_b])
        >>> merged_state = jdem.State.merge(state_a, state_b)
        >>>
        >>> print(f"Merged state N: {merged_state.N}")  # Expected: 4
        >>> print(f"Merged state positions:\\n{merged_state.pos}")
        >>> print(f"Merged state clump_ids: {merged_state.clump_id}")  # Expected: [0, 1, 2, 3]

        """
        states_to_merge = [state2] if isinstance(state2, State) else list(state2)
        all_states = [state1, *states_to_merge]

        for s in all_states:
            assert s.is_valid, "Invalid state detected"

        non_empty = [s for s in all_states if s.N > 0]

        if not non_empty:
            # Validate that any non-wildcard empty states agree on dim/batch.
            ref = next((s for s in all_states if s.dim != 0), None)
            for s in all_states:
                if ref is not None:
                    if s.dim != 0:
                        assert s.dim == ref.dim, "Dimension mismatch"
                    assert s.batch_size == ref.batch_size, "Batch size mismatch"
            return all_states[-1]

        ref = non_empty[0]
        pos_ndim = ref.pos_c.ndim
        for s in all_states:
            if s.N == 0:
                if s.dim != 0:
                    assert s.dim == ref.dim, "Dimension mismatch"
                assert s.batch_size == ref.batch_size, "Batch size mismatch"
            else:
                assert s.dim == ref.dim, "Dimension mismatch"
                assert s.batch_size == ref.batch_size, "Batch size mismatch"

        if len(non_empty) == 1:
            return non_empty[0]

        # Shift IDs of each state by the cumulative offsets of the preceding states.
        c_offset = jnp.asarray(0, dtype=int)
        u_offset = jnp.asarray(0, dtype=int)
        f_offset = jnp.asarray(0, dtype=int)
        shifted: list[State] = []
        for s in non_empty:
            shifted.append(
                dataclasses.replace(
                    s,
                    clump_id=s.clump_id + c_offset,
                    bond_id=jnp.where(s.bond_id != -1, s.bond_id + u_offset, -1),
                    unique_id=s.unique_id + u_offset,
                    facet_id=jnp.where(s.facet_id != -1, s.facet_id + f_offset, -1),
                    facet_vertices=jnp.where(
                        s.facet_vertices != -1, s.facet_vertices + u_offset, -1
                    ),
                )
            )
            c_offset = c_offset + jnp.max(s.clump_id) + 1
            u_offset = u_offset + jnp.max(s.unique_id) + 1
            f_offset = f_offset + jnp.maximum(0, jnp.max(s.facet_id) + 1)

        # Single concatenation pass per leaf.
        def cat(*leaves: jax.Array) -> jax.Array:
            # If we are merging bond_id (or any 2D/3D array where last dims mismatch), pad with -1
            if leaves[0].ndim > 1 and len({a.shape[-1] for a in leaves}) > 1:
                max_w = max(a.shape[-1] for a in leaves)
                leaves = tuple(
                    jnp.pad(
                        a,
                        ((0, 0),) * (a.ndim - 1) + ((0, max_w - a.shape[-1]),),
                        constant_values=-1,
                    )
                    for a in leaves
                )
            # Particles are at -2 for vector fields, -1 for scalars
            axis = -2 if leaves[0].ndim == pos_ndim else -1
            return jnp.concatenate(leaves, axis=axis)

        current_state = jax.tree.map(cat, shifted[0], *shifted[1:])

        if not current_state.is_valid:
            raise ValueError("Merged state is not valid")

        return current_state

    @staticmethod
    @partial(jax.named_call, name="State.add")
    def add(
        state: State,
        pos: ArrayLike,
        *,
        pos_p: ArrayLike | None = None,
        vel: ArrayLike | None = None,
        force: ArrayLike | None = None,
        q: Quaternion | None | ArrayLike | None = None,
        ang_vel: ArrayLike | None = None,
        torque: ArrayLike | None = None,
        rad: ArrayLike | None = None,
        _rad: ArrayLike | None = None,
        volume: ArrayLike | None = None,
        mass: ArrayLike | None = None,
        inertia: ArrayLike | None = None,
        clump_id: ArrayLike | None = None,
        bond_id: ArrayLike | None = None,
        mat_id: ArrayLike | None = None,
        species_id: ArrayLike | None = None,
        fixed: ArrayLike | None = None,
        mat_table: MaterialTable | None = None,
    ) -> State:
        """Adds new particles to an existing :class:`State` instance, returning a new `State`.

        Parameters
        ----------
        state : State
            The existing `State` to which particles will be added.
        pos : jax.typing.ArrayLike
            Array of particle center of mass positions, equivalent to state.pos_c.
            Expected shape: `(..., N, dim)`.
        pos_p : jax.typing.ArrayLike
            Vector relative to the center of mass (pos_p = pos - pos_c) in the
            principal reference frame. This field should be constant. Shape is `(..., N, dim)`.
        vel : jax.typing.ArrayLike or None, optional
            Velocities of the new particle(s). Defaults to zeros.
        force : jax.typing.ArrayLike or None, optional
            Forces of the new particle(s). Defaults to zeros.
        q : Quaternion or array-like, optional
            Initial orientations of the new particle(s). Defaults to identity quaternions.
        ang_vel : jax.typing.ArrayLike or None, optional
            Angular velocities of the new particle(s). Defaults to zeros.
        torque : jax.typing.ArrayLike or None, optional
            Torques of the new particle(s). Defaults to zeros.
        rad : jax.typing.ArrayLike or None, optional
            Radii of the new particle(s). Defaults to ones.
        _rad : jax.typing.ArrayLike or None, optional
            Broad-phase search radii of the new particle(s), used by
            cell-list/neighbor colliders to size the contact-detection box.
            Defaults to `rad`.
        volume : jax.typing.ArrayLike or None, optional
            Volume of the new particle(s) (or area in 2D). Defaults to hypersphere volumes of the radii.
        mass : jax.typing.ArrayLike or None, optional
            Masses of the new particle(s). Defaults to ones. Ignored when a
            `mat_table` is provided.
        inertia : jax.typing.ArrayLike or None, optional
            Moments of inertia of the new particle(s). Defaults to solid disks (2D)
            or spheres (3D).
        clump_id : jax.typing.ArrayLike or None, optional
            clump_ids of the new clump(s). If `None`, new IDs are generated.
        bond_id : jax.typing.ArrayLike or None, optional
            List of connected unique_id values for each particle, storing the unique_ids of
            the particles it is connected to. Can be passed as a nested list
            (potentially with uneven lengths), or a 2D array.
            Connections are automatically symmetrized and padded with -1.
            If `None`, defaults to no connections.
        mat_id : jax.typing.ArrayLike or None, optional
            Material IDs of the new particle(s). Defaults to zeros.
        species_id : jax.typing.ArrayLike or None, optional
            Species IDs of the new particle(s). Defaults to zeros.
        fixed : jax.typing.ArrayLike or None, optional
            Fixed status of the new particle(s). Defaults to all `False`.
        mat_table : MaterialTable or None, optional
            Optional material table providing per-material densities. When provided,
            masses are computed from `density` and particle volume.

        Returns
        -------
        State
            A new `State` instance containing all particles from the original
            `state` plus the newly added particles.

        Raises
        ------
        ValueError
            If the created new particle state or the merged state is invalid.
        AssertionError
            If batch size or dimension mismatch between existing state and new particles.

        Example
        -------
        >>> import jaxdem as jdem
        >>> import jax.numpy as jnp
        >>>
        >>> # Initial state with 4 particles
        >>> state = jdem.State.create(pos=jnp.zeros((4, 2)))
        >>> print(f"Original state N: {state.N}, clump_ids: {state.clump_id}")
        >>>
        >>> # Add a single new particle
        >>> state_with_added_particle = jdem.State.add(
        ...     state,
        ...     pos=jnp.array([[10.0, 10.0]]),
        ...     rad=jnp.array([0.5]),
        ...     mass=jnp.array([2.0]),
        ... )
        >>> print(f"New state N: {state_with_added_particle.N}, clump_ids: {state_with_added_particle.clump_id}")
        >>> print(f"New particle position: {state_with_added_particle.pos[-1]}")
        >>>
        >>> # Add multiple new particles
        >>> state_multiple_added = jdem.State.add(
        ...     state,
        ...     pos=jnp.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]),
        ... )
        >>> print(f"State with multiple added N: {state_multiple_added.N}, clump_ids: {state_multiple_added.clump_id}")

        """
        state2 = State.create(
            pos=pos,
            pos_p=pos_p,
            vel=vel,
            force=force,
            q=q,
            ang_vel=ang_vel,
            torque=torque,
            rad=rad,
            _rad=_rad,
            volume=volume,
            mass=mass,
            inertia=inertia,
            clump_id=clump_id,
            bond_id=bond_id,
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
            mat_table=mat_table,
        )
        return State.merge(state, state2)

    @staticmethod
    @partial(jax.named_call, name="State.stack")
    def stack(states: Sequence[State]) -> State:
        r"""Concatenates a sequence of :class:`State` snapshots into a trajectory or batch along axis 0.

        This method is useful for collecting simulation snapshots over time into a
        single `State` object where the leading dimension represents time or when
        preparing a batched state.

        Parameters
        ----------
        states : Sequence[State]
            A sequence (e.g., list, tuple) of :class:`State` instances to be stacked.

        Returns
        -------
        State
            A new :class:`State` instance where each attribute is a JAX array with an
            additional leading dimension representing the stacked trajectory.
            For example, if input `pos` was `(N, dim)`, output `pos` will be `(T, N, dim)`.

        Raises
        ------
        ValueError
            If the input `states` sequence is empty.
            If the stacked `State` is invalid.
        AssertionError
            If any input state is invalid, or if there is a mismatch in
            spatial dimension (`dim`), batch size (`batch_size`), or
            number of particles (`N`) between the states in the sequence.

        Notes
        -----
        - No clump_id shifting is performed because the leading axis represents **time** (or another batch dimension), not new particles.

        Example
        -------
        >>> import jaxdem as jdem
        >>> import jax.numpy as jnp
        >>>
        >>> # Create a sequence of 3 simple 2D snapshots
        >>> snapshot1 = jdem.State.create(pos=jnp.array([[0.,0.], [1.,1.]]), vel=jnp.array([[0.1,0.], [0.0,0.1]]))
        >>> snapshot2 = jdem.State.create(pos=jnp.array([[0.1,0.], [1.,1.1]]), vel=jnp.array([[0.1,0.], [0.0,0.1]]))
        >>> snapshot3 = jdem.State.create(pos=jnp.array([[0.2,0.], [1.,1.2]]), vel=jnp.array([[0.1,0.], [0.0,0.1]]))
        >>>
        >>> trajectory_state = State.stack([snapshot1, snapshot2, snapshot3])
        >>>
        >>> print(f"Trajectory positions shape: {trajectory_state.pos.shape}") # Expected: (3, 2, 2)
        >>> print(f"Positions at time step 0:\\n{trajectory_state.pos[0]}")
        >>> print(f"Positions at time step 1:\\n{trajectory_state.pos[1]}")

        """
        states = list(states)
        if not states:
            raise ValueError("State.stack() received an empty list")

        ref = states[0]
        assert ref.is_valid, "first state is invalid"

        # ---------- consistency checks ---------------------------------
        for s in states[1:]:
            assert s.is_valid, "one state is invalid"
            assert s.dim == ref.dim, "dimension mismatch"
            assert s.batch_size == ref.batch_size, "batch size mismatch"
            assert s.N == ref.N, "particle count mismatch"

        # ---------- concatenate every leaf -----------------------------
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *states)

        if not stacked.is_valid:
            raise ValueError("stacked State is not valid")

        return stacked

    @staticmethod
    @partial(jax.named_call, name="State.unstack")
    def unstack(state: State) -> list[State]:
        """Split a stacked/batched :class:`State` along the leading axis into a Python list.

        This is the convenient inverse of :meth:`State.stack`:

        - If `stacked = State.stack([s0, s1, ...])`, then `State.unstack(stacked)` returns `[s0, s1, ...]`.

        Notes
        -----
        - The split is performed along axis 0 (the leading axis).
        - A single snapshot `State` (e.g. `pos.shape == (N, dim)`) cannot be unstacked with this
          method, because axis 0 would refer to particles, not snapshots.

        """
        if state.pos_c.ndim < 3:
            raise ValueError(
                "State.unstack() expects a stacked/batched State with a leading axis "
                f"(pos_c.ndim >= 3). Got pos_c.shape={state.pos_c.shape}."
            )

        n = int(state.pos_c.shape[0])
        return [jax.tree.map(lambda x, i=i: x[i], state) for i in range(n)]

    @staticmethod
    @partial(jax.named_call, name="State.add_clump")
    def add_clump(
        state: State,
        pos: ArrayLike,
        *,
        pos_p: ArrayLike | None = None,
        vel: ArrayLike | None = None,
        force: ArrayLike | None = None,
        q: Quaternion | None | ArrayLike | None = None,
        ang_vel: ArrayLike | None = None,
        torque: ArrayLike | None = None,
        rad: ArrayLike | None = None,
        volume: ArrayLike | None = None,
        mass: ArrayLike | None = None,
        inertia: ArrayLike | None = None,
        bond_id: ArrayLike | None = None,
        mat_id: ArrayLike | None = None,
        species_id: ArrayLike | None = None,
        fixed: ArrayLike | None = None,
    ) -> State:
        """Adds a new clump consisting of multiple spheres to an existing State.
        Rigid body properties (center of mass position pos_c, velocity, mass, orientation q,
        angular velocity, force, torque, inertia, fixed, and clump_id) are broadcasted/shared by all
        spheres in the new clump. The per-sphere properties that can vary within a rigid clump are
        pos_p (offsets in the body reference frame), rad, and individual ID fields (mat_id, species_id,
        and bond_id).

        Parameters
        ----------
        state : State
            The existing `State` to which particles will be added.
        pos : jax.typing.ArrayLike
            If `pos_p` is `None`, this represents the absolute coordinates of the spheres in the clump.
            If `pos_p` is not `None`, this represents the center of mass (COM) position of the clump.
            Expected shape: `(..., N, dim)`.
        pos_p : jax.typing.ArrayLike or None, optional
            Vector relative to the center of mass (pos_p = pos - pos_c) in the
            principal reference frame. This field should be constant. Shape is `(..., N, dim)`.
            If `None`, it is computed from the absolute coordinates provided in `pos` and the center
            of mass is calculated using the sphere volume weights.
        vel : jax.typing.ArrayLike or None, optional
            Velocities of the new particle(s). Defaults to zeros.
        force : jax.typing.ArrayLike or None, optional
            Forces of the new particle(s). Defaults to zeros.
        q : Quaternion or array-like, optional
            Initial orientations of the new particle(s). Defaults to identity quaternions.
        ang_vel : jax.typing.ArrayLike or None, optional
            Angular velocities of the new particle(s). Defaults to zeros.
        torque : jax.typing.ArrayLike or None, optional
            Torques of the new particle(s). Defaults to zeros.
        rad : jax.typing.ArrayLike or None, optional
            Radii of the new particle(s). Defaults to ones.
        volume : jax.typing.ArrayLike or None, optional
            Volume of the new particle(s) (or area in 2D). Defaults to hypersphere volumes of the radii.
        mass : jax.typing.ArrayLike or None, optional
            Masses of the new particle(s). Defaults to ones.
        inertia : jax.typing.ArrayLike or None, optional
            Moments of inertia of the new particle(s). Defaults to solid disks (2D)
            or spheres (3D).
        bond_id : jax.typing.ArrayLike or None, optional
            List of connected unique_id values for each particle, storing the unique_ids of
            the particles it is connected to. Can be passed as a nested list
            (potentially with uneven lengths), or a 2D array.
            Connections are automatically symmetrized and padded with -1.
            If `None`, defaults to no connections.
        mat_id : jax.typing.ArrayLike or None, optional
            Material IDs of the new particle(s). Defaults to zeros.
        species_id : jax.typing.ArrayLike or None, optional
            Species IDs of the new particle(s). Defaults to zeros.
        fixed : jax.typing.ArrayLike or None, optional
            Fixed status of the new particle(s). Defaults to all `False`.

        Returns
        -------
        State
            A new `State` instance containing all particles from the original
            `state` plus the newly added particles.

        """
        pos = jnp.asarray(pos)
        dim = pos.shape[-1]
        ang_dim = 1 if dim == 2 else 3

        if pos_p is None:
            # Case 1: pos represents absolute positions of the spheres.
            # We compute the COM using sphere volumes as weights.
            N_new = pos.shape[-2]
            if rad is None:
                rad_arr = jnp.ones(pos.shape[:-1])
            else:
                rad_arr = jnp.asarray(rad)

            # Compute sphere volumes for center of mass weighting
            vol_arr = _hypersphere_volume(rad_arr, dim)

            vol_sum = jnp.sum(vol_arr, axis=-1, keepdims=True)
            com = jnp.sum(pos * vol_arr[..., None], axis=-2) / jnp.where(
                vol_sum == 0, 1.0, vol_sum
            )

            pos_c = jnp.broadcast_to(com[..., None, :], pos.shape)
            raw_pos_p = pos - pos_c

            if q is not None:
                if isinstance(q, Quaternion):
                    q_obj = q
                else:
                    q_arr = jnp.asarray(q)
                    q_obj = Quaternion(w=q_arr[..., 0:1], xyz=q_arr[..., 1:])
                pos_p = Quaternion.rotate_back(q_obj, raw_pos_p)
            else:
                pos_p = raw_pos_p
        else:
            # Case 2: pos is already the COM (broadcast it to pos_c of shape (..., N_new, dim))
            pos_p = jnp.asarray(pos_p)
            N_new = pos_p.shape[-2]

            if pos.ndim == 1:
                com = pos
            else:
                com = pos[..., 0, :]

            pos_c = jnp.broadcast_to(com[..., None, :], (*pos_p.shape[:-2], N_new, dim))

        def broadcast_field(
            val: ArrayLike | None, shape: tuple[int, ...], dtype: Any
        ) -> None | jax.Array:
            return (
                jnp.broadcast_to(jnp.asarray(val, dtype=dtype), shape)
                if val is not None
                else None
            )

        rad = jnp.asarray(rad) if rad is not None else None

        vel = broadcast_field(vel, pos_c.shape, float)
        force = broadcast_field(force, pos_c.shape, float)

        ang_vel = broadcast_field(ang_vel, (*pos_c.shape[:-1], ang_dim), float)
        torque = broadcast_field(torque, (*pos_c.shape[:-1], ang_dim), float)
        inertia = broadcast_field(inertia, (*pos_c.shape[:-1], ang_dim), float)

        if q is not None:
            if isinstance(q, Quaternion):
                w = jnp.broadcast_to(
                    jnp.asarray(q.w, dtype=float), (*pos_c.shape[:-1], 1)
                )
                xyz = jnp.broadcast_to(
                    jnp.asarray(q.xyz, dtype=float), (*pos_c.shape[:-1], 3)
                )
                q = Quaternion(w=w, xyz=xyz)
            else:
                # If passed as ArrayLike (w, x, y, z), broadcast to (..., N, 4)
                q = broadcast_field(q, (*pos_c.shape[:-1], 4), float)

        mass = broadcast_field(mass, pos_c.shape[:-1], float)
        volume = broadcast_field(volume, pos_c.shape[:-1], float)

        clump_id = broadcast_field(0, pos_c.shape[:-1], int)
        mat_id = broadcast_field(mat_id, pos_c.shape[:-1], int)
        species_id = broadcast_field(species_id, pos_c.shape[:-1], int)
        fixed = broadcast_field(fixed, pos_c.shape[:-1], bool)

        state2 = State.create(
            pos=pos_c,
            pos_p=pos_p,
            vel=vel,
            force=force,
            q=q,
            ang_vel=ang_vel,
            torque=torque,
            rad=rad,
            volume=volume,
            mass=mass,
            inertia=inertia,
            clump_id=clump_id,
            bond_id=bond_id,
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
        )
        return State.merge(state, state2)

    @staticmethod
    def add_facet(
        state: State,
        vertices: ArrayLike,
        *,
        vel: ArrayLike | None = None,
        force: ArrayLike | None = None,
        q: Quaternion | None | ArrayLike | None = None,
        ang_vel: ArrayLike | None = None,
        torque: ArrayLike | None = None,
        thickness: float = 0.0,
        mass: ArrayLike | None = None,
        mat_id: ArrayLike | None = None,
        species_id: ArrayLike | None = None,
        fixed: ArrayLike | None = None,
        rigid: bool = True,
        safety_factor: float = 1.0,
    ) -> State:
        """Adds a new facet clump (2D line segment or 3D triangle) consisting of vertex spheres to an existing State.

        Note: Facets added via this method do not share vertices (i.e. each facet has its own independent copy of vertex particles).

        Parameters
        ----------
        state : State
            The existing state.
        vertices : ArrayLike
            Vertices of the facets, shape (..., V, dim).
        vel : ArrayLike or None, optional
            Initial linear velocity.
        force : ArrayLike or None, optional
            Initial force.
        q : Quaternion or ArrayLike or None, optional
            Initial orientations.
        ang_vel : ArrayLike or None, optional
            Initial angular velocities.
        torque : ArrayLike or None, optional
            Initial torques.
        thickness : float, optional
            Physical thickness/radius of the facet vertex spheres.
        mass : ArrayLike or None, optional
            Mass of the facets.
        mat_id : ArrayLike or None, optional
            Material IDs.
        species_id : ArrayLike or None, optional
            Species IDs.
        fixed : ArrayLike or None, optional
            Whether the facet vertices are fixed in space.
        rigid : bool, default True
            If True, the facet is rigid, grouping all vertices under the same clump ID, with
            proper inertia and orientation calculations. If False, the facet is flexible/deformable,
            meaning its vertices behave like individual spheres (having standard sphere moment of inertia,
            identity orientation, and unique clump IDs).
        safety_factor : float, default 1.0
            Factor to scale/multiply _rad to enlarge the broad-phase detection box.
        """
        import numpy as np

        vertices = jnp.asarray(vertices, dtype=float)
        dim = vertices.shape[-1]
        V = vertices.shape[-2]
        assert V in (
            2,
            3,
        ), f"Facets must be 2D segments (2 vertices) or 3D triangles (3 vertices). Got {V} vertices."
        assert (
            V == dim
        ), f"Number of vertices ({V}) must match spatial dimension ({dim})."

        batch_shape = vertices.shape[:-2]
        if batch_shape:
            raise NotImplementedError(
                "State.add_facet supports a single facet per call: `vertices` must "
                f"have shape (V, dim). Got shape {vertices.shape}. "
                "Call add_facet once per facet instead."
            )
        num_facets = int(np.prod(batch_shape)) if batch_shape else 1
        ang_dim = 1 if dim == 2 else 3
        vel_arr: Any
        force_arr: Any
        ang_vel_arr: Any
        torque_arr: Any
        mass_arr: Any
        mat_id_arr: Any
        species_id_arr: Any
        fixed_arr: Any

        if rigid:

            def check_single(
                val: Any, name: str, suffix_shape: tuple[int, ...] = ()
            ) -> None:
                if val is None:
                    return
                arr = jnp.asarray(val)
                if arr.shape == (*batch_shape, V, *suffix_shape) and V != 1:
                    raise ValueError(
                        f"For rigid facets, `{name}` must be a single value per facet, not per-vertex. "
                        f"Got shape {arr.shape}."
                    )

            check_single(vel, "vel", (dim,))
            check_single(force, "force", (dim,))
            check_single(ang_vel, "ang_vel", (ang_dim,))
            check_single(torque, "torque", (ang_dim,))
            check_single(thickness, "thickness", ())
            check_single(mass, "mass", ())
            check_single(mat_id, "mat_id", ())
            check_single(species_id, "species_id", ())
            check_single(fixed, "fixed", ())

            com = jnp.mean(vertices, axis=-2)
            rad_scalar = jnp.asarray(thickness)
            while rad_scalar.ndim > len(batch_shape):
                rad_scalar = jnp.squeeze(rad_scalar, axis=-1)
            rad_scalar = jnp.broadcast_to(rad_scalar, batch_shape)
            rad = jnp.broadcast_to(rad_scalar[..., None], (*batch_shape, V))

            # Broad-phase radius: facet contacts are keyed on the facet's
            # primary vertex, so each vertex's search ball must reach the
            # farthest contact point of its own facet — the farthest vertex
            # of the (convex) facet, plus the surface thickness.
            vertex_sep = linalg.norm(
                vertices[..., :, None, :] - vertices[..., None, :, :]
            )
            max_vertex_sep = jnp.max(vertex_sep, axis=-1)
            _rad = (max_vertex_sep + rad) * safety_factor

            if dim == 3:
                v10 = vertices[..., 1, :] - vertices[..., 0, :]
                v20 = vertices[..., 2, :] - vertices[..., 0, :]
                volume_scalar = 0.5 * linalg.norm(linalg.cross(v10, v20))
            else:
                v10 = vertices[..., 1, :] - vertices[..., 0, :]
                volume_scalar = linalg.norm(v10)

            if mass is None:
                mass_scalar = jnp.ones(batch_shape, dtype=float)
            else:
                mass_scalar = jnp.asarray(mass, dtype=float)
                while mass_scalar.ndim > len(batch_shape):
                    mass_scalar = jnp.squeeze(mass_scalar, axis=-1)
                mass_scalar = jnp.broadcast_to(mass_scalar, batch_shape)

            if dim == 3:
                com_3d = com[..., None, :]
                m1 = (vertices[..., 0, :] + vertices[..., 1, :]) / 2.0 - com_3d[
                    ..., 0, :
                ]
                m2 = (vertices[..., 1, :] + vertices[..., 2, :]) / 2.0 - com_3d[
                    ..., 0, :
                ]
                m3 = (vertices[..., 2, :] + vertices[..., 0, :]) / 2.0 - com_3d[
                    ..., 0, :
                ]
                m_pt = mass_scalar / 3.0

                def cov(r: jax.Array) -> jax.Array:
                    return r[..., :, None] * r[..., None, :]

                C = (cov(m1) + cov(m2) + cov(m3)) * m_pt[..., None, None]
                trace_C = jnp.trace(C, axis1=-2, axis2=-1)[..., None, None]
                I_tensor = trace_C * jnp.eye(3) - C

                w, v_rot = jnp.linalg.eigh(I_tensor)
                inertia_scalar = w + (mass_scalar * INERTIA_REGULARIZATION)[..., None]

                det = jnp.linalg.det(v_rot)
                v_rot = jnp.where(
                    det[..., None, None] < 0, v_rot.at[..., :, 2].multiply(-1.0), v_rot
                )

                q_obj = Quaternion.from_rotation_matrix(v_rot)
                pos_p = Quaternion.rotate_back(q_obj, vertices - com[..., None, :])
                inertia = inertia_scalar
            else:
                half_len = volume_scalar / 2.0
                inertia_scalar = (1.0 / 12.0) * mass_scalar * volume_scalar**2
                inertia = (inertia_scalar + mass_scalar * INERTIA_REGULARIZATION)[
                    ..., None
                ]

                diff = vertices[..., 1, :] - vertices[..., 0, :]
                theta = jnp.atan2(diff[..., 1], diff[..., 0])
                q_w = jnp.cos(theta / 2.0)[..., None]
                q_xyz = jnp.zeros((*theta.shape, 3))
                q_xyz = q_xyz.at[..., 2].set(jnp.sin(theta / 2.0))
                q_obj = Quaternion.create(w=q_w, xyz=q_xyz)

                v0 = jnp.stack([-half_len, jnp.zeros_like(half_len)], axis=-1)
                v1 = jnp.stack([half_len, jnp.zeros_like(half_len)], axis=-1)
                pos_p = jnp.stack([v0, v1], axis=-2)

            volume_arr = jnp.broadcast_to(
                (volume_scalar[..., None] / V), (*batch_shape, V)
            )
            mass_arr = jnp.broadcast_to(mass_scalar[..., None], (*batch_shape, V))
            inertia_arr = jnp.broadcast_to(
                inertia[..., None, :], (*batch_shape, V, ang_dim)
            )

            mat_id_arr = _broadcast_param(
                mat_id if mat_id is not None else 0, batch_shape, V, (), int
            )
            species_id_arr = _broadcast_param(
                species_id if species_id is not None else 0, batch_shape, V, (), int
            )
            fixed_arr = _broadcast_param(
                fixed if fixed is not None else False, batch_shape, V, (), bool
            )

            state2 = State.add_clump(
                state=state,
                pos=jnp.broadcast_to(com[..., None, :], (*batch_shape, V, dim)),
                pos_p=pos_p,
                vel=vel,
                force=force,
                q=q_obj,
                ang_vel=ang_vel,
                torque=torque,
                rad=rad,
                volume=volume_arr,
                mass=mass_arr,
                inertia=inertia_arr,
                mat_id=mat_id_arr,
                species_id=species_id_arr,
                fixed=fixed_arr,
            )

            N_old = state.N
            f_offset = jnp.maximum(0, jnp.max(state.facet_id, initial=-1) + 1)

            facet_indices = jnp.arange(num_facets, dtype=int)
            vertices_of_facets = facet_indices[:, None] * V + jnp.arange(V)
            facet_vertices_flat = jnp.broadcast_to(
                vertices_of_facets[:, None, :], (num_facets, V, V)
            )
            new_facet_vertices_local = facet_vertices_flat.reshape((*batch_shape, V, V))

            facet_id_to_set = (
                jnp.broadcast_to(
                    jnp.arange(num_facets, dtype=int).reshape((*batch_shape, 1)),
                    (*batch_shape, V),
                )
                + f_offset
            )
            facet_vertices_to_set = new_facet_vertices_local + N_old

            updated_facet_id = state2.facet_id.at[..., N_old:].set(
                facet_id_to_set.ravel()
            )
            updated_facet_vertices = state2.facet_vertices.at[..., N_old:, :].set(
                facet_vertices_to_set.reshape((*batch_shape, V, dim))
            )
            updated_rad_search = state2._rad.at[..., N_old:].set(_rad.ravel())

            state2.facet_id = updated_facet_id
            state2.facet_vertices = updated_facet_vertices
            state2._rad = updated_rad_search

            return state2

        else:
            rad = jnp.broadcast_to(jnp.asarray(thickness), (*batch_shape, V))

            # Broad-phase radius keyed on the primary vertex: see the rigid
            # branch above.
            vertex_sep = linalg.norm(
                vertices[..., :, None, :] - vertices[..., None, :, :]
            )
            max_vertex_sep = jnp.max(vertex_sep, axis=-1)
            _rad = (max_vertex_sep + rad) * safety_factor

            vel_arr = _broadcast_param(vel, batch_shape, V, (dim,), float)
            force_arr = _broadcast_param(force, batch_shape, V, (dim,), float)
            ang_vel_arr = _broadcast_param(ang_vel, batch_shape, V, (ang_dim,), float)
            torque_arr = _broadcast_param(torque, batch_shape, V, (ang_dim,), float)
            mass_arr = _broadcast_param(mass, batch_shape, V, (), float)
            mat_id_arr = _broadcast_param(mat_id, batch_shape, V, (), int)
            species_id_arr = _broadcast_param(species_id, batch_shape, V, (), int)
            fixed_arr = _broadcast_param(fixed, batch_shape, V, (), bool)

            if q is not None:
                if not isinstance(q, Quaternion):
                    q_arr = jnp.asarray(q)
                    q_obj = Quaternion.create(w=q_arr[..., 0:1], xyz=q_arr[..., 1:])
                else:
                    q_obj = q

                if q_obj.w.shape == (*batch_shape, V, 1):
                    q_broadcasted = q_obj
                else:
                    qw = q_obj.w
                    qxyz = q_obj.xyz
                    if qw.shape == (*batch_shape, 1):
                        qw = jnp.expand_dims(qw, -2)
                        qxyz = jnp.expand_dims(qxyz, -2)
                    qw_broadcasted = jnp.broadcast_to(qw, (*batch_shape, V, 1))
                    qxyz_broadcasted = jnp.broadcast_to(qxyz, (*batch_shape, V, 3))
                    q_broadcasted = Quaternion.create(
                        w=qw_broadcasted, xyz=qxyz_broadcasted
                    )
            else:
                q_broadcasted = None

            facet_indices = jnp.arange(num_facets, dtype=int)
            vertices_of_facets = facet_indices[:, None] * V + jnp.arange(V)
            facet_vertices_flat = jnp.broadcast_to(
                vertices_of_facets[:, None, :], (num_facets, V, V)
            )
            new_facet_vertices_local = facet_vertices_flat.reshape((*batch_shape, V, V))

            facet_id_arr = jnp.broadcast_to(
                jnp.arange(num_facets, dtype=int).reshape((*batch_shape, 1)),
                (*batch_shape, V),
            )
            clump_id = jnp.arange(num_facets * V, dtype=int).reshape((*batch_shape, V))

            state2 = State.create(
                pos=vertices,
                pos_p=jnp.zeros_like(vertices),
                vel=vel_arr,
                force=force_arr,
                q=q_broadcasted,
                ang_vel=ang_vel_arr,
                torque=torque_arr,
                rad=rad,
                _rad=_rad,
                mass=mass_arr,
                clump_id=clump_id,
                facet_id=facet_id_arr,
                facet_vertices=new_facet_vertices_local,
                mat_id=mat_id_arr,
                species_id=species_id_arr,
                fixed=fixed_arr,
            )
            return State.merge(state, state2)

    @staticmethod
    def add_mesh(
        state: State,
        vertices: ArrayLike,
        faces: ArrayLike,
        *,
        vel: ArrayLike | None = None,
        force: ArrayLike | None = None,
        q: Quaternion | None | ArrayLike | None = None,
        ang_vel: ArrayLike | None = None,
        torque: ArrayLike | None = None,
        thickness: float = 0.0,
        mass: ArrayLike | None = None,
        mat_id: ArrayLike | None = None,
        species_id: ArrayLike | None = None,
        fixed: ArrayLike | None = None,
        rigid: bool = True,
        filled: bool = True,
        safety_factor: float = 1.0,
    ) -> State:
        """Adds a new mesh (collection of facets) consisting of vertex spheres to an existing State.

        Parameters
        ----------
        state : State
            The existing state.
        vertices : ArrayLike
            Vertices of the mesh, shape (..., V_mesh, dim).
        faces : ArrayLike
            Faces of the mesh (indices into vertices), shape (..., F, dim).
        vel : ArrayLike or None, optional
            Initial linear velocity.
        force : ArrayLike or None, optional
            Initial force.
        q : Quaternion or ArrayLike or None, optional
            Initial orientations.
        ang_vel : ArrayLike or None, optional
            Initial angular velocities.
        torque : ArrayLike or None, optional
            Initial torques.
        thickness : float, optional
            Physical thickness/radius of the facet vertex spheres.
        mass : ArrayLike or None, optional
            Mass of the mesh.
        mat_id : ArrayLike or None, optional
            Material IDs.
        species_id : ArrayLike or None, optional
            Species IDs.
        fixed : ArrayLike or None, optional
            Whether the facet vertices are fixed in space.
        rigid : bool, default True
            If True, the mesh is rigid, grouping all vertices under the same clump ID, with
            proper inertia and orientation calculations. If False, the mesh is flexible/deformable.
        filled : bool, default True
            If True, the mesh represents a filled solid polyhedron/polygon. If False, it represents
            a hollow boundary shell.
        safety_factor : float, default 1.0
            Factor to scale/multiply _rad to enlarge the broad-phase detection box.
        """
        import numpy as np

        vertices = jnp.asarray(vertices, dtype=float)
        faces = jnp.asarray(faces, dtype=int)
        dim = vertices.shape[-1]
        V_face = faces.shape[-1]
        assert (
            V_face == dim
        ), f"Each face must have {dim} vertices. Got shape {faces.shape}."

        batch_shape = vertices.shape[:-2]
        if batch_shape:
            raise NotImplementedError(
                "State.add_mesh supports a single mesh per call: `vertices` must "
                f"have shape (V_mesh, dim). Got shape {vertices.shape}. "
                "Call add_mesh once per mesh instead."
            )
        num_meshes = int(np.prod(batch_shape)) if batch_shape else 1
        ang_dim = 1 if dim == 2 else 3
        vel_arr: Any
        force_arr: Any
        ang_vel_arr: Any
        torque_arr: Any
        mass_arr: Any
        mat_id_arr: Any
        species_id_arr: Any
        fixed_arr: Any

        if batch_shape:

            def index_mesh(v: jax.Array, f: jax.Array) -> jax.Array:
                return v[f]

            v_flat = vertices.reshape((-1, *vertices.shape[-2:]))
            f_flat = faces.reshape((-1, *faces.shape[-2:]))
            face_vertices_flat = jax.vmap(index_mesh)(v_flat, f_flat)
            face_vertices = face_vertices_flat.reshape(
                (*batch_shape, *faces.shape[-2:], dim, dim)
            )
        else:
            face_vertices = vertices[faces]

        F = faces.shape[-2]
        V = dim

        if rigid:

            def check_single(
                val: Any, name: str, suffix_shape: tuple[int, ...] = ()
            ) -> None:
                if val is None:
                    return
                arr = jnp.asarray(val)
                if arr.shape == (*batch_shape, vertices.shape[-2], *suffix_shape):
                    raise ValueError(
                        f"For rigid meshes, `{name}` must be a single value per mesh, not per-vertex. "
                        f"Got shape {arr.shape}."
                    )

            check_single(vel, "vel", (dim,))
            check_single(force, "force", (dim,))
            check_single(ang_vel, "ang_vel", (ang_dim,))
            check_single(torque, "torque", (ang_dim,))
            check_single(thickness, "thickness", ())
            check_single(mass, "mass", ())
            check_single(mat_id, "mat_id", ())
            check_single(species_id, "species_id", ())
            check_single(fixed, "fixed", ())

            if batch_shape:
                v_flat = vertices.reshape((-1, *vertices.shape[-2:]))
                f_flat = faces.reshape((-1, *faces.shape[-2:]))
                com, I_val, vol_or_area = jax.vmap(
                    lambda v, f: _compute_single_mesh_properties(v, f, dim, filled)
                )(v_flat, f_flat)
                com = com.reshape((*batch_shape, dim))
                vol_or_area = vol_or_area.reshape(batch_shape)
                if dim == 3:
                    I_val = I_val.reshape((*batch_shape, 3, 3))
                else:
                    I_val = I_val.reshape(batch_shape)
            else:
                com, I_val, vol_or_area = _compute_single_mesh_properties(
                    vertices, faces, dim, filled
                )

            if mass is None:
                mass_scalar = jnp.ones(batch_shape, dtype=float)
            else:
                mass_scalar = jnp.asarray(mass, dtype=float)

            density = mass_scalar / jnp.where(vol_or_area == 0.0, 1.0, vol_or_area)
            if dim == 3:
                scaled_I_val = I_val * density[..., None, None]
                w, v_rot = jnp.linalg.eigh(scaled_I_val)
                inertia_scalar = w + (mass_scalar * INERTIA_REGULARIZATION)[..., None]

                det = jnp.linalg.det(v_rot)
                v_rot = jnp.where(
                    det[..., None, None] < 0, v_rot.at[..., :, 2].multiply(-1.0), v_rot
                )

                q_obj = Quaternion.from_rotation_matrix(v_rot)
                raw_pos_p = face_vertices - com[..., None, None, :]
                raw_pos_p_flat = raw_pos_p.reshape((*batch_shape, F * V, dim))
                pos_p_flat = Quaternion.rotate_back(q_obj, raw_pos_p_flat)
                pos_p = pos_p_flat.reshape((*batch_shape, F, V, dim))
                inertia = inertia_scalar
            else:
                scaled_I_val = I_val * density
                inertia_scalar = scaled_I_val + mass_scalar * INERTIA_REGULARIZATION
                inertia = inertia_scalar[..., None]

                q_obj = Quaternion.create(
                    jnp.ones((*batch_shape, 1)), jnp.zeros((*batch_shape, 3))
                )
                raw_pos_p = face_vertices - com[..., None, None, :]
                pos_p = raw_pos_p

            rad = jnp.broadcast_to(
                jnp.asarray(thickness)[..., None, None], (*batch_shape, F, V)
            )

            # Broad-phase radius: per-vertex within each face — facet contacts
            # are keyed on each face's primary vertex (see State.add_facet).
            vertex_sep = linalg.norm(
                face_vertices[..., :, None, :] - face_vertices[..., None, :, :]
            )
            max_vertex_sep = jnp.max(vertex_sep, axis=-1)
            _rad = (max_vertex_sep + rad) * safety_factor

            volume_arr = jnp.broadcast_to(
                (vol_or_area[..., None, None] / (F * V)), (*batch_shape, F, V)
            )
            # Clump convention: every clump member stores the total clump mass.
            mass_arr = jnp.broadcast_to(
                mass_scalar[..., None, None], (*batch_shape, F, V)
            )
            inertia_arr = jnp.broadcast_to(
                inertia[..., None, None, :], (*batch_shape, F, V, ang_dim)
            )

            mat_id_arr = _broadcast_mesh_param(
                mat_id if mat_id is not None else 0, batch_shape, F, V, (), int
            )
            species_id_arr = _broadcast_mesh_param(
                species_id if species_id is not None else 0, batch_shape, F, V, (), int
            )
            fixed_arr = _broadcast_mesh_param(
                fixed if fixed is not None else False, batch_shape, F, V, (), bool
            )

            state2 = State.add_clump(
                state=state,
                pos=jnp.broadcast_to(
                    com[..., None, None, :], (*batch_shape, F, V, dim)
                ).reshape((*batch_shape, F * V, dim)),
                pos_p=pos_p.reshape((*batch_shape, F * V, dim)),
                vel=vel,
                force=force,
                q=q_obj,
                ang_vel=ang_vel,
                torque=torque,
                rad=rad.reshape((*batch_shape, F * V)),
                volume=volume_arr.reshape((*batch_shape, F * V)),
                mass=mass_arr.reshape((*batch_shape, F * V)),
                inertia=inertia_arr.reshape((*batch_shape, F * V, ang_dim)),
                mat_id=mat_id_arr.reshape((*batch_shape, F * V)),
                species_id=species_id_arr.reshape((*batch_shape, F * V)),
                fixed=fixed_arr.reshape((*batch_shape, F * V)),
            )

            N_old = state.N
            f_offset = jnp.maximum(0, jnp.max(state.facet_id, initial=-1) + 1)

            face_indices = jnp.arange(num_meshes * F, dtype=int)
            vertices_of_faces = face_indices[:, None] * V + jnp.arange(V)
            facet_vertices_flat = jnp.broadcast_to(
                vertices_of_faces[:, None, :], (num_meshes * F, V, V)
            )
            new_facet_vertices = facet_vertices_flat.reshape((*batch_shape, F, V, V))

            facet_id_to_set = (
                jnp.broadcast_to(
                    jnp.arange(num_meshes * F, dtype=int).reshape((*batch_shape, F, 1)),
                    (*batch_shape, F, V),
                )
                + f_offset
            )
            facet_vertices_to_set = new_facet_vertices + N_old

            updated_facet_id = state2.facet_id.at[..., N_old:].set(
                facet_id_to_set.ravel()
            )
            updated_facet_vertices = state2.facet_vertices.at[..., N_old:, :].set(
                facet_vertices_to_set.reshape((*batch_shape, F * V, dim))
            )
            updated_rad_search = state2._rad.at[..., N_old:].set(
                _rad.reshape((*batch_shape, F * V)).ravel()
            )

            state2.facet_id = updated_facet_id
            state2.facet_vertices = updated_facet_vertices
            state2._rad = updated_rad_search

            return state2

        else:
            # Flexible mesh
            # Compute _rad: per-vertex within each face — facet contacts are
            # keyed on each face's primary vertex (see State.add_facet).
            rad = jnp.broadcast_to(
                jnp.asarray(thickness)[..., None, None], (*batch_shape, F, V)
            )
            vertex_sep = linalg.norm(
                face_vertices[..., :, None, :] - face_vertices[..., None, :, :]
            )
            max_vertex_sep = jnp.max(vertex_sep, axis=-1)
            _rad = (max_vertex_sep + rad) * safety_factor

            pos = face_vertices.reshape((*batch_shape, F * V, dim))
            pos_p = jnp.zeros_like(pos)

            vel_arr = _broadcast_mesh_param(vel, batch_shape, F, V, (dim,), float)
            if vel_arr is not None:
                vel_arr = vel_arr.reshape((*batch_shape, F * V, dim))

            force_arr = _broadcast_mesh_param(force, batch_shape, F, V, (dim,), float)
            if force_arr is not None:
                force_arr = force_arr.reshape((*batch_shape, F * V, dim))

            ang_vel_arr = _broadcast_mesh_param(
                ang_vel, batch_shape, F, V, (ang_dim,), float
            )
            if ang_vel_arr is not None:
                ang_vel_arr = ang_vel_arr.reshape((*batch_shape, F * V, ang_dim))

            torque_arr = _broadcast_mesh_param(
                torque, batch_shape, F, V, (ang_dim,), float
            )
            if torque_arr is not None:
                torque_arr = torque_arr.reshape((*batch_shape, F * V, ang_dim))

            mass_arr = _broadcast_mesh_param(mass, batch_shape, F, V, (), float)
            if mass_arr is not None:
                mass_arr = mass_arr.reshape((*batch_shape, F * V))

            mat_id_arr = _broadcast_mesh_param(
                mat_id if mat_id is not None else 0, batch_shape, F, V, (), int
            )
            if mat_id_arr is not None:
                mat_id_arr = mat_id_arr.reshape((*batch_shape, F * V))

            species_id_arr = _broadcast_mesh_param(
                species_id if species_id is not None else 0, batch_shape, F, V, (), int
            )
            if species_id_arr is not None:
                species_id_arr = species_id_arr.reshape((*batch_shape, F * V))

            fixed_arr = _broadcast_mesh_param(
                fixed if fixed is not None else False, batch_shape, F, V, (), bool
            )
            if fixed_arr is not None:
                fixed_arr = fixed_arr.reshape((*batch_shape, F * V))

            if q is not None:
                if not isinstance(q, Quaternion):
                    q_arr = jnp.asarray(q)
                    q_obj = Quaternion.create(w=q_arr[..., 0:1], xyz=q_arr[..., 1:])
                else:
                    q_obj = q
                qw = jnp.broadcast_to(q_obj.w, (*batch_shape, F, V, 1))
                qxyz = jnp.broadcast_to(q_obj.xyz, (*batch_shape, F, V, 3))
                q_broadcasted = Quaternion.create(
                    w=qw.reshape((*batch_shape, F * V, 1)),
                    xyz=qxyz.reshape((*batch_shape, F * V, 3)),
                )
            else:
                q_broadcasted = None

            face_indices = jnp.arange(num_meshes * F, dtype=int)
            vertices_of_faces = face_indices[:, None] * V + jnp.arange(V)
            facet_vertices_flat = jnp.broadcast_to(
                vertices_of_faces[:, None, :], (num_meshes * F, V, V)
            )
            new_facet_vertices = facet_vertices_flat.reshape((*batch_shape, F, V, V))

            facet_id_to_set = jnp.broadcast_to(
                jnp.arange(num_meshes * F, dtype=int).reshape((*batch_shape, F, 1)),
                (*batch_shape, F, V),
            )
            clump_id = jnp.arange(num_meshes * F * V, dtype=int).reshape(
                (*batch_shape, F * V)
            )

            state2 = State.create(
                pos=pos,
                pos_p=pos_p,
                vel=vel_arr,
                force=force_arr,
                q=q_broadcasted,
                ang_vel=ang_vel_arr,
                torque=torque_arr,
                rad=rad.reshape((*batch_shape, F * V)),
                _rad=_rad.reshape((*batch_shape, F * V)),
                mass=mass_arr,
                clump_id=clump_id,
                facet_id=facet_id_to_set.reshape((*batch_shape, F * V)),
                facet_vertices=new_facet_vertices.reshape((*batch_shape, F * V, dim)),
                mat_id=mat_id_arr,
                species_id=species_id_arr,
                fixed=fixed_arr,
            )
            return State.merge(state, state2)

    @staticmethod
    def add_connected_facet(
        state: State,
        vertex_specs: list[int | ArrayLike],
        *,
        vel: ArrayLike | None = None,
        force: ArrayLike | None = None,
        q: Quaternion | None | ArrayLike | None = None,
        ang_vel: ArrayLike | None = None,
        torque: ArrayLike | None = None,
        thickness: float = 0.0,
        mass: ArrayLike | None = None,
        mat_id: ArrayLike | None = None,
        species_id: ArrayLike | None = None,
        fixed: ArrayLike | None = None,
        rigid: bool = True,
        safety_factor: float = 1.0,
    ) -> State:
        """Adds a new facet connecting existing vertices and/or newly added vertices to the State.

        Parameters
        ----------
        state : State
            The existing state.
        vertex_specs : list of int or ArrayLike
            Each spec represents a vertex of the new facet.
            If a spec is a scalar integer (or scalar array), it is treated as the unique_id of
            an existing vertex.
            Otherwise, it is treated as a position array of shape (dim,) for a new vertex.
        """
        import numpy as np

        dim = state.dim
        V = len(vertex_specs)
        assert V in (
            2,
            3,
        ), f"Facet must be a 2D segment (2 vertices) or 3D triangle (3 vertices). Got {V} vertices."
        assert (
            V == dim
        ), f"Number of vertices ({V}) must match spatial dimension ({dim})."

        existing_uids = []
        new_positions = []
        spec_is_existing = []

        for spec in vertex_specs:
            spec_arr = jnp.asarray(spec)
            if spec_arr.ndim == 0 and jnp.issubdtype(spec_arr.dtype, jnp.integer):
                uid = int(spec_arr)
                if uid < 0 or uid >= state.N:
                    raise ValueError(
                        f"Existing vertex unique_id {uid} is out of bounds."
                    )
                existing_uids.append(uid)
                spec_is_existing.append(True)
            else:
                new_positions.append(spec_arr)
                spec_is_existing.append(False)

        # 1. Check species_id condition
        if species_id is None:
            if existing_uids:
                target_species_id = int(cast(Any, state.species_id[existing_uids[0]]))
                for uid in existing_uids:
                    if int(cast(Any, state.species_id[uid])) != target_species_id:
                        raise ValueError(
                            "All connected existing vertices must share the same species_id."
                        )
            else:
                target_species_id = 0
        else:
            target_species_id = int(cast(Any, jnp.asarray(species_id).ravel()[0]))
            for uid in existing_uids:
                if int(cast(Any, state.species_id[uid])) != target_species_id:
                    raise ValueError(
                        f"Existing vertex {uid} species_id ({state.species_id[uid]}) "
                        f"does not match target species_id ({target_species_id})."
                    )

        # 2. Check hybrid facets (rigid vs flexible)
        if existing_uids:
            is_rigid_vertex = []
            for uid in existing_uids:
                clump_id_val = int(state.clump_id[uid])
                clump_size = int(np.sum(np.asarray(state.clump_id) == clump_id_val))
                is_rigid_vertex.append(clump_size > 1)
            if any(is_rigid_vertex) != all(is_rigid_vertex):
                raise ValueError(
                    "Cannot mix rigid and flexible existing vertices in the same facet."
                )

            clump_rigid = is_rigid_vertex[0]
            if rigid != clump_rigid:
                raise ValueError(
                    f"Requested rigid={rigid} does not match the rigidity of the existing connected vertices "
                    f"(rigid={clump_rigid}). Hybrid facets are not supported."
                )

            if clump_rigid:
                target_clump_id = int(state.clump_id[existing_uids[0]])
                for uid in existing_uids:
                    if int(state.clump_id[uid]) != target_clump_id:
                        raise ValueError(
                            "All connected existing rigid vertices must belong to the same clump."
                        )
            else:
                target_clump_id = None
        else:
            clump_rigid = rigid
            target_clump_id = None

        # If all vertices are new, defer to add_facet directly
        if len(existing_uids) == 0:
            return State.add_facet(
                state,
                jnp.stack(new_positions, axis=0),
                vel=vel,
                force=force,
                q=q,
                ang_vel=ang_vel,
                torque=torque,
                thickness=thickness,
                mass=mass,
                mat_id=mat_id,
                species_id=species_id,
                fixed=fixed,
                rigid=rigid,
                safety_factor=safety_factor,
            )

        ang_dim = 1 if dim == 2 else 3

        # Create new particles if any
        new_N = len(new_positions)
        if new_N > 0:
            new_pos = jnp.stack(new_positions, axis=0)

            if mass is None:
                new_mass = jnp.ones(new_N, dtype=float)
            else:
                mass_val = jnp.asarray(mass).ravel()[0]
                new_mass = jnp.full(new_N, mass_val, dtype=float)

            new_vel = jnp.broadcast_to(
                jnp.asarray(vel) if vel is not None else jnp.zeros(dim), (new_N, dim)
            )
            new_force = jnp.broadcast_to(
                jnp.asarray(force) if force is not None else jnp.zeros(dim),
                (new_N, dim),
            )
            new_ang_vel = jnp.broadcast_to(
                jnp.asarray(ang_vel) if ang_vel is not None else jnp.zeros(ang_dim),
                (new_N, ang_dim),
            )
            new_torque = jnp.broadcast_to(
                jnp.asarray(torque) if torque is not None else jnp.zeros(ang_dim),
                (new_N, ang_dim),
            )
            new_mat_id = jnp.broadcast_to(
                jnp.asarray(mat_id) if mat_id is not None else jnp.zeros((), dtype=int),
                (new_N,),
            )
            new_fixed = jnp.broadcast_to(
                jnp.asarray(fixed) if fixed is not None else jnp.zeros((), dtype=bool),
                (new_N,),
            )
            new_species_id = jnp.full(new_N, target_species_id, dtype=int)

            state_new = State.create(
                pos=new_pos,
                vel=new_vel,
                force=new_force,
                q=None,
                ang_vel=new_ang_vel,
                torque=new_torque,
                rad=jnp.full(new_N, thickness, dtype=float),
                mass=new_mass,
                mat_id=new_mat_id,
                species_id=new_species_id,
                fixed=new_fixed,
            )

            N_old = state.N
            state2 = State.merge(state, state_new)
            new_uids = [N_old + i for i in range(new_N)]
        else:
            state2 = state
            N_old = state.N
            new_uids = []

        # Map specs to the unique IDs in the merged state2
        facet_uids = []
        new_idx = 0
        for i, is_existing in enumerate(spec_is_existing):
            if is_existing:
                facet_uids.append(int(cast(Any, vertex_specs[i])))
            else:
                facet_uids.append(new_uids[new_idx])
                new_idx += 1

        # 3. Update facet_id and facet_vertices
        f_offset = jnp.maximum(0, jnp.max(state2.facet_id, initial=-1) + 1)
        updated_facet_id = state2.facet_id.at[jnp.array(facet_uids)].set(f_offset)
        updated_facet_vertices = state2.facet_vertices.at[jnp.array(facet_uids), :].set(
            jnp.array(facet_uids)
        )

        state2.facet_id = updated_facet_id
        state2.facet_vertices = updated_facet_vertices

        # 4. Update search radius _rad
        facet_uid_arr = jnp.array(facet_uids)
        com_facet = jnp.mean(state2.pos[facet_uid_arr], axis=0)
        dist = linalg.norm(state2.pos[facet_uid_arr] - com_facet)
        rad_val = (dist + state2.rad[facet_uid_arr]) * safety_factor
        state2._rad = state2._rad.at[facet_uid_arr].max(rad_val)

        # 5. Handle rigid clump properties if rigid
        if rigid:
            if target_clump_id is None:
                pass
            else:
                if new_uids:
                    updated_clump_id = state2.clump_id.at[jnp.array(new_uids)].set(
                        target_clump_id
                    )
                    state2.clump_id = updated_clump_id

                clump_idxs = jnp.where(state2.clump_id == target_clump_id)[0]
                P = state2.pos[clump_idxs]
                R = state2.rad[clump_idxs]

                # Clump convention: every member stores the total clump mass.
                # Recover *physical* per-member masses for the COM/inertia
                # integrals: pre-existing members share the previous clump
                # total equally, while newly added members still carry their
                # own physical mass.
                stored_mass = state2.mass[clump_idxs]
                if new_uids:
                    is_new = jnp.isin(clump_idxs, jnp.asarray(new_uids, dtype=int))
                else:
                    is_new = jnp.zeros_like(clump_idxs, dtype=bool)
                n_old_members = int(jnp.sum(~is_new))
                old_total_mass = stored_mass[~is_new][0]
                M = jnp.where(is_new, stored_mass, old_total_mass / n_old_members)

                M_sum = jnp.sum(M)
                com = jnp.sum(P * M[:, None], axis=0) / jnp.where(
                    M_sum == 0.0, 1.0, M_sum
                )

                relative_pos = P - com[None, :]
                cov_pts = relative_pos[:, :, None] * relative_pos[:, None, :]
                if dim == 3:
                    cov_sph = (
                        0.2
                        * M[:, None, None]
                        * (R[:, None, None] ** 2)
                        * jnp.eye(3)[None, :, :]
                    )
                    C_total = jnp.sum(M[:, None, None] * cov_pts + cov_sph, axis=0)

                    trace_C = jnp.trace(C_total)
                    I_tensor = trace_C * jnp.eye(3) - C_total

                    w, v_rot = jnp.linalg.eigh(I_tensor)
                    inertia_val = w + M_sum * INERTIA_REGULARIZATION
                    det = jnp.linalg.det(v_rot)
                    v_rot = jnp.where(det < 0, v_rot.at[:, 2].multiply(-1.0), v_rot)
                    q_clump_full = Quaternion.from_rotation_matrix(v_rot[None, :, :])
                    q_clump = Quaternion.create(
                        w=q_clump_full.w[0], xyz=q_clump_full.xyz[0]
                    )
                    pos_p_clump = Quaternion.rotate_back(q_clump, relative_pos)
                else:
                    cov_sph = (
                        0.25
                        * M[:, None, None]
                        * (R[:, None, None] ** 2)
                        * jnp.eye(2)[None, :, :]
                    )
                    C_total = jnp.sum(M[:, None, None] * cov_pts + cov_sph, axis=0)

                    trace_C = jnp.trace(C_total)
                    inertia_val = jnp.array([trace_C + M_sum * INERTIA_REGULARIZATION])
                    q_clump = Quaternion.create(jnp.ones(1), jnp.zeros(3))
                    pos_p_clump = relative_pos

                updated_pos_c = state2.pos_c.at[clump_idxs].set(com)
                updated_pos_p = state2.pos_p.at[clump_idxs].set(pos_p_clump)

                updated_qw = state2.q.w.at[clump_idxs].set(q_clump.w)
                updated_qxyz = state2.q.xyz.at[clump_idxs].set(q_clump.xyz)
                updated_q = Quaternion.create(w=updated_qw, xyz=updated_qxyz)

                updated_inertia = state2.inertia.at[clump_idxs].set(inertia_val)

                # Write the new clump total mass to every member (convention).
                updated_mass = state2.mass.at[clump_idxs].set(M_sum)

                state2.pos_c = updated_pos_c
                state2.inertia = updated_inertia
                state2.mass = updated_mass
                # Assigning `pos_p` and `q` also refreshes the `_pos_p_rot`
                # cache (consumed by `state.pos`).
                state2.pos_p = updated_pos_p
                state2.q = updated_q

        return state2


def _broadcast_param(
    val: Any,
    batch_shape: tuple[int, ...],
    V: int,
    suffix_shape: tuple[int, ...],
    dtype: Any,
) -> jax.Array | None:
    if val is None:
        return None
    arr = jnp.asarray(val, dtype=dtype)
    target_shape_per_vertex = (*batch_shape, V, *suffix_shape)
    target_shape_single = (*batch_shape, *suffix_shape)
    if arr.shape == target_shape_per_vertex:
        return arr
    if arr.shape == target_shape_single:
        axis_idx = len(batch_shape)
        arr_expanded = jnp.expand_dims(arr, axis_idx)
        return jnp.broadcast_to(arr_expanded, target_shape_per_vertex)
    return jnp.broadcast_to(arr, target_shape_per_vertex)


def _broadcast_mesh_param(
    val: Any,
    batch_shape: tuple[int, ...],
    F: int,
    V: int,
    suffix_shape: tuple[int, ...],
    dtype: Any,
) -> jax.Array | None:
    if val is None:
        return None
    arr = jnp.asarray(val, dtype=dtype)
    target_shape_per_vertex = (*batch_shape, F, V, *suffix_shape)
    target_shape_single = (*batch_shape, *suffix_shape)
    if arr.shape == target_shape_per_vertex:
        return arr
    if arr.shape == target_shape_single:
        axis_idx = len(batch_shape)
        arr_expanded = jnp.expand_dims(arr, axis_idx)
        arr_expanded = jnp.expand_dims(arr_expanded, axis_idx)
        return jnp.broadcast_to(arr_expanded, target_shape_per_vertex)
    return jnp.broadcast_to(arr, target_shape_per_vertex)


@jax.jit(static_argnames=["dim", "filled"])
def _compute_single_mesh_properties(
    vertices: jax.Array, faces: jax.Array, dim: int, filled: bool
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # vertices: (V_mesh, dim)
    # faces: (F, V_face)
    A = vertices[faces[:, 0]]
    B = vertices[faces[:, 1]]
    if dim == 3:
        C = vertices[faces[:, 2]]
        if filled:
            V = (1.0 / 6.0) * linalg.dot(linalg.cross(A, B), C)
            COM_tet = (A + B + C) / 4.0

            def outer(x: jax.Array) -> jax.Array:
                return x[:, :, None] * x[:, None, :]

            cov_tets = (V[:, None, None] / 20.0) * (
                outer(A) + outer(B) + outer(C) + outer(A + B + C)
            )

            total_vol = jnp.sum(V)
            safe_vol = jnp.where(total_vol == 0.0, 1.0, total_vol)
            com = jnp.sum(V[:, None] * COM_tet, axis=0) / safe_vol

            total_cov = jnp.sum(cov_tets, axis=0)
            cov_com = total_cov - total_vol * (com[:, None] * com[None, :])

            trace_cov = jnp.trace(cov_com)
            I_tensor = trace_cov * jnp.eye(3) - cov_com

            return com, I_tensor, total_vol
        else:
            cross_prod = linalg.cross(B - A, C - A)
            Area = 0.5 * linalg.norm(cross_prod)
            COM_tri = (A + B + C) / 3.0

            def outer(x: jax.Array) -> jax.Array:
                return x[:, :, None] * x[:, None, :]

            cov_tris = (Area[:, None, None] / 12.0) * (
                outer(A) + outer(B) + outer(C) + outer(A + B + C)
            )

            total_area = jnp.sum(Area)
            safe_area = jnp.where(total_area == 0.0, 1.0, total_area)
            com = jnp.sum(Area[:, None] * COM_tri, axis=0) / safe_area

            total_cov = jnp.sum(cov_tris, axis=0)
            cov_com = total_cov - total_area * (com[:, None] * com[None, :])

            trace_cov = jnp.trace(cov_com)
            I_tensor = trace_cov * jnp.eye(3) - cov_com

            return com, I_tensor, total_area
    else:
        if filled:
            Area = 0.5 * linalg.cross(A, B)[..., 0]
            COM_tri = (A + B) / 3.0

            def outer(x: jax.Array) -> jax.Array:
                return x[:, :, None] * x[:, None, :]

            cov_tris = (Area[:, None, None] / 12.0) * (
                outer(A) + outer(B) + outer(A + B)
            )

            total_area = jnp.sum(Area)
            safe_area = jnp.where(total_area == 0.0, 1.0, total_area)
            com = jnp.sum(Area[:, None] * COM_tri, axis=0) / safe_area

            total_cov = jnp.sum(cov_tris, axis=0)
            cov_com = total_cov - total_area * (com[:, None] * com[None, :])

            I_polar = jnp.trace(cov_com)
            return com, I_polar, total_area
        else:
            Length = linalg.norm(B - A)
            COM_seg = (A + B) / 2.0

            def outer(x: jax.Array) -> jax.Array:
                return x[:, :, None] * x[:, None, :]

            cov_segs = (Length[:, None, None] / 6.0) * (
                outer(A) + outer(B) + outer(A + B)
            )

            total_len = jnp.sum(Length)
            safe_len = jnp.where(total_len == 0.0, 1.0, total_len)
            com = jnp.sum(Length[:, None] * COM_seg, axis=0) / safe_len

            total_cov = jnp.sum(cov_segs, axis=0)
            cov_com = total_cov - total_len * (com[:, None] * com[None, :])

            I_polar = jnp.trace(cov_com)
            return com, I_polar, total_len
