# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Defines the simulation State.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, replace
from typing import Optional, final, Sequence, Tuple, TYPE_CHECKING, Any
from functools import partial

from .utils.quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from .materials import MaterialTable


@final
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class State:
    r"""
    Represents the complete simulation state for a system of N particles in 2D or 3D.

    Notes
    -----
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
        `pos.shape = (B, T_1, T_2, ..., T_k, N, dim)` for particle properties,
        and `(B, T_1, T_2, ..., T_k, N)` for scalar properties.

        - The first dimension (i.e., `pos.shape[0]`) is always interpreted as the **batch dimension (`B`)**.

        - All preceding leading dimensions (`T_1, T_2, ... T_k`) are interpreted as **trajectory dimensions**
            and they are **flattened at save time** if there is more than 1 trajectory dimension.

    The class is `final` and cannot be subclassed.

    Example
    -------
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
    """

    pos_p: jax.Array
    """
    Vector relative to the center of mass (pos_p = pos - pos_c) in the principal reference frame. This field should be constant. Shape is `(..., N, dim)`.
    """

    vel: jax.Array
    """
    Array of particle center of mass velocities. Shape is `(..., N, dim)`.
    """

    force: jax.Array
    """
    Array of particle forces. Shape is `(..., N, dim)`.
    """

    q: Quaternion
    """
    Quaternion representing the orientation of the particle.
    """

    angVel: jax.Array
    """
    Array of particle center of mass angular velocities. Shape is `(..., N, 1 | 3)` depending on 2D or 3D simulations.
    """

    torque: jax.Array
    """
    Array of particle torques. Shape is `(..., N, 1 | 3)` depending on 2D or 3D simulations.
    """

    rad: jax.Array
    """
    Array of particle radii. Shape is `(..., N)`.
    """

    volume: jax.Array
    """
    Array of particle volumes (or areas if 2D). Shape is `(..., N)`.
    """

    mass: jax.Array
    """
    Array of particle masses. Shape is `(..., N)`.
    """

    inertia: jax.Array
    """
    Inertia tensor in the principal axis frame `(..., N, 1 | 3)` depending on 2D or 3D simulations.
    """

    clump_ID: jax.Array
    """
    Array of clump identifiers. Bodies with the same clump_ID are treated as part of the same rigid body. Shape is `(..., N)`.
    IDs need to be between 0 and N.
    """

    deformable_ID: jax.Array
    """
    Array of deformable particle identifiers. Spheres (nodes) with the same deformable_ID are treated as part of the same deformable particle
    for collision masking purposes. Shape is `(..., N)`. IDs need to be between 0 and N.
    """

    unique_ID: jax.Array
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
    """

    @property
    def N(self) -> int:
        """
        Number of particles in the state.
        """
        return self.pos_c.shape[-2]

    @property
    def dim(self) -> int:
        """
        Spatial dimension of the simulation.
        """
        return self.pos_c.shape[-1]

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Number of particles in the state.
        """
        return self.pos_c.shape

    @property
    def batch_size(self) -> int:
        """
        Return the batch size of the state.
        """
        return 1 if self.pos_c.ndim < 3 else self.pos_c.shape[-3]

    @property
    @partial(jax.named_call, name="State.pos")
    def pos(self) -> jax.Array:
        """
        Returns the position of each sphere in the state.
        pos_c is the center of mass
        pos_p is the vector relative to the center of mass
        such that pos = pos_c = pos_p in the principal reference frame.
        Therefore, pos_p needs to be transformed to the lab frame.
        """
        return self.pos_c + self.q.rotate(self.q, self.pos_p)

    @property
    @partial(jax.named_call, name="State.is_valid")
    def is_valid(self) -> bool:
        """
        Check if the internal representation of the State is consistent.

        Verifies that:

            - The spatial dimension (`dim`) is either 2 or 3.

            - All position-like arrays (`pos_c`, `pos_p`, `vel`, `force`) have the same shape.

            - All angular-like arrays (`angVel`, `torque`, `inertia`) have the same shape.

            - All scalar-per-particle arrays (`rad`, `mass`, `clump_ID`, `deformable_ID`, `mat_id`, `species_id`, `fixed`) have a shape consistent with `pos.shape[:-1]`.

        Raises
        ------
        AssertionError
            If any shape inconsistency is found.
        """
        valid = self.dim in (2, 3)
        assert valid, f"Simulation dimension (pos.shape[-1]={self.dim}) must be 2 or 3."

        for name in (
            "pos",
            "pos_c",
            "pos_p",
            "vel",
            "force",
        ):
            arr = getattr(self, name)
            valid = valid and self.pos.shape == arr.shape
            assert (
                valid
            ), f"{name}.shape={arr.shape} is not equal to pos.shape={self.pos.shape}."

        ang_dim = 1 if self.dim == 2 else 3
        expected_ang_shape = self.pos.shape[:-1] + (ang_dim,)
        for name in ("angVel", "torque", "inertia"):
            arr = getattr(self, name)
            valid = valid and arr.shape == expected_ang_shape
            assert (
                valid
            ), f"{name}.shape={arr.shape} is not equal to {expected_ang_shape}."

        for name in (
            "rad",
            "mass",
            "volume",
            "clump_ID",
            "deformable_ID",
            "mat_id",
            "species_id",
            "fixed",
        ):
            arr = getattr(self, name)
            valid = valid and self.pos.shape[:-1] == arr.shape
            assert (
                valid
            ), f"{name}.shape={arr.shape} is not equal to pos.shape[:-1]={self.pos.shape[:-1]}."

        return valid

    @staticmethod
    @partial(jax.named_call, name="State.create")
    def create(
        pos: ArrayLike,
        *,
        pos_p: Optional[ArrayLike] = None,
        vel: Optional[ArrayLike] = None,
        force: Optional[ArrayLike] = None,
        q: Optional[Quaternion] | Optional[ArrayLike] = None,
        angVel: Optional[ArrayLike] = None,
        torque: Optional[ArrayLike] = None,
        rad: Optional[ArrayLike] = None,
        volume: Optional[ArrayLike] = None,
        mass: Optional[ArrayLike] = None,
        inertia: Optional[ArrayLike] = None,
        clump_ID: Optional[ArrayLike] = None,
        deformable_ID: Optional[ArrayLike] = None,
        mat_id: Optional[ArrayLike] = None,
        species_id: Optional[ArrayLike] = None,
        fixed: Optional[ArrayLike] = None,
        mat_table: Optional["MaterialTable"] = None,
    ) -> "State":
        r"""
        Factory method to create a new :class:`State` instance.

        This method handles default values and ensures consistent array shapes
        for all state attributes.

        Parameters
        ----------
        pos : jax.typing.ArrayLike
            Array of particle center of mass positions, equivalent to state.pos_c.
            Expected shape: `(..., N, dim)`.
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
        angVel : jax.typing.ArrayLike or None, optional
            Initial angular velocities of particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, 1)` in 2D or `(..., N, 3)` in 3D.
        torque : jax.typing.ArrayLike or None, optional
            Initial torques on particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, 1)` in 2D or `(..., N, 3)` in 3D.
        rad : jax.typing.ArrayLike or None, optional
            Radii of particles. If `None`, defaults to ones.
            Expected shape: `(..., N)`.
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
        clump_ID : jax.typing.ArrayLike or None, optional
            Unique identifiers for clumps. If `None`, defaults to
            :func:`jnp.arange`. Expected shape: `(..., N)`.
        deformable_ID : jax.typing.ArrayLike or None, optional
            Unique identifiers for deformable particles. If `None`, defaults to
            :func:`jnp.arange`. Expected shape: `(..., N)`.
        mat_id : jax.typing.ArrayLike or None, optional
            Material IDs for particles. If `None`, defaults to zeros.
            Expected shape: `(..., N)`.
        species_id : jax.typing.ArrayLike or None, optional
            Species IDs for particles. If `None`, defaults to zeros.
            Expected shape: `(..., N)`.
        fixed : jax.typing.ArrayLike or None, optional
            Boolean array indicating fixed particles. If `None`, defaults to all `False`.
            Expected shape: `(..., N)`.
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
        pos_c = jnp.asarray(pos, dtype=float)
        N = pos_c.shape[-2]
        dim = pos_c.shape[-1]
        ang_dim = 1 if dim == 2 else 3
        ang_shape = pos_c.shape[:-1] + (ang_dim,)

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
            q = Quaternion.create(
                jnp.ones(pos_c.shape[:-1] + (1,), dtype=float),
                jnp.zeros(pos_c.shape[:-1] + (3,), dtype=float),
            )
        elif isinstance(q, Quaternion):
            q = q
        else:
            # If it's ArrayLike, assume (..., 4) with [w, x, y, z]
            q_arr = jnp.asarray(q, dtype=float)
            q = Quaternion.create(
                w=q_arr[..., 0:1],
                xyz=q_arr[..., 1:],
            )

        angVel = (
            jnp.zeros(ang_shape, dtype=float)
            if angVel is None
            else jnp.asarray(angVel, dtype=float)
        )
        torque = (
            jnp.zeros_like(angVel, dtype=float)
            if torque is None
            else jnp.asarray(torque, dtype=float)
        )

        rad = (
            jnp.ones(pos_c.shape[:-1], dtype=float)
            if rad is None
            else jnp.asarray(rad, dtype=float)
        )
        volume = (
            jnp.exp(
                0.5 * dim * jnp.log(jnp.pi)
                + dim * jnp.log(rad)
                - jax.scipy.special.gammaln(0.5 * dim + 1.0)
            )
            if volume is None
            else jnp.asarray(volume, dtype=float)
        )

        clump_ID = (
            jnp.broadcast_to(jnp.arange(N, dtype=int), pos_c.shape[:-1])
            if clump_ID is None
            else jnp.asarray(clump_ID, dtype=int)
        )
        deformable_ID = (
            jnp.broadcast_to(jnp.arange(N, dtype=int), pos_c.shape[:-1])
            if deformable_ID is None
            else jnp.asarray(deformable_ID, dtype=int)
        )

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
            inertia_scalar[..., None] * jnp.ones_like(angVel, dtype=float)
            if inertia is None
            else jnp.asarray(inertia, dtype=float)
        )

        _, clump_ID = jnp.unique(clump_ID, return_inverse=True, size=N)
        _, deformable_ID = jnp.unique(deformable_ID, return_inverse=True, size=N)

        state = State(
            pos_c=pos_c,
            pos_p=pos_p,
            vel=vel,
            force=force,
            q=q,
            angVel=angVel,
            torque=torque,
            rad=rad,
            volume=volume,
            mass=mass,
            inertia=inertia,
            clump_ID=jnp.asarray(clump_ID),
            deformable_ID=jnp.asarray(deformable_ID),
            unique_ID=jnp.arange(N, dtype=int),
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
        )

        if not state.is_valid:
            raise ValueError(f"State is not valid, state={state}")

        return state

    @staticmethod
    @partial(jax.named_call, name="State.merge")
    def merge(state1: "State", state2: "State" | Sequence["State"]) -> "State":
        """
        Merges multiple :class:`State` instances into a single new :class:`State`.

        This method concatenates the particles from the provided state(s) onto `state1`.
        Particle clump_IDs, deformable_IDs, and unique_IDs are shifted to ensure
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
        >>> state_a = jdem.State.create(pos=jnp.array([[0.0, 0.0], [1.0, 1.0]]), clump_ID=jnp.array([0, 1]))
        >>> state_b = jdem.State.create(pos=jnp.array([[2.0, 2.0], [3.0, 3.0]]), clump_ID=jnp.array([0, 1]))
        >>> merged_state = jdem.State.merge(state_a, [state_b, state_b, state_b])
        >>> merged_state = jdem.State.merge(state_a, state_b)
        >>>
        >>> print(f"Merged state N: {merged_state.N}")  # Expected: 4
        >>> print(f"Merged state positions:\n{merged_state.pos}")
        >>> print(f"Merged state clump_IDs: {merged_state.clump_ID}")  # Expected: [0, 1, 2, 3]

        """
        states_to_merge = [state2] if isinstance(state2, State) else list(state2)
        current_state = state1
        pos_ndim = current_state.pos.ndim

        for next_state in states_to_merge:
            assert (
                current_state.is_valid and next_state.is_valid
            ), "Invalid state detected"
            assert current_state.dim == next_state.dim, "Dimension mismatch"
            assert (
                current_state.batch_size == next_state.batch_size
            ), "Batch size mismatch"

            # Calculate offsets based on current_state's max IDs
            c_offset = jnp.max(current_state.clump_ID) + 1
            d_offset = jnp.max(current_state.deformable_ID) + 1
            u_offset = jnp.max(current_state.unique_ID) + 1

            # Apply offsets to the next_state
            next_state = replace(
                next_state,
                clump_ID=next_state.clump_ID + c_offset,
                deformable_ID=next_state.deformable_ID + d_offset,
                unique_ID=next_state.unique_ID + u_offset,
            )

            # Define concatenation logic per leaf
            def cat(a: jax.Array, b: jax.Array) -> jax.Array:
                # Particles are at -2 for vector fields, -1 for scalars
                axis = -2 if a.ndim == pos_ndim else -1
                return jnp.concatenate((a, b), axis=axis)

            current_state = jax.tree_util.tree_map(cat, current_state, next_state)

        if not current_state.is_valid:
            raise ValueError(f"Merged state is not valid")

        return current_state

    @staticmethod
    @partial(jax.named_call, name="State.add")
    def add(
        state: "State",
        pos: ArrayLike,
        *,
        pos_p: Optional[ArrayLike] = None,
        vel: Optional[ArrayLike] = None,
        force: Optional[ArrayLike] = None,
        q: Optional[Quaternion] | Optional[ArrayLike] = None,
        angVel: Optional[ArrayLike] = None,
        torque: Optional[ArrayLike] = None,
        rad: Optional[ArrayLike] = None,
        volume: Optional[ArrayLike] = None,
        mass: Optional[ArrayLike] = None,
        inertia: Optional[ArrayLike] = None,
        clump_ID: Optional[ArrayLike] = None,
        deformable_ID: Optional[ArrayLike] = None,
        mat_id: Optional[ArrayLike] = None,
        species_id: Optional[ArrayLike] = None,
        fixed: Optional[ArrayLike] = None,
        mat_table: Optional["MaterialTable"] = None,
    ) -> "State":
        """
        Adds new particles to an existing :class:`State` instance, returning a new `State`.

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
        angVel : jax.typing.ArrayLike or None, optional
            Angular velocities of the new particle(s). Defaults to zeros.
        torque : jax.typing.ArrayLike or None, optional
            Torques of the new particle(s). Defaults to zeros.
        rad : jax.typing.ArrayLike or None, optional
            Radii of the new particle(s). Defaults to ones.
        volume : jax.typing.ArrayLike or None, optional
            Volume of the new particle(s) (or area in 2D). Defaults to hypersphere volumes of the radii.
        mass : jax.typing.ArrayLike or None, optional
            Masses of the new particle(s). Defaults to ones. Ignored when a
            `mat_table` is provided.
        inertia : jax.typing.ArrayLike or None, optional
            Moments of inertia of the new particle(s). Defaults to solid disks (2D)
            or spheres (3D).
        clump_ID : jax.typing.ArrayLike or None, optional
            clump_IDs of the new clump(s). If `None`, new IDs are generated.
        deformable_ID : jax.typing.ArrayLike or None, optional
            Unique identifiers for deformable particles. If `None`, defaults to
            :func:`jnp.arange`. Expected shape: `(..., N)`.
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
        >>> print(f"Original state N: {state.N}, clump_IDs: {state.clump_ID}")
        >>>
        >>> # Add a single new particle
        >>> state_with_added_particle = jdem.State.add(
        ...     state,
        ...     pos=jnp.array([[10.0, 10.0]]),
        ...     rad=jnp.array([0.5]),
        ...     mass=jnp.array([2.0]),
        ... )
        >>> print(f"New state N: {state_with_added_particle.N}, clump_IDs: {state_with_added_particle.clump_ID}")
        >>> print(f"New particle position: {state_with_added_particle.pos[-1]}")
        >>>
        >>> # Add multiple new particles
        >>> state_multiple_added = jdem.State.add(
        ...     state,
        ...     pos=jnp.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]),
        ... )
        >>> print(f"State with multiple added N: {state_multiple_added.N}, clump_IDs: {state_multiple_added.clump_ID}")

        """
        state2 = State.create(
            pos=pos,
            pos_p=pos_p,
            vel=vel,
            force=force,
            q=q,
            angVel=angVel,
            torque=torque,
            rad=rad,
            volume=volume,
            mass=mass,
            inertia=inertia,
            clump_ID=clump_ID,
            deformable_ID=deformable_ID,
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
        )
        return State.merge(state, state2)

    @staticmethod
    @partial(jax.named_call, name="State.stack")
    def stack(states: Sequence["State"]) -> "State":
        """
        Concatenates a sequence of :class:`State` snapshots into a trajectory or batch along axis 0.

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
        ----
        - No clump_ID shifting is performed because the leading axis represents **time** (or another batch dimension), not new particles.

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
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

        if not stacked.is_valid:
            raise ValueError("stacked State is not valid")

        return stacked

    @staticmethod
    @partial(jax.named_call, name="State.unstack")
    def unstack(state: "State") -> list["State"]:
        """
        Split a stacked/batched :class:`State` along the leading axis into a Python list.

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
        return [jax.tree_util.tree_map(lambda x, i=i: x[i], state) for i in range(n)]

    @staticmethod
    @partial(jax.named_call, name="State.add_clump")
    def add_clump(
        state: "State",
        pos: ArrayLike,
        *,
        pos_p: Optional[ArrayLike] = None,
        vel: Optional[ArrayLike] = None,
        force: Optional[ArrayLike] = None,
        q: Optional[Quaternion] | Optional[ArrayLike] = None,
        angVel: Optional[ArrayLike] = None,
        torque: Optional[ArrayLike] = None,
        rad: Optional[ArrayLike] = None,
        volume: Optional[ArrayLike] = None,
        mass: Optional[ArrayLike] = None,
        inertia: Optional[ArrayLike] = None,
        deformable_ID: Optional[ArrayLike] = None,
        mat_id: Optional[ArrayLike] = None,
        species_id: Optional[ArrayLike] = None,
        fixed: Optional[ArrayLike] = None,
    ) -> "State":
        """
        Adds a new clump consisting of multiple spheres to an existing State.
        Rigid body properties (velocity, mass, material, etc.) are broadcasted
        to all spheres in the new clump. The only per sphere properties that
        vary in a rigid body are pos_c, pos_p, and rad.

        TO DO: broadcast the quaternion

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
        angVel : jax.typing.ArrayLike or None, optional
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
        deformable_ID : jax.typing.ArrayLike or None, optional
            Unique identifiers for deformable particles. If `None`, defaults to
            :func:`jnp.arange`. Expected shape: `(..., N)`.
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
        pos_c = jnp.asarray(pos)
        dim = pos_c.shape[-1]
        ang_dim = 1 if dim == 2 else 3

        def broadcast_field(
            val: ArrayLike | None, shape: Tuple[int, ...], dtype: Any
        ) -> None | jax.Array:
            return (
                jnp.broadcast_to(jnp.asarray(val, dtype=dtype), shape)
                if val is not None
                else None
            )

        pos_p = jnp.asarray(pos_p) if pos_p is not None else None
        rad = jnp.asarray(rad) if rad is not None else None

        vel = broadcast_field(vel, pos_c.shape, float)
        force = broadcast_field(force, pos_c.shape, float)

        angVel = broadcast_field(angVel, pos_c.shape[:-1] + (ang_dim,), float)
        torque = broadcast_field(torque, pos_c.shape[:-1] + (ang_dim,), float)
        inertia = broadcast_field(inertia, pos_c.shape[:-1] + (ang_dim,), float)

        if q is not None:
            if isinstance(q, Quaternion):
                w = jnp.broadcast_to(
                    jnp.asarray(q.w, dtype=float), pos_c.shape[:-1] + (1,)
                )
                xyz = jnp.broadcast_to(
                    jnp.asarray(q.xyz, dtype=float), pos_c.shape[:-1] + (3,)
                )
                q = Quaternion(w=w, xyz=xyz)
            else:
                # If passed as ArrayLike (w, x, y, z), broadcast to (..., N, 4)
                q = broadcast_field(q, pos_c.shape[:-1] + (4,), float)

        mass = broadcast_field(mass, pos_c.shape[:-1], float)
        volume = broadcast_field(volume, pos_c.shape[:-1], float)

        clump_ID = broadcast_field(0, pos_c.shape[:-1], int)
        mat_id = broadcast_field(mat_id, pos_c.shape[:-1], int)
        species_id = broadcast_field(species_id, pos_c.shape[:-1], int)
        fixed = broadcast_field(fixed, pos_c.shape[:-1], bool)

        state2 = State.create(
            pos=pos_c,
            pos_p=pos_p,
            vel=vel,
            force=force,
            q=q,
            angVel=angVel,
            torque=torque,
            rad=rad,
            volume=volume,
            mass=mass,
            inertia=inertia,
            clump_ID=clump_ID,
            deformable_ID=deformable_ID,
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
        )
        return State.merge(state, state2)
