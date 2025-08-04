# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Defines the simulation State.
"""

from dataclasses import dataclass
from typing import Optional, final, Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@final
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class State:
    """
    Represents the complete simulation state for a system of N particles in 2D or 3D.

    Notes
    -----
    `State` is designed to support various data layouts:

    - **Single snapshot:**
        (N, dim) for position-like arrays, (N,) for scalar properties.

    - **Batched states:**
        (B, N, dim) for position-like arrays, (B, N) for scalar properties,
        where 'B' is the batch dimension.

    - **Trajectories:**
        (T, N, dim) for position-like arrays, (T, N) for scalar properties,
        where 'T' is the time/trajectory dimension.

    - **Trajectories of batched states:**
        (T, B, N, dim) for position-like arrays, (T, B, N) for scalar properties.

    Any leading axis beyond `ndim > 4` (for position-like arrays) or `ndim > 3` (for scalar properties)
    is treated as a trajectory.

    The class is `final` and cannot be subclassed.

    Example
    -------
    Creating a simple 2D state for 4 particles:

    >>> import jaxdem as jdem
    >>> import jax.numpy as jnp
    >>>
    >>> positions = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> # All other attributes are optional
    >>> state = jdem.State.create(pos=positions)
    >>>
    >>> print(f"Number of particles (N): {state.N}")
    >>> print(f"Spatial dimension (dim): {state.dim}")
    >>> print(f"Positions:\\n{state.pos}")

    Creating a batched state:

    >>> batched_state = jax.vmap(lambda _: State.create(pos=positions))(jnp.arange(10))
    >>>
    >>> print(f"Batch size: {batched_state.batch_size}") # 10
    >>> print(f"Positions shape: {batched_state.pos.shape}")
    """

    pos: jax.Array
    """
    Array of particle positions. Shape is `(..., N, dim)`.
    """

    vel: jax.Array
    """
    Array of particle velocities. Shape is `(..., N, dim)`.
    """

    accel: jax.Array
    """
    Array of particle accelerations. Shape is `(..., N, dim)`.
    """

    rad: jax.Array
    """
    Array of particle radii. Shape is `(..., N)`.
    """
    mass: jax.Array
    """
    Array of particle masses. Shape is `(..., N)`.
    """

    ID: jax.Array
    """
    Array of unique particle identifiers. Shape is `(..., N)`.
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
        return self.pos.shape[-2]

    @property
    def dim(self) -> int:
        """
        Spatial dimension of the simulation.
        """
        return self.pos.shape[-1]

    @property
    def batch_size(self) -> int:
        """
        Return the batch size of the state.
        """
        return 1 if self.pos.ndim < 3 else self.pos.shape[-3]

    @property
    def is_valid(self) -> bool:
        """
        Check if the internal representation of the State is consistent.

        Verifies that:

            - The spatial dimension (`dim`) is either 2 or 3.

            - All position-like arrays (`pos`, `vel`, `accel`) have the same shape.

            - All scalar-per-particle arrays (`rad`, `mass`, `ID`, `mat_id`, `species_id`, `fixed`) have a shape consistent with `pos.shape[:-1]`.

        Raises
        ------
        AssertionError
            If any shape inconsistency is found.
        """
        valid = self.dim in (2, 3)
        assert valid, f"Simulation dimension (pos.shape[-1]={self.dim}) must be 2 or 3."

        for name in (
            "pos",
            "vel",
            "accel",
        ):
            arr = getattr(self, name)
            valid = valid and self.pos.shape == arr.shape
            assert (
                valid
            ), f"{name}.shape={arr.shape} is not equal to pos.shape={self.pos.shape}."

        for name in (
            "rad",
            "mass",
            "ID",
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

    def __init_subclass__(cls, *args, **kw):
        raise TypeError(f"{State.__name__} is final and cannot be subclassed")

    @staticmethod
    def create(
        pos: ArrayLike,
        *,
        vel: Optional[ArrayLike] = None,
        accel: Optional[ArrayLike] = None,
        rad: Optional[ArrayLike] = None,
        mass: Optional[ArrayLike] = None,
        ID: Optional[ArrayLike] = None,
        mat_id: Optional[ArrayLike] = None,
        species_id: Optional[ArrayLike] = None,
        fixed: Optional[ArrayLike] = None,
    ) -> "State":
        """
        Factory method to create a new :class:`State` instance.

        This method handles default values and ensures consistent array shapes
        for all state attributes.

        Parameters
        ----------
        pos : jax.typing.ArrayLike
            Initial positions of particles.
            Expected shape: `(..., N, dim)`.
        vel : jax.typing.ArrayLike or None, optional
            Initial velocities of particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, dim)`.
        accel : jax.typing.ArrayLike or None, optional
            Initial accelerations of particles. If `None`, defaults to zeros.
            Expected shape: `(..., N, dim)`.
        rad : jax.typing.ArrayLike or None, optional
            Radii of particles. If `None`, defaults to ones.
            Expected shape: `(..., N)`.
        mass : jax.typing.ArrayLike or None, optional
            Masses of particles. If `None`, defaults to ones.
            Expected shape: `(..., N)`.
        ID : jax.typing.ArrayLike or None, optional
            Unique identifiers for particles. If `None`, defaults to
            :func:`jnp.arange(N)`. Expected shape: `(..., N)`.
        mat_id : jax.typing.ArrayLike or None, optional
            Material IDs for particles. If `None`, defaults to zeros.
            Expected shape: `(..., N)`.
        species_id : jax.typing.ArrayLike or None, optional
            Species IDs for particles. If `None`, defaults to zeros.
            Expected shape: `(..., N)`.
        fixed : jax.typing.ArrayLike or None, optional
            Boolean array indicating fixed particles. If `None`, defaults to all `False`.
            Expected shape: `(..., N)`.

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

        pos = jnp.asarray(pos, dtype=float)
        N = pos.shape[-2]

        vel = (
            jnp.zeros_like(pos, dtype=float)
            if vel is None
            else jnp.asarray(vel, dtype=float)
        )
        accel = (
            jnp.zeros_like(pos, dtype=float)
            if accel is None
            else jnp.asarray(accel, dtype=float)
        )
        rad = (
            jnp.ones(pos.shape[:-1], dtype=float)
            if rad is None
            else jnp.asarray(rad, dtype=float)
        )
        mass = (
            jnp.ones(pos.shape[:-1], dtype=float)
            if mass is None
            else jnp.asarray(mass, dtype=float)
        )
        ID = (
            jnp.broadcast_to(jnp.arange(N, dtype=int), pos.shape[:-1])
            if ID is None
            else jnp.asarray(ID, dtype=int)
        )
        mat_id = (
            jnp.zeros(pos.shape[:-1], dtype=int)
            if mat_id is None
            else jnp.asarray(mat_id, dtype=int)
        )
        species_id = (
            jnp.zeros(pos.shape[:-1], dtype=int)
            if species_id is None
            else jnp.asarray(species_id, dtype=int)
        )
        fixed = (
            jnp.zeros(pos.shape[:-1], dtype=bool)
            if fixed is None
            else jnp.asarray(fixed, dtype=bool)
        )

        state = State(
            pos=pos,
            vel=vel,
            accel=accel,
            rad=rad,
            mass=mass,
            ID=ID,
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
        )

        if not state.is_valid:
            raise ValueError(f"State is not valid, state={state}")

        return state

    @staticmethod
    def merge(state1: "State", state2: "State") -> "State":
        """
        Merges two :class:`State` instances into a single new :class:`State`.

        This method concatenates the particles from `state2` onto `state1`.
        Particle IDs in `state2` are shifted to ensure uniqueness in the merged state.

        Parameters
        ----------
        state1 : State
            The first `State` instance. Its particles will appear first in the merged state.
        state2 : State
            The second `State` instance. Its particles will be appended to the first.

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
        >>> state_a = jdem.create(pos=jnp.array([[0.,0.], [1.,1.]]), ID=jnp.array([0, 1]))
        >>> state_b = jdem.State.create(pos=jnp.array([[2.,2.], [3.,3.]]), ID=jnp.array([0, 1]))
        >>> merged_state = State.merge(state_a, state_b)
        >>>
        >>> print(f"Merged state N: {merged_state.N}") # Expected: 4
        >>> print(f"Merged state positions:\\n{merged_state.pos}")
        >>> print(f"Merged state IDs: {merged_state.ID}") # Expected: [0, 1, 2, 3]
        """
        assert state1.is_valid and state2.is_valid, "One of the states is invalid"
        assert state1.dim == state2.dim, f"dim mismatch: {state1.dim} vs {state2.dim}"
        assert (
            state1.batch_size == state2.batch_size
        ), f"batch_size mismatch: {state1.batch_size} vs {state2.batch_size}"
        state2.ID += jnp.max(state1.ID) + 1

        # ----------------- tree-wise concatenation --------------------------
        # Arrays that have the same rank as `pos` (`pos`, `vel`, `accel`) are
        # concatenated along axis -2 (particle axis).  Everything else
        # (`rad`, `mass`, `ID`, `mat_id`, `species_id`, `fixed`) is concatenated along axis -1.
        pos_ndim = state1.pos.ndim

        def cat(a, b):
            axis = -2 if a.ndim == pos_ndim else -1
            return jnp.concatenate((a, b), axis=axis)

        state = jax.tree_util.tree_map(cat, state1, state2)
        if not state.is_valid:
            raise ValueError(f"State is not valid, state={state}")

        return state

    @staticmethod
    def add(
        state: "State",
        pos: ArrayLike,
        *,
        vel: Optional[ArrayLike] = None,
        accel: Optional[ArrayLike] = None,
        rad: Optional[ArrayLike] = None,
        mass: Optional[ArrayLike] = None,
        ID: Optional[ArrayLike] = None,
        mat_id: Optional[ArrayLike] = None,
        species_id: Optional[ArrayLike] = None,
        fixed: Optional[ArrayLike] = None,
    ) -> "State":
        """
        Adds new particles to an existing :class:`State` instance, returning a new `State`.

        Parameters
        ----------
        state : State
            The existing `State` to which particles will be added.
        pos : jax.typing.ArrayLike
            Positions of the new particle(s). Shape `(..., N_new, dim)`.
        vel : jax.typing.ArrayLike or None, optional
            Velocities of the new particle(s). Defaults to zeros.
        accel : jax.typing.ArrayLike or None, optional
            Accelerations of the new particle(s). Defaults to zeros.
        rad : jax.typing.ArrayLike or None, optional
            Radii of the new particle(s). Defaults to ones.
        mass : jax.typing.ArrayLike or None, optional
            Masses of the new particle(s). Defaults to ones.
        ID : jax.typing.ArrayLike or None, optional
            IDs of the new particle(s). If `None`, new IDs are generated.
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
        >>> state = jdem.State.create(jnp.zeros((4, 2)))
        >>> print(f"Original state N: {state.N}, IDs: {state.ID}")
        >>>
        >>> # Add a single new particle
        >>> state_with_added_particle = state.add(state,
        ...                                       pos=jnp.array([[10., 10.]]),
        ...                                       rad=jnp.array([0.5]),
        ...                                       mass=jnp.array([2.0]))
        >>> print(f"New state N: {state_with_added_particle.N}, IDs: {state_with_added_particle.ID}")
        >>> print(f"New particle position: {state_with_added_particle.pos[-1]}")

        Adding multiple new particles:

        >>> state_multiple_added = state.add(state,
        ...                                  pos=jnp.array([[10., 10.], [11., 11.], [12., 12.]]))
        >>> print(f"State with multiple added N: {state_multiple_added.N}, IDs: {state_multiple_added.ID}")
        """
        state2 = State.create(
            pos,
            vel=vel,
            accel=accel,
            rad=rad,
            mass=mass,
            ID=ID,
            mat_id=mat_id,
            species_id=species_id,
            fixed=fixed,
        )
        return State.merge(state, state2)

    @staticmethod
    def stack(states: Sequence["State"]) -> "State":
        """
        Concatenates a sequence of :class:`State` snapshots into a trajectory along axis 0.

        This method is useful for collecting simulation snapshots over time into a
        single `State` object where the leading dimension represents time.

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
        - No ID shifting is performed because the leading axis represents
          **time** (or another batch dimension), not new particles.

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
