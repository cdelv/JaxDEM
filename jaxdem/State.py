import jax
import jax.numpy as jnp
from jax.experimental import checkify

from dataclasses import dataclass
from typing import Optional, Union, List

from jaxdem.Shape import sphere

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class state:
    """
    Container for the state of the simulation.

    Attributes
    ----------
    dim : int
        The dimensionality of the simulation domain (2 or 3).
    N : int
        The number of particles in the simulation.
    pos : jnp.ndarray
        The position vectors of the particles with shape (N, dim).
    vel : jnp.ndarray
        The velocity vectors of the particles with shape (N, dim).
    accel : jnp.ndarray
        The acceleration vectors of the particles with shape (N, dim).
    rad : jnp.ndarray
        The radii of the particles with shape (N,).
    mass : jnp.ndarray
        The masses of the particles with shape (N,).

    Notes
    -----
    Users should use state.create() to create their states. This function ensures all data is consistent.
    When calling state() directly, all parameters must be passed correctly; there is no check for compatibility with jax.vmap.
    state.create() can be called inside @jax.jit functions with static arguments.
    """
    dim: int = 3
    N: int = 1
    pos: jnp.ndarray = jnp.zeros((1, 3))
    vel: jnp.ndarray = jnp.zeros((1, 3))
    accel: jnp.ndarray = jnp.zeros((1, 3))
    rad: jnp.ndarray = jnp.ones(1)
    mass: jnp.ndarray = jnp.ones(1)

    @property
    def is_valid(self) -> bool:
        """
        Validate that the state has the expected structure.

        Returns
        -------
        bool
            True if the state is valid; otherwise, False.
            When assertions are active, any invalid structure will raise an AssertionError.
        """
        ndim = self.pos.ndim
        assert ndim >= 2, f"pos.ndim must be at least 2, got {ndim}"
        valid = True

        for field_name in ['N', 'dim']:
            value = getattr(self, field_name)
            assert isinstance(value, int) or isinstance(value, jnp.ndarray), f"'{field_name}' should be an int, got {type(value)}"
            valid = valid and (isinstance(value, int) or isinstance(value, jnp.ndarray))

        if ndim == 2:
            # Single state: pos shape should be (N, dim)
            for field_name in ['pos', 'vel', 'accel']:
                arr = getattr(self, field_name)
                assert arr.shape[0] == self.N, f"'{field_name}' first dimension should be {self.N}, got {arr.shape[0]}"
                valid = valid and (arr.shape[0] == self.N)
                assert arr.shape[1] == self.dim, f"'{field_name}' second dimension should be {self.dim}, got {arr.shape[1]}"
                valid = valid and (arr.shape[1] == self.dim)
            for field_name in ['rad', 'mass']:
                arr = getattr(self, field_name)
                assert arr.shape[0] == self.N, f"'{field_name}' should have length {self.N}, got {arr.shape[0]}"
                valid = valid and (arr.shape[0] == self.N)
        elif ndim == 3:
            # Batched state: pos shape should be (B, N, dim) where B is batch size.
            B = self.pos.shape[0]
            for field_name in ['pos', 'vel', 'accel']:
                arr = getattr(self, field_name)
                assert jnp.all(self.N == arr.shape[1]), f"'{field_name}' second dimension should be {self.N}, got {arr.shape[1]}"
                valid = valid and jnp.all(self.N == arr.shape[1])
                assert jnp.all(self.dim == arr.shape[2]), f"'{field_name}' third dimension should be {self.dim}, got {arr.shape[2]}"
                valid = valid and jnp.all(self.dim == arr.shape[2])
            for field_name in ['rad', 'mass']:
                arr = getattr(self, field_name)
                assert jnp.all(self.N == arr.shape[1]), f"'{field_name}' second dimension should be {self.N}, got {arr.shape[1]}"
                valid = valid and jnp.all(self.N == arr.shape[1])
        return bool(valid)

    @staticmethod
    def create(
        dim: Optional[int] = None,
        N: Optional[int] = None,
        pos: Optional[jnp.ndarray] = None,
        vel: Optional[jnp.ndarray] = None,
        accel: Optional[jnp.ndarray] = None,
        rad: Optional[jnp.ndarray] = None,
        mass: Optional[jnp.ndarray] = None
    ) -> 'state':
        """
        Factory method to create and initialize a new state instance.

        Parameters
        ----------
        dim : Optional[int], default 3
            The dimensionality of the simulation domain (2 or 3). If not provided,
            it is inferred from the 'pos' array or defaults to 3 if 'pos' is None.
        N : Optional[int], default 1
            The number of particles. If not provided but 'pos' is non-empty, N is inferred
            from pos.shape[0]. Otherwise, defaults to 1.
        pos : jnp.ndarray, optional
            The position vectors of the particles with shape (N, dim). If not provided,
            defaults to an array of zeros with shape (N, dim).
        vel : jnp.ndarray, optional
            The velocity vectors of the particles with shape (N, dim). If not provided,
            defaults to an array of zeros with shape (N, dim).
        accel : jnp.ndarray, optional
            The acceleration vectors of the particles with shape (N, dim). If not provided,
            defaults to an array of zeros with shape (N, dim).
        rad : jnp.ndarray, optional
            The radii of the particles with shape (N,). If not provided, defaults to an array
            of ones with shape (N,).
        mass : jnp.ndarray, optional
            The masses of the particles with shape (N,). If not provided, defaults to an array
            of ones with shape (N,).

        Returns
        -------
        state
            A new state instance with the provided (or default) parameters, with all fields validated.

        Notes
        -----
        - If N is not provided but 'pos' is non-empty, N is inferred from pos.shape[0].
        - If 'dim' is not provided, it is inferred from pos.shape[1] (defaulting to 3 if pos is None).
        - If any of the vector fields (pos, vel, accel) are provided empty,
          they are reinitialized to arrays of zeros with shape (N, dim).
        - If either 'rad' or 'mass' is provided as an empty array, they are reinitialized to arrays of ones with shape (N,).
        - An AssertionError is raised if 'dim' is not 2 or 3 (when 'dim' is a concrete integer),
          and a ValueError is raised if any field does not have the expected shape.
        """
        if N is None:
            if pos is not None:
                N = pos.shape[0]
            else:
                N = 1
        assert N > 0, f"N must be possitive, but got {N}"
        if dim is None:
            if pos is not None:
                dim = pos.shape[1]
            else:
                dim = 3
        assert dim in (2, 3), f"dim must be 2 or 3, but got {dim}"
        if pos is None:
            pos = jnp.zeros((N, dim))
        pos = jnp.asarray(pos, dtype=float)
        if vel is None:
            vel = jnp.zeros((N, dim))
        vel = jnp.asarray(vel, dtype=float)
        if accel is None:
            accel = jnp.zeros((N, dim))
        accel = jnp.asarray(accel, dtype=float)
        if rad is None:
            rad = jnp.ones(N)
        rad = jnp.asarray(rad, dtype=float)
        if mass is None:
            mass = jnp.ones(N)
        mass = jnp.asarray(mass, dtype=float)

        s = state(dim=dim, N=N, pos=pos, vel=vel, accel=accel, rad=rad, mass=mass)
        if not s.is_valid:
            raise ValueError(f"The state is not valid, got {s}")
        return s

    @staticmethod
    def add_sphere(
        current_state: 'state',
        spheres: Union['sphere', List['sphere']],
        sphere_state: Optional['state'] = None
    ) -> 'state':
        """
        Append sphere data to a simulation state.

        Parameters
        ----------
        current_state : state
            The simulation state to which the sphere data will be added.
        spheres : sphere or list of sphere
            A single sphere instance or a list of sphere instances.
        sphere_state : state, optional
            The state of the new spheres. If None, a new state is created from the provided
            spheres with N equal to the number of spheres. Otherwise, sphere_state.N must match
            the number of spheres provided.

        Returns
        -------
        state
            A new state instance with the sphere data appended.
        """
        assert isinstance(current_state, state), f"current_state must be a state instance, got {current_state}"
        assert current_state.is_valid, f"current_state must be a valid state, got {current_state}"
        assert isinstance(spheres, sphere) or (isinstance(spheres, list) and all(isinstance(s, sphere) for s in spheres)), \
            f"spheres must be a sphere instance or list of sphere, got {spheres}"
        if not isinstance(spheres, list):
            spheres = [spheres]
        if sphere_state is None:
            sphere_state = state.create(dim=current_state.dim, N=len(spheres))
        assert isinstance(sphere_state, state), f"sphere_state must be a state instance, got {sphere_state}"
        assert sphere_state.is_valid, f"sphere_state must be a valid state, got {sphere_state}"
        assert sphere_state.N == len(spheres), \
            f"sphere_state.N = {sphere_state.N} must equal the number of spheres provided (len(spheres) = {len(spheres)})"
        return state(
            dim=current_state.dim,
            N=current_state.N + sphere_state.N,
            pos=jnp.append(current_state.pos, sphere_state.pos, axis=0),
            vel=jnp.append(current_state.vel, sphere_state.vel, axis=0),
            accel=jnp.append(current_state.accel, sphere_state.accel, axis=0),
            rad=jnp.append(current_state.rad, sphere_state.rad, axis=0),
            mass=jnp.append(current_state.mass, sphere_state.mass, axis=0),
        )



