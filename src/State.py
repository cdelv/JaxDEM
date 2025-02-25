# This file is part of the JaxDEM library. For more information and source code
# availability, visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions.

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from functools import partial

from jaxdem import Sphere


@dataclass
class State:
    """
    Container for the dynamic state of a simulation particle.

    Attributes
    ----------
    dim : jnp.ndarray
        The dimensionality of the simulation domain (2D or 3D).
    pos : jnp.ndarray
        The position vector of the particle (length = dim).
    vel : jnp.ndarray
        The velocity vector of the particle (length = dim).
    accel : jnp.ndarray
        The acceleration vector of the particle (length = dim).
    mass : jnp.ndarray
        The mass of the particle.
    """
    dim: jnp.ndarray = jnp.asarray(3, dtype=int)
    pos: jnp.ndarray = field(default=None)
    vel: jnp.ndarray = field(default=None)
    accel: jnp.ndarray = field(default=None)
    mass: jnp.ndarray = jnp.asarray(1.0, dtype=float)

    def __post_init__(self):
        """
        Ensures that mass and dimension are JAX arrays of the correct type
        and sets default zeros for pos, vel, or accel if they are not
        explicitly provided. Also validates the shapes of pos, vel, and
        accel if they are provided.
        """
        self.mass = jnp.asarray(self.mass, dtype=float)
        self.dim = jnp.asarray(self.dim, dtype=int)

        # Check or set position
        if self.pos is None:
            self.pos = jnp.zeros(int(self.dim), dtype=float)
        else:
            self.pos = jnp.asarray(self.pos, dtype=float)
            if self.pos.shape[0] != int(self.dim):
                raise ValueError(
                    f"pos must be a vector of length {self.dim}, got shape {self.pos.shape}"
                )

        # Check or set velocity
        if self.vel is None:
            self.vel = jnp.zeros(int(self.dim), dtype=float)
        else:
            self.vel = jnp.asarray(self.vel, dtype=float)
            if self.vel.shape[0] != int(self.dim):
                raise ValueError(
                    f"vel must be a vector of length {self.dim}, got shape {self.vel.shape}"
                )

        # Check or set acceleration
        if self.accel is None:
            self.accel = jnp.zeros(int(self.dim), dtype=float)
        else:
            self.accel = jnp.asarray(self.accel, dtype=float)
            if self.accel.shape[0] != int(self.dim):
                raise ValueError(
                    f"accel must be a vector of length {self.dim}, got shape {self.accel.shape}"
                )


class StateContainer:
    """
    Consolidated memory container for sphere particles using a Structure-of-Arrays (SoA) layout.

    This container stores the dynamic state and geometric properties of spheres
    in a unified manner. It supports efficient batch updates of particle states
    (positions, velocities, accelerations) and can return or set these states
    in a tuple (pos, vel, accel) format. It also provides random access to a
    single particle's state via the `getIthState` method.

    Attributes
    ----------
    _dim : jnp.ndarray
        The simulation domain's dimensionality as a JAX DeviceArray (2 or 3).
    _maxSpheres : jnp.ndarray
        The maximum count of sphere particles that can be stored (allocated).
    _nSpheres : jnp.ndarray
        The current number of active sphere particles in this container.
    _pos : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing particle positions.
    _vel : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing particle velocities.
    _accel : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing particle accelerations.
    _mass : jnp.ndarray
        A 1D array of length maxSpheres storing particle masses.
    _rad : jnp.ndarray
        A 1D array of length maxSpheres storing sphere radii.
    """

    def __init__(self, dim: int, maxSpheres: int = 1):
        """
        Initialize the StateContainer.

        Parameters
        ----------
        dim : int
            The dimensionality of the simulation domain (must be 2 or 3).
        maxSpheres : int, optional
            The maximum number of sphere particles to preallocate memory for.
            Default is 1.

        Raises
        ------
        ValueError
            If `dim` is not 2 or 3, or if `maxSpheres` is negative.
        """
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. Got dim={dim}")
        self._dim = jnp.asarray(dim, dtype=int)

        if maxSpheres < 0:
            raise ValueError(
                f"maxSpheres must be a non-negative integer. Got maxSpheres={maxSpheres}"
            )
        self._maxSpheres = jnp.asarray(maxSpheres, dtype=int)
        self._nSpheres = jnp.asarray(0, dtype=int)

        # State data
        self._pos = jnp.zeros((self._maxSpheres, int(self._dim)), dtype=float)
        self._vel = jnp.zeros((self._maxSpheres, int(self._dim)), dtype=float)
        self._accel = jnp.zeros((self._maxSpheres, int(self._dim)), dtype=float)
        self._mass = jnp.zeros(self._maxSpheres, dtype=float)

        # Shape data
        self._rad = jnp.zeros(self._maxSpheres, dtype=float)

    def getIthState(self, i: int) -> State:
        """
        Retrieve the State of the i-th sphere in the container.

        Parameters
        ----------
        i : int
            The index of the sphere whose state is to be retrieved. Must be
            in the range [0, nSpheres - 1].

        Returns
        -------
        State
            A `State` object containing the position, velocity, acceleration,
            and mass of the i-th sphere. The dimension is also carried over.

        Raises
        ------
        IndexError
            If `i` is out of range (i >= self._nSpheres).
        """
        if i >= self._nSpheres:
            raise IndexError("Index out of range.")

        return State(
            dim=self._dim,
            pos=self._pos[i],
            vel=self._vel[i],
            accel=self._accel[i],
            mass=self._mass[i],
        )

    def setState(self, state):
        """
        Update all spheres' states in bulk.

        Parameters
        ----------
        state : tuple of jnp.ndarray
            A tuple `(pos, vel, accel)` where each is a JAX array:
            - pos.shape = (nSpheres, dim)
            - vel.shape = (nSpheres, dim)
            - accel.shape = (nSpheres, dim)

        Notes
        -----
        This method is typically called after a time-integration step where the
        positions, velocities, and accelerations of all spheres have been
        updated. JIT compilation requires static types, meaning that we cant modify self inside a JITed function.
        """
        self._pos[:self._nSpheres], self._vel[:self._nSpheres], self._accel[:self._nSpheres] = state

    @partial(jax.jit, static_argnames=("self",))
    def getState(self):
        """
        Retrieve a tuple of all spheres' states in bulk.

        Returns
        -------
        tuple of jnp.ndarray
            A tuple `(pos, vel, accel)` where each is a JAX DeviceArray:
            - pos.shape = (maxSpheres, dim)
            - vel.shape = (maxSpheres, dim)
            - accel.shape = (maxSpheres, dim)

        Notes
        -----
        - Only the first `nSpheres` rows of these arrays are meaningful, since
          `_nSpheres` may be less than `_maxSpheres`.
        """
        return (self._pos[:self._nSpheres], self._vel[:self._nSpheres], self._accel[:self._nSpheres])



