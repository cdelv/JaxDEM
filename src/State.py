# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax 
import jax.numpy as jnp

from dataclasses import dataclass, field

from jaxdem import Sphere

@dataclass
class State:
    """
    Container for the dynamic state of a simulation particle.
    
    Attributes
    ----------
    dim : jnp.ndarray
        The dimensionality of the simulation (e.g., 2 or 3).
    pos : jnp.ndarray
        The position vector of the particle.
    vel : jnp.ndarray
        The velocity vector of the particle.
    accel : jnp.ndarray
        The acceleration vector of the particle.
    mass : jnp.ndarray
        The mass of the particle.
    """
    dim: jnp.ndarray = jnp.asarray(3, dtype=int)
    pos: jnp.ndarray = field(default=None)
    vel: jnp.ndarray = field(default=None)
    accel: jnp.ndarray = field(default=None)
    mass: jnp.ndarray = jnp.asarray(1.0, dtype=float)

    def __post_init__(self):
        self.mass = jnp.asarray(self.mass, dtype=float)
        self.dim = jnp.asarray(self.dim, dtype=int)
        
        if self.pos is None:
            self.pos = jnp.zeros(self.dim, dtype=float)
        else:
            self.pos = jnp.asarray(self.pos, dtype=float)
            if self.pos.shape[0] != self.dim:
                raise ValueError(f"pos must be a vector of length {self.dim}, got shape {self.pos.shape}")

        if self.vel is None:
            self.vel = jnp.zeros(self.dim, dtype=float)
        else:
            self.vel = jnp.asarray(self.vel, dtype=float)
            if self.vel.shape[0] != self.dim:
                raise ValueError(f"vel must be a vector of length {self.dim}, got shape {self.vel.shape}")

        if self.accel is None:
            self.accel = jnp.zeros(self.dim, dtype=float)
        else:
            self.accel = jnp.asarray(self.accel, dtype=float)
            if self.accel.shape[0] != self.dim:
                raise ValueError(f"accel must be a vector of length {self.dim}, got shape {self.accel.shape}")


class SphereStateContainer():
    """
    Memory container for sphere particles using a Structure-of-Arrays (SoA) layout.

    This container preallocates memory for storing the dynamic state and geometric properties and the state
    of sphere particles.

    Attributes
    ----------
    _pos : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing the positions of the spheres.
    _vel : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing the velocities of the spheres.
    _accel : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing the accelerations of the spheres.
    _rad : jnp.ndarray
        A 1D array of length maxSpheres storing the radii of the spheres.
    _spatialHash : jnp.ndarray
        A 1D array of length maxSpheres used to store spatial hash values for neighbor search or sorting.
    _sortedIndices : jnp.ndarray
        A 1D array of length maxSpheres representing the sorted order of spheres.
    """
    def __init__(self, dim, maxSpheres):
        """
        Initialize the SphereStateContainer.

        Parameters
        ----------
        dim : int
            The dimensionality of the simulation domain.
        maxSpheres : int
            The maximum number of sphere particles for which to preallocate memory.
        """
        # State data
        self._pos = jnp.zeros((maxSpheres, dim), dtype=float)
        self._vel = jnp.zeros((maxSpheres, dim), dtype=float)
        self._accel = jnp.zeros((maxSpheres, dim), dtype=float)
        self._mass = jnp.zeros(maxSpheres, dtype=float)

        # Shape data
        self._rad = jnp.zeros(maxSpheres, dtype=float)
        
        # Contact data
        self._spatialHash = jnp.zeros(maxSpheres, dtype=int)
        self._sortedIndices = jnp.arange(maxSpheres, dtype=int)

class StateContainer():
    """
    Aggregates memory containers for different types of simulation particles.

    Currently, the container supports only spheres using a Structure-of-Arrays (SoA) layout.
    Future extensions may include additional particle types (e.g., walls, clumps).

    Attributes
    ----------
    _dim : jnp.ndarray
        The simulation domain's dimensionality stored as a JAX array.
    _maxSpheres : jnp.ndarray
        The maximum number of sphere particles allocated in memory.
    _nSpheres : jnp.ndarray
        The current count of spheres in the simulation.
    _spheres : SphereStateContainer
        The container holding the state data for sphere particles.
    """
    def __init__(self, dim, maxSpheres = 1):
        """
        Initialize the StateContainer.

        Parameters
        ----------
        dim : int
            The dimensionality of the simulation domain (must be 2 or 3).
        maxSpheres : int, optional
            The maximum number of sphere particles to preallocate memory for. Default is 1.

        Raises
        ------
        ValueError
            If `dim` is not 2 or 3, or if `maxSpheres` is negative.
        """
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. Got dim={dim}")
        self._dim = jnp.asarray(dim, dtype=int)

        if maxSpheres < 0:
            raise ValueError(f"maxSpheres has to be a positive integer. Got maxSpheres={maxSpheres}")
        self._maxSpheres = jnp.asarray(maxSpheres, dtype=int)
        self._nSpheres = jnp.asarray(0, dtype=int)

        # Containers for each type of particle.
        self._spheres = SphereStateContainer(dim, maxSpheres)
        # self._walls = WallMemory(dim, maxSpheres)

    def getIthSphereState(self, i:int) -> State:
        if i >= self._nSpheres:
            raise IndexError("Index out of range.")

        return State(self._dim, self._spheres.pos[i], self._spheres.vel[i], self._spheres.accel[i], self._spheres.mass[i])
