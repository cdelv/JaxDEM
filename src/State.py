# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
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
            self.pos = jnp.zeros(int(self.dim), dtype=float)
        else:
            self.pos = jnp.asarray(self.pos, dtype=float)
            if self.pos.shape[0] != int(self.dim):
                raise ValueError(f"pos must be a vector of length {self.dim}, got shape {self.pos.shape}")

        if self.vel is None:
            self.vel = jnp.zeros(int(self.dim), dtype=float)
        else:
            self.vel = jnp.asarray(self.vel, dtype=float)
            if self.vel.shape[0] != int(self.dim):
                raise ValueError(f"vel must be a vector of length {self.dim}, got shape {self.vel.shape}")

        if self.accel is None:
            self.accel = jnp.zeros(int(self.dim), dtype=float)
        else:
            self.accel = jnp.asarray(self.accel, dtype=float)
            if self.accel.shape[0] != int(self.dim):
                raise ValueError(f"accel must be a vector of length {self.dim}, got shape {self.accel.shape}")

class StateContainer():
    """
    Consolidated memory container for sphere particles using a Structure-of-Arrays (SoA) layout.

    This container stores the dynamic state and geometric properties of spheres in a unified manner.
    It supports batch updates and can generate an Array-of-Structures (AoS) view on demand.
    
    Attributes
    ----------
    dim : jnp.ndarray
        The simulation domain's dimensionality.
    maxSpheres : jnp.ndarray
        The maximun count of sphere particles.
    nSpheres : jnp.ndarray
        The current count of sphere particles.
    pos : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing positions.
    vel : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing velocities.
    accel : jnp.ndarray
        A 2D array of shape (maxSpheres, dim) storing accelerations.
    mass : jnp.ndarray
        A 1D array of length maxSpheres storing particle masses.
    rad : jnp.ndarray
        A 1D array of length maxSpheres storing sphere radii.
    """
    def __init__(self, dim, maxSpheres = 1):
        """
        Initialize the StateContainer.

        Parameters
        ----------
        dim : int
            The dimensionality of the simulation domain (must be 2 or 3).
        nSpheres : int, optional
            The number of sphere particles to preallocate memory for. Default is 1.

        Raises
        ------
        ValueError
            If `dim` is not 2 or 3, or if `nSpheres` is negative.
        """
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. Got dim={dim}")
        self._dim = jnp.asarray(dim, dtype=int)

        if maxSpheres < 0:
            raise ValueError(f"maxSpheres has to be a positive integer. Got maxSpheres={maxSpheres}")
        self._maxSpheres = jnp.asarray(maxSpheres, dtype=int)
        self._nSpheres = jnp.asarray(0, dtype=int)

        # State data
        self._pos = jnp.zeros((self._maxSpheres, int(self._dim)), dtype=float)
        self._vel = jnp.zeros((self._maxSpheres, int(self._dim)), dtype=float)
        self._accel = jnp.zeros((self._maxSpheres, int(self._dim)), dtype=float)
        self._mass = jnp.zeros(self._maxSpheres, dtype=float)

        # Shape data
        self._rad = jnp.zeros(self._maxSpheres, dtype=float)

    def getIthState(self, i:int) -> State:
        if i >= self._nSpheres:
            raise IndexError("Index out of range.")

        return State(self._dim, self.pos[i], self.vel[i], self.accel[i], self.mass[i])

    def setState(self, state):
        """
        Update the simulation step after a simulation step.
        """
        self._pos, self._vel, self._accel = state

    @partial(jax.jit, static_argnames=("self"))
    def getState(self):
        return (self._pos, self._vel, self._accel)


