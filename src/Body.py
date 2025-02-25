# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions

import jax
import jax.numpy as jnp
import vtk

from dataclasses import dataclass

from jaxdem import Shape
from jaxdem import Sphere
from jaxdem import State, StateContainer

from functools import partial


@dataclass
class Body:
    """
    Represents a simulation body.

    A Body instance encapsulates the state and geometric shape of a simulation
    particle. It holds a reference to the particle's state and its geometric
    representation (e.g., a Sphere). The `index` indicates the location of this
    body in the state container.

    Attributes
    ----------
    state : State
        The dynamic state of the body (position, velocity, acceleration, mass).
    shape : Shape
        The geometric representation of the body (e.g., `Sphere`).
    index : int
        The index of the body in the unified state container's memory arrays.
    """
    state: State
    shape: Shape
    index: int


class BodyContainer:
    """
    Container for simulation bodies providing an Array-of-Structures (AoS) interface.

    Attributes
    ----------
    idTracking : bool
        Flag indicating whether to maintain a mapping of logical indices to
        physical memory indices. This is primarily useful as arrays get
        sorted or re-ordered for performance reasons.
    """

    def __init__(self, dim: int = 3, maxSpheres: int = 1, idTracking: bool = True):
        """
        Initialize the BodyContainer.

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the simulation (2 or 3). Default is 3.
        maxSpheres : int, optional
            The maximum number of spheres that can be stored in the container.
            Default is 1.
        idTracking : bool, optional
            If True, the container maintains a mapping from logical indices
            (as exposed by this class) to the underlying physical memory
            indices in the StateContainer. Default is True.

        Notes
        -----
        Internally, a `StateContainer` is allocated to store sphere data in a
        Structure-of-Arrays layout (positions, velocities, accelerations, etc.).
        """
        self._memory = StateContainer(dim, maxSpheres)
        self.idTracking = idTracking
        self._sphereIndex = jnp.arange(self._memory._nSpheres, dtype=int)

    def addSphere(self, spheres, states=None):
        """
        Add one or more spheres to the simulation memory.

        This method accepts either a single `Sphere` instance or a list of
        `Sphere` instances. It adds the provided sphere(s) to the simulation's
        Structure-of-Arrays memory, updates the corresponding arrays, and
        increments the count of spheres. If state information is provided, it
        uses that state to update position, velocity, acceleration, and mass
        arrays; otherwise, it uses a default `State`. It returns a list of IDs
        corresponding to the indices of the newly added spheres.

        Parameters
        ----------
        spheres : Sphere or list of Sphere
            A single Sphere instance or a list of Sphere instances to add to
            the simulation.
        states : State or list of State, optional
            A single State instance or a list of State instances corresponding
            to each sphere. Must have the same length as `spheres` if provided.
            If not provided, a default `State` is used (i.e., zeros for
            velocity and acceleration, with an arbitrary mass).

        Returns
        -------
        list of int
            A list of integer IDs representing the indices of the newly added
            spheres in the memory container.

        Raises
        ------
        ValueError
            If the maximum number of spheres (`maxSpheres`) has been reached.
            If the provided objects are not of type `Sphere` or `State`.
            If the number of provided states does not match the number of
            spheres.
        """
        if not isinstance(spheres, list):
            spheres = [spheres]

        if states is not None:
            if not isinstance(states, list):
                states = [states]
            if len(spheres) != len(states):
                raise ValueError("Length of `states` and `spheres` doesn't match.")

        new_ids = []
        for i, sphere in enumerate(spheres):
            if not isinstance(sphere, Sphere):
                raise ValueError("The object provided is not a Sphere.")

            if states is not None and not isinstance(states[i], State):
                raise ValueError("The object provided is not a State.")

            idx = self._memory._nSpheres
            if idx >= self._memory._maxSpheres:
                raise ValueError("Maximum number of spheres reached.")

            self._memory._rad = self._memory._rad.at[idx].set(sphere.rad)

            state = states[i] if states is not None else State(dim=self._memory._dim)

            self._memory._pos = self._memory._pos.at[idx].set(state.pos)
            self._memory._vel = self._memory._vel.at[idx].set(state.vel)
            self._memory._accel = self._memory._accel.at[idx].set(state.accel)
            self._memory._mass = self._memory._mass.at[idx].set(state.mass)

            self._memory._nSpheres += 1
            new_ids.append(idx)

        self._sphereIndex = jnp.append(self._memory._nSpheres, idx + jnp.arange(len(spheres)))
        return new_ids

    def getIthBody(self, i: int) -> Body:
        """
        Retrieve the i-th body (in logical indexing).

        This method reconstructs a `Body` instance for the logical index `i` by
        reading the corresponding state from the simulation's SoA memory using
        the maintained index mapping (`_sphereIndex`). If ID tracking is disabled,
        it will use the raw index `i` directly.

        Parameters
        ----------
        i : int
            The logical (user-facing) index of the body to retrieve.

        Returns
        -------
        Body
            A `Body` instance containing the state and geometric shape of the
            i-th body, along with its current memory index (`index`).

        Raises
        ------
        IndexError
            If the index `i` is out of range (i.e., >= `_nSpheres`).
        """
        if not self.idTracking:
            print("Warning: ID tracking is disabled; index may be outdated.")

        if i < self._memory._nSpheres:
            j = self._sphereIndex[i]  # Physical index
            rad = self._memory._spheres._rad[j]
            sphere = Sphere(rad=rad)
            state = self._memory.getIthSphereState(j)
        else:
            raise IndexError("Index out of range.")

        return Body(state=state, shape=sphere, index=j)

    @property
    def dim(self) -> jnp.ndarray: # int
        """
        The dimensionality of the simulation (2D or 3D).
        """
        return self._memory._dim

    @property
    def nSpheres(self) -> jnp.ndarray: # int
        """
        The current number of spheres stored in the container.
        """
        return self._memory._nSpheres

    @property
    def memory(self) -> StateContainer:
        """
        StateContainer:
            The underlying Structure-of-Arrays memory object that holds the
            physical data (positions, velocities, etc.) for all spheres.
        """
        return self._memory