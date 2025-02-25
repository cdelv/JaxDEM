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

    A Body instance encapsulates the state and geometric shape of a simulation particle.
    It holds a reference to the particle's state and its
    geometric representation. The index indicates the location in the stateContainer.
    
    Attributes
    ----------
    state : State
        The dynamic state of the body.
    shape : Sphere
        The geometric representation of the body.
    index : int
        The index of the body in the unified state container.
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
        Flag indicating whether to maintain a mapping of logical indices to physical memory indices.
    """
    def __init__(self, dim: int = 3, maxSpheres: int = 1, idTracking: bool = True):
        """
        Initialize the BodyContainer.

        Parameters
        ----------
        memory : StateContainer
            The memory container holding the simulation state for bodies.
        idTracking : bool, optional
            If True, the container maintains a mapping from logical indices to physical memory indices.
            This is useful when the memory arrays are sorted or re-ordered. Default is True.
        """
        self._memory = StateContainer(dim, maxSpheres)
        self.idTracking = idTracking

        # Initialize the mapping array. Initially, the logical order is the same as the physical order.
        self._sphereIndex = jnp.arange(self._memory._nSpheres, dtype=int)

    def addSphere(self, spheres, states=None):
        """
        Add one or more spheres to the simulation memory.

        This method accepts either a single Sphere instance or a list of Sphere instances. It adds
        the provided sphere(s) to the simulation's Structure-of-Arrays memory, updates the corresponding
        arrays, and increments the count of spheres. If state information is provided, it uses that state
        to update position, velocity, acceleration, and mass arrays; otherwise, it uses a default State.
        It returns a list of IDs corresponding to the indices of the newly added spheres.
        
        Parameters
        ----------
        spheres : Sphere or list of Sphere
            A single Sphere instance or a list of Sphere instances to add to the simulation.
        states : State or list of State, optional
            A single State instance or a list of State instances corresponding to each sphere.
            Must have the same length as spheres if provided.
            
        Returns
        -------
        list of int
            A list of integer IDs representing the indices of the newly added spheres in the memory container.

        Raises
        ------
        ValueError
            If the maximum number of spheres has been reached.
            If the provided objects are not of type Sphere or State.
        """
        # Ensure spheres is a list.
        if not isinstance(spheres, list):
            spheres = [spheres]

        if states is not None:
            if not isinstance(states, list):
                states = [states]
            if len(spheres) != len(states):
                raise ValueError("Length of states and spheres doesn't match.")

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

        self._sphereIndex = jnp.arange(self._memory._nSpheres, dtype=int)
        return new_ids

    def getIthBody(self, i: int) -> Body:
        """
        Retrieve the i-th body as an Array-of-Structures (AoS) representation.

        This method reconstructs a Body instance for the logical index `i` by reading the corresponding
        state from the simulation's Structure-of-Arrays memory using the maintained index mapping.
        If ID tracking is disabled, a warning is printed and the raw index is used.

        Parameters
        ----------
        i : int
            The logical (user-facing) index of the body to retrieve.

        Returns
        -------
        Body
            A Body instance containing the state and geometric shape of the i-th body, along with its current memory index.

        Raises
        ------
        IndexError
            If the index `i` is out of bounds for the number of spheres stored in the memory.
        """
        if not self._idTracking:
            print("Warning: id tracking is disabled; index may be outdated.")

        if i < self._memory._nSpheres:    
            j = self._sphereIndex[i]
            rad = self._memory._spheres._rad[j]
            sphere = Sphere(rad=rad)
            state = self._memory.getIthSphereState(j)
        else:
            raise IndexError("Index out of range.")

        return Body(state=state, shape=sphere, index=j)

    def saveSpheres(self, filename: str, binary: bool = True):
        """
        Save all spheres as a XML file, compatible with Paraview.

        Parameters
        ----------
        filename : str
            The file's name (or path) where the domain will be saved. DON'T INCLUDE THE FILE EXTENSION. 
            Paraview expects the .vtp file extension. We will add it for you. 

        binary : bool, optional
            Whether or not to save the data using binary format.

        # TO DO: ADD STATE INFORMATION TO THE SPHERES.
        """
        points = vtk.vtkPoints()

        radiusArray = vtk.vtkFloatArray()
        radiusArray.SetName("Radius")
        velocityArray = vtk.vtkFloatArray()
        velocityArray.SetName("Velocity")
        forceArray = vtk.vtkFloatArray()
        forceArray.SetName("Force")

        dim = self._memory._dim
        velocityArray.SetNumberOfComponents(dim)
        forceArray.SetNumberOfComponents(dim)

        n = self._memory._nSpheres
        for i in range(n):
            pos = self._memory._spheres._pos[i]
            rad = self._memory._spheres._rad[i]
            vel = self._memory._spheres._vel[i]
            force = self._memory._spheres._accel[i]

            if dim == 3:
                point = [pos[0], pos[1], pos[2]]
                velTuple = [vel[0], vel[1], vel[2]]
                forceTuple = [force[0], force[1], force[2]]
            elif dim == 2:
                point = [pos[0], pos[1], 0.0]
                velTuple = [vel[0], vel[1]]
                forceTuple = [force[0], force[1]]

            points.InsertNextPoint(point)
            radiusArray.InsertNextValue(rad)
            velocityArray.InsertNextTuple(velTuple)
            forceArray.InsertNextTuple(forceTuple)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        pointData = polydata.GetPointData()
        pointData.AddArray(radiusArray)
        pointData.AddArray(velocityArray)
        pointData.AddArray(forceArray)
        pointData.SetActiveScalars("Radius")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename + ".vtp")
        writer.SetInputData(polydata)
        if binary:
            writer.SetDataModeToBinary()
        writer.Write()

    @property    
    def dim(self) -> jnp.ndarray: # int
        return self._memory._dim

    @property    
    def nSpheres(self) -> jnp.ndarray: # int
        return self._memory._nSpheres

    @property    
    def memory(self) -> StateContainer:
        return self._memory


