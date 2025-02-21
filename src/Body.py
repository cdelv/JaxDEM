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
    It holds a reference to the particle's state (position, velocity, acceleration) and its
    geometric representation (e.g., a sphere). The `index` attribute indicates the location
    of the body's data within the simulation's Structure-of-Arrays (SoA) memory layout.

    Attributes
    ----------
    state : State
        The state of the body, including its position, velocity, and acceleration.
    shape : Shape TO DO: CHANGE THIS
        The geometric representation of the body (e.g., a sphere).
    index : int
        The index of the body in the simulation's SoA memory.
    """
    state: State
    shape: Shape
    index: int
    # material

class BodyContainer:
    """
    Container for simulation bodies providing an Array-of-Structures (AoS) interface.

    This class bridges the simulation memory stored in a Structure-of-Arrays (SoA) layout with a
    user-friendly AoS interface. It manages a mapping between logical (user-facing) body indices and
    physical indices in the simulation memory, enabling access to particle data even after
    the underlying arrays are re-ordered.

    Attributes
    ----------
    idTracking : bool
        Flag indicating whether to maintain a mapping of logical indices to physical memory indices.
    """
    def __init__(self, memory: StateContainer, idTracking: bool = True):
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
        self._memory = memory
        self.idTracking = idTracking

        # Initialize the mapping array. Initially, the logical order is the same as the physical order.
        self._sphereIndex = jnp.arange(self._memory._maxSpheres, dtype=int)

    def addSphere(self, spheres):
        """
        Add one or more spheres to the simulation memory.

        This method accepts either a single Sphere instance or a list of Sphere instances. It adds
        the provided sphere(s) to the simulation's Structure-of-Arrays memory, updates the corresponding
        arrays, and increments the count of spheres. It returns a list of IDs corresponding to the
        indices of the newly added spheres.
        
        TO DO: ADD ABILITY TO SET STATE

        Parameters
        ----------
        spheres : Sphere or list of Sphere
            A single Sphere instance or a list of Sphere instances to add to the simulation.

        Returns
        -------
        list of int
            A list of integer IDs representing the indices of the newly added spheres in the memory container.

        Raises
        ------
        ValueError
            If the maximum number of spheres has been reached.
            If the object passed is not a Sphere or list of Sphere.
        """
        # Ensure spheres is a list.
        if not isinstance(spheres, list):
            spheres = [spheres]
        
        new_ids = []
        for sphere in spheres:
            if not isinstance(sphere, Sphere):
                raise ValueError("The object provided is not a sphere.")

            idx = self._memory._nSpheres
            if idx >= self._memory._maxSpheres:
                raise ValueError("Maximum number of spheres reached.")

            # Update the SoA arrays in the memory container.
            self._memory._spheres._rad = self._memory._spheres._rad.at[idx].set(sphere.rad) 

            self._memory._nSpheres += 1
            new_ids.append(idx)
        
        # Update the mapping array (logical order remains identity for new spheres).
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
        if i < 0 or i >= self._memory._nSpheres:
            raise IndexError("Index out of range.")
        
        if not self._idTracking:
            print("Warning: id tracking is disabled; index may be outdated.")
        
        j = self._sphereIndex[i]
        pos = self._memory._spheres._pos[j]
        vel = self._memory._spheres._vel[j]
        accel = self._memory._spheres._accel[j]
        rad = self._memory._spheres._rad[j]
        sphere = Sphere(rad=rad)
        state = State(pos, vel, accel)
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


