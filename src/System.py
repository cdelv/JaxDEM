# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions

import os
import jax 
import jax.numpy as jnp
import vtk 

from dataclasses import dataclass, field
from functools import partial

from jaxdem import Domain
from jaxdem import Sphere 
from jaxdem import BodyContainer

class System:
    """
    A simulation system for the JaxDEM framework.

    This class encapsulates simulation parameters, domain, memory management, and the simulation workflow.

    Attributes
    ----------
    dim : jax.numpy.DeviceArray
        The simulation dimension.
    dt : jax.numpy.DeviceArray
        The simulation time step.
    tFinal : jax.numpy.DeviceArray
        Final simulation time.
    saveTime : jax.numpy.DeviceArray
        Time frequency for saving data.
    gravity : jax.numpy.DeviceArray
        The gravitational acceleration vector.
    domain : Domain
        The simulation domain.
    memory : UnifiedStateContainer
        The unified container holding the simulation state.
    datadir : str
        Base directory for saving simulation data.
    saveCounter : int
        Counter for saved data files.
    """
    def __init__(self, 
                 dim: int = 3, dt: float = 0.01, tFinal: float = 1.0, saveTime: float = 0.1, 
                 gravity = None, domain: Domain = None, datadir = "data"):
        self.tFinal = jnp.asarray(tFinal, dtype=float)
        self.saveTime = jnp.asarray(saveTime, dtype=float)
        self._dt = jnp.asarray(dt, dtype=float)
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. Got dim={dim}")
        self._dim = jnp.asarray(dim, dtype=int)
        if gravity is None:
            gravity = jnp.zeros(self._dim, dtype=float)
        if self._dim != gravity.shape[0]:
            raise ValueError(f"dim and len(gravity) don't match. Got dim={self._dim} and len(gravity)={gravity.shape[0]}")
        self._gravity = gravity
        if domain is None:
            domain = Domain(dim=self._dim)
        self._domain = domain
        if self._dim != domain.dim:
            raise ValueError(f"dim and domain.dim don't match. Got dim={self._dim} and domain.dim={domain.dim}")
        self._bodies = None  # Will be allocated via allocateMemory()

        self.datadir = datadir  
        self.saveCounter = 0

    def allocateMemory(self, nSpheres=1, maxMaterials=0):
        """
        Allocate memory for simulation bodies.

        This method creates a unified state container that stores particle data using a
        Structure-of-Arrays layout, replacing separate containers for spheres and other particles.
        """
        self._bodies = BodyContainer(self._dim, nSpheres)

    def save(self, binary: bool = True):
        """
        Save the current simulation data to disk in the self.datadir directory.

        Parameters
        ----------
        binary : bool, optional
            If True, data is saved in binary format. Default is True.

        Returns
        -------
        None
        """
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        domain_filename = os.path.join(self.datadir, f"domain_{self.saveCounter:08d}")
        spheres_filename = os.path.join(self.datadir, f"spheres_{self.saveCounter:08d}")
        
        self._domain.save(domain_filename, binary=binary)
        
        # Save sphere data as VTK points.
        points = vtk.vtkPoints()
        radiusArray = vtk.vtkFloatArray()
        radiusArray.SetName("Radius")
        velocityArray = vtk.vtkFloatArray()
        velocityArray.SetName("Velocity")
        forceArray = vtk.vtkFloatArray()
        forceArray.SetName("Force")
        velocityArray.SetNumberOfComponents(self.dim)
        forceArray.SetNumberOfComponents(self.dim)
        n = int(self.bodies.nSpheres)
        Force = self.bodies.memory._accel * self.bodies.memory._mass

        for i in range(n):
            pos = self.bodies.memory._pos[i]
            rad = self.bodies.memory._rad[i]
            vel = self.bodies.memory._vel[i]
            force = Force[i]
            if self.dim == 3:
                pt = [float(pos[0]), float(pos[1]), float(pos[2])]
                velTuple = [float(vel[0]), float(vel[1]), float(vel[2])]
                forceTuple = [float(force[0]), float(force[1]), float(force[2])]
            else:
                pt = [float(pos[0]), float(pos[1]), 0.0]
                velTuple = [float(vel[0]), float(vel[1]), 0.0]
                forceTuple = [float(force[0]), float(force[1]), 0.0]
            points.InsertNextPoint(pt)
            radiusArray.InsertNextValue(float(rad))
            velocityArray.InsertNextTuple(velTuple)
            forceArray.InsertNextTuple(forceTuple)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        pd = polydata.GetPointData()
        pd.AddArray(radiusArray)
        pd.AddArray(velocityArray)
        pd.AddArray(forceArray)
        pd.SetActiveScalars("Radius")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(spheres_filename + ".vtp")
        writer.SetInputData(polydata)
        if binary:
            writer.SetDataModeToBinary()
        writer.Write()
        self.saveCounter += 1

    @property    
    def dt(self):
        return self._dt

    @property    
    def gravity(self):
        return self._gravity

    @property    
    def domain(self):
        return self._domain

    @property    
    def bodies(self):
        return self._bodies

    @property    
    def dim(self):
        return self._dim

    def run(self, steps=None, binary=True, render=False):
        """
        Run the simulation loop.

        Parameters
        ----------
        steps : int, optional
            The total number of simulation steps to run. If None, computed from tFinal and dt.
        binary : bool, optional
            If True, data is saved in binary format. Default is True.
        render : bool, optional
            If True, VTK rendering is activated during simulation. The render window is set to 1/3 of the screen size.
        
        Returns
        -------
        None
        """
        if steps is None:
            steps = int(self.tFinal / self.dt)
        
        saveSteps = max(1, int(steps * float(self.dt) / float(self.saveTime)))


        for i in range(saveSteps):
            currentState = self.bodies.memory.getState()
            currentState = jax.lax.fori_loop(0, int(steps / saveSteps), self.step, currentState)
            self.save()

    @partial(jax.jit, static_argnames=("self"))
    def step(self, ii, state):
        pos, vel, accel = state
        
        # Contact detection

        # Compute forces


        # update velocity and position.
        vel += self.dt * accel
        pos += self.dt * vel
        return (pos, vel, accel)