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
from jaxdem import Renderer


class System:
    """
    This class encapsulates simulation parameters, domain, memory management,
    and the simulation workflow. It manages a set of particles,
    handles time stepping, and stores simulation data.

    Attributes
    ----------
    dim : jax.numpy.DeviceArray
        The simulation dimension (2D or 3D).
    dt : jax.numpy.DeviceArray
        The simulation time step.
    finalTime : jax.numpy.DeviceArray
        Final simulation time.
    saveTime : jax.numpy.DeviceArray
        Time frequency for saving data.
    gravity : jax.numpy.DeviceArray
        The gravitational acceleration vector.
    domain : Domain
        The simulation domain.
    bodies : BodyContainer
        The unified container holding all simulation bodies (particles).
    dataDir : str
        Base directory for saving simulation data.
    saveCounter : int
        Counter for saved data files.
    """

    def __init__(self,
                dim: int = 3,
                dt: float = 0.01,
                finalTime: float = 1.0,
                saveTime: float = 0.1,
                gravity = None,
                domain: Domain = None,
                dataDir: str = "data"):
        """
        Initialize the simulation System.

        Parameters
        ----------
        dim : int, optional
            Dimensionality of the domain (2 or 3). Default is 3.
        dt : float, optional
            Time step size for the simulation. Default is 0.01.
        finalTime : float, optional
            Final time up to which the simulation runs. Default is 1.0.
        saveTime : float, optional
            Frequency for saving data to disk. Default is 0.1.
        gravity : array-like of shape (dim,), optional
            Gravitational acceleration vector. If None, defaults to a zero vector.
        domain : Domain, optional
            A pre-initialized simulation Domain object. If None, a default domain is created.
        dataDir : str, optional
            Directory path for saving simulation data. Default is "data".

        Raises
        ------
        ValueError
            If `dim` is not 2 or 3.
            If the length of `gravity` does not match `dim`.
            If `dim` does not match `domain.dim` when a domain is provided.
        """
        self.finalTime = jnp.asarray(finalTime, dtype=float)
        self.saveTime = jnp.asarray(saveTime, dtype=float)
        self._dt = jnp.asarray(dt, dtype=float)

        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. Got dim={dim}")
        self._dim = jnp.asarray(dim, dtype=int)

        if gravity is None:
            gravity = jnp.zeros(self._dim, dtype=float)
        if self._dim != gravity.shape[0]:
            raise ValueError(
                f"dim and len(gravity) don't match. "
                f"Got dim={self._dim} and len(gravity)={gravity.shape[0]}"
            )
        self._gravity = gravity

        if domain is None:
            domain = Domain(dim=self._dim)
        if self._dim != domain.dim:
            raise ValueError(
                f"dim and domain.dim don't match. "
                f"Got dim={self._dim} and domain.dim={domain.dim}"
            )
        self._domain = domain

        # Will be allocated via allocateMemory().
        self._bodies = None

        self.dataDir = dataDir
        self.saveCounter = 0

    def allocateMemory(self, nSpheres: int = 1, maxMaterials: int = 1):
        """
        Allocate memory for the simulation.

        This method creates a unified state container that stores particle data
        (e.g., positions, velocities, radii, etc.) using a Structure-of-Arrays
        layout.

        Parameters
        ----------
        nSpheres : int, optional
            Number of spherical particles to allocate memory for. Default is 1.
        maxMaterials : int, optional
            Number of distinct material types. Default is 1.

        Returns
        -------
        None
        """
        self._bodies = BodyContainer(self._dim, nSpheres)

    def save(self, binary: bool = True):
        """
        Save the current simulation data to disk.

        This method saves the domain data as well as the current state
        of the spheres (positions, velocities, radii, forces) in VTK format
        within the directory specified by `self.dataDir`.

        Parameters
        ----------
        binary : bool, optional
            If True, data is saved in binary format. Default is True.

        Returns
        -------
        None
        """
        if not os.path.exists(self.dataDir):
            os.makedirs(self.dataDir)

        # Prepare filenames
        domainFilename = os.path.join(self.dataDir, f"domain_{self.saveCounter:08d}")
        spheresFilename = os.path.join(self.dataDir, f"spheres_{self.saveCounter:08d}")

        # Save domain data
        self._domain.save(domainFilename, binary=binary)

        # Prepare VTK objects for sphere data
        points = vtk.vtkPoints()
        radiusArray = vtk.vtkFloatArray()
        radiusArray.SetName("Radius")

        massArray = vtk.vtkFloatArray()
        massArray.SetName("Mass")

        velocityArray = vtk.vtkFloatArray()
        velocityArray.SetName("Velocity")
        velocityArray.SetNumberOfComponents(self.dim)

        forceArray = vtk.vtkFloatArray()
        forceArray.SetName("Force")
        forceArray.SetNumberOfComponents(self.dim)

        # Compute forces (mass * acceleration)
        n = int(self.bodies.nSpheres)
        Force = self.bodies.memory._accel * self.bodies.memory._mass

        # Ensure data is ready on host
        self.bodies.memory._pos.block_until_ready()
        Force.block_until_ready()

        # Fill VTK data arrays
        for i in range(n):
            pos = self.bodies.memory._pos[i]
            rad = self.bodies.memory._rad[i]
            mass = self.bodies.memory._mass[i]
            vel = self.bodies.memory._vel[i]
            force = Force[i]

            if self.dim == 2:
                pos = jnp.append(pos, 0.0)

            points.InsertNextPoint(pos)
            radiusArray.InsertNextValue(rad)
            massArray.InsertNextValue(mass)
            velocityArray.InsertNextTuple(vel)
            forceArray.InsertNextTuple(force)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        pd = polydata.GetPointData()
        pd.AddArray(radiusArray)
        pd.AddArray(massArray)
        pd.AddArray(velocityArray)
        pd.AddArray(forceArray)
        pd.SetActiveScalars("Radius")

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(spheresFilename + ".vtp")
        writer.SetInputData(polydata)
        if binary:
            writer.SetDataModeToBinary()
        writer.Write()

        self.saveCounter += 1

    @property
    def dt(self) -> jnp.ndarray: # float
        """
        jax.numpy.DeviceArray: The simulation time step.
        """
        return self._dt

    @property
    def gravity(self) -> jnp.ndarray:
        """
        jax.numpy.DeviceArray: The gravitational acceleration vector.
        """
        return self._gravity

    @property
    def domain(self):
        """
        Domain: The simulation domain object.
        """
        return self._domain

    @property
    def bodies(self):
        """
        BodyContainer: The container holding all simulation bodies.
        """
        return self._bodies

    @property
    def dim(self) -> jnp.ndarray: # int
        """
        jax.numpy.DeviceArray: The dimension of the simulation (2 or 3).
        """
        return self._dim

    def run(self, steps: int = None, binary: bool = True, render: bool = True):
        """
        Run the simulation loop.

        This method performs time stepping from t=0 up to `finalTime`.
        It periodically saves the simulation state based on `saveTime`,
        and can optionally render the simulation via VTK.

        Parameters
        ----------
        steps : int, optional
            The total number of simulation steps to run. If None, it is computed
            from `finalTime` and `dt`.
        binary : bool, optional
            If True, data is saved in binary VTK format. Default is True.
        render : bool, optional
            If True, VTK rendering is activated during simulation. The render
            window is set to 1/3 of the screen size. Default is False.

        Returns
        -------
        None
        """
        if render:
            renderer = Renderer(self)
            renderer.start()
            
        if steps is None:
            steps = int(self.finalTime / self.dt)

        # Number of "save intervals" throughout the run
        saveSteps = max(1, int(steps * float(self.dt) / float(self.saveTime)))

        for i in range(saveSteps):
            currentState = self.bodies.memory.getState()
            currentState = jax.lax.fori_loop(0, int(steps / saveSteps), self.step, currentState)
            self.bodies.memory.setState(currentState)
            self.save()

    @partial(jax.jit, static_argnames=("self",))
    def step(self, ii: int, state):
        """
        A single time step of the simulation.

        Parameters
        ----------
        ii : int
            Loop counter.
        state : tuple of jax.numpy.DeviceArray
            A tuple `(pos, vel, accel)` representing the current state of all
            particles at this time step.

        Returns
        -------
        tuple of jax.numpy.DeviceArray
            The updated state `(pos, vel, accel)` after this time step.
        """
        pos, vel, accel = state

        # ---------------------------------------------------
        # Insert contact detection logic here (if needed)
        # ---------------------------------------------------

        # ---------------------------------------------------
        # Insert force computation logic here
        # e.g., accel = F / mass - gravity, etc.
        # ---------------------------------------------------

        # Update velocity and position
        vel += self.dt * accel
        pos += self.dt * vel

        return (pos, vel, accel)
