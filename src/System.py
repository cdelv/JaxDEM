# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import os

import jax 
import jax.numpy as jnp
from functools import partial

from jaxdem import Domain
from jaxdem import StateContainer
from jaxdem import Body, BodyContainer

class System:
    """
    A simulation system for the JaxDEM framework.

    This class encapsulates the simulation parameters, domain, and memory
    management. Also manages the simulation workflow. 

    Attributes
    ----------
    dim: jax.numpy.DeviceArray
        The simulation dimension.
    dt : jax.numpy.DeviceArray
        The simulation time step.
    gravity : jax.numpy.DeviceArray
        The gravitational acceleration vector.
    domain : Domain
        The simulation domain.
    bodies : BodyContainer
        The container for simulation bodies.
    datadir : str
        Base directory name for saving simulation data. 
    saveDomain: bool
        
    saveSpheres: bool
        
    saveGrid: bool
    """

    def __init__(self, dim: int = 3, dt: float = 1.0, gravity = None, domain: Domain = None, datadir = "data"):
        """
        Initialize a new simulation system.

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the simulation domain (2 or 3). Default is 3.
        dt : float, optional
            The simulation time step. Default is 1.0.
        gravity : array or None, optional
            The gravitational acceleration vector. If None, a zero vector of length `dim` is used.
        domain : Domain or None, optional
            The simulation domain. If None, a default domain with the specified dimension is created.

        Raises
        ------
        ValueError
            If `dim` is not 2 or 3.
            If the length of the gravity vector does not match the specified `dim`.
            If the domain's dimension does not match the specified `dim`.
        """
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

        # Bodies container is allocated via allocateMemory()
        self._bodies = None    

        self.datadir = datadir  
        self.saveCounter = 0

    def allocateMemory(self, maxSpheres=1, maxMaterials=0, maxClumps=0, maxPlanes=0, maxFacets=0):
        """
        Allocate memory for simulation bodies and other particle types.

        This method creates the state container that stores particle data using a
        Structure-of-Arrays (SoA) layout and initializes the BodyContainer that provides
        an Array-of-Structures (AoS) interface to the user. 

        Parameters
        ----------
        maxSpheres : int, optional
            The maximum number of spheres for which to allocate memory. Default is 1.
        maxMaterials : int, optional
            The maximum number of materials. Default is 0.
        maxClumps : int, optional
            The maximum number of clumps. Default is 0.
        maxPlanes : int, optional
            The maximum number of planes. Default is 0.
        maxFacets : int, optional
            The maximum number of facets. Default is 0.

        Returns
        -------
        None
        """
        memory = StateContainer(self._dim, maxSpheres=maxSpheres)
        self._bodies = BodyContainer(memory)

    def save(self, binary: bool = True):
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)

        self._domain.save(os.path.join(self.datadir, f"domain_{self.saveCounter:8d}"), binary = binary)
        self._bodies.saveSpheres(os.path.join(self.datadir, f"spheres_{self.saveCounter:8d}"), binary = binary)

        self.saveCounter += 1

    @property    
    def dt(self):
        """Return the simulation time step."""
        return self._dt

    @property    
    def gravity(self):
        """Return the gravitational acceleration vector."""
        return self._gravity

    @property    
    def domain(self):
        """Return the simulation domain."""
        return self._domain

    @property    
    def bodies(self):
        """
        Return the container of simulation bodies.
        """
        return self._bodies

