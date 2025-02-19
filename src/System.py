# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax 
import jax.numpy as jnp
import vtk

from functools import partial

from jaxdem import Domain

class System:
    """
    Class for defining a system. The class holds all the necessary information to evolve the simulation state.

    Attributes
    ----------
    dim : int
        Dimension of the simulation domain (must be 2 or 3).
        Defaults to 3.
    dt : float
        Time step.
        Defaults to 1.0
    gravity : list of float
        Defaults to zeros(dim).
    domain : Domain
        Instance of the Domain class

    # TO DO: CREATE @property FOR ACCESING THE CLASS DATA AND FUNCTIONF FOR MODIFYING CLASS DATA 
    """

    def __init__(self, dim:int = 3, dt:float = 1.0, gravity = None, domain:Domain = None):
        """
        Initialize the simulation

        Parameters
        ----------
        dim : int
            Dimension of the simulation domain (must be 2 or 3).
            Defaults to 3.
        dt : float
            Time step.
            Defaults to 1.0
        gravity : list of float
            Defaults to zeros(dim).
        domain : Domain
            Instance of the Domain class. Defaults to Domain(dim=dim)
        """
        self.dt = jnp.array([dt], dtype=float)
        
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. Got dim={dim}")
        self._dim = dim

        if gravity is None:
            gravity = jnp.zeros(self._dim, dtype=float)

        if self._dim != len(gravity):
            raise ValueError(f"dim and len(gravity) dont match. Got dim={self._dim} and len(gravity)={len(gravity)}")

        self.gravity = gravity
        
        if domain is None:
            domain = Domain(dim=self._dim)
        
        if self._dim != domain.dim:
            raise ValueError(f"dim and domain.dim dont match. Got dim={self._dim} and domain.dim={domain.dim}")

        # Memory allocations, TO DO: think on a way to store the bodies
        # Body container for all this data
        # Shape classes
        # State container
        self._Max_spheres = 0
        self._N_spheres = 0
        self._pos = jnp.zeros((self._Max_spheres, self._dim), dtype=float)
        self._vel = jnp.zeros_like(self._pos, dtype=float)
        self._accel =  jnp.zeros_like(self._pos, dtype=float)
        self._spatialHash = jnp.zeros(self._Max_spheres, dtype=int)
        self._sortedIndices = jnp.zeros_like(self._spatialHash, dtype=int)
        # Max planes
        # Max facets
        # Max clumps
        # Max DPs
        # Max material

        # Integrator
        # Search algorithm

        # Save interval
        # Simulation time
        # Iteration
        
        # Contact model

        # Material

        # Add surface information between materials

        # methods for adding spheres
        # methods for adding particles from file, template
        # methods for adding walls
        # methods for saving data

        # run method

        # render simulation