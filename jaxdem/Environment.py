# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions

import jax 
import jax.numpy as jnp
import vtk

from functools import partial

class Domain:
    """
    Class for defining a simulation domain.

    Attributes
    ----------
    dim : int
        Dimension of the simulation domain (must be 2 or 3).
    length : list of float
        The lengths of the domain boundaries along the x, y, and z axes.
    anchor : list of float
        The coordinates of the lower (left-bottom) corner of the domain.
        Defaults to [0.0, 0.0, 0.0] if not specified.
    boundary : list of str or None
        The type for each domain boundary. Options include "free", "periodic", or "reflect".
    boundary_tags : any
        Additional labels associated with the domain boundaries (used when saving in VTK format).

    # TO DO: CREATE @PROPERTIES FOR ACCESING THE CLASS DATA AND FUNCTIONF FOR MODIFYING CLASS PROPERTIES 
    """

    def __init__(self, dim: int=3, length=None, anchor=None, boundary=None, boundary_tags=None):
        """
        Initialize a simulation domain.

        Parameters
        ----------
        dim : int, optional
            Dimension of the domain. Must be 2 or 3.
        length : list of float, optional
            The length(s) of the domain boundaries. This can be an iterable of floats with lenght dim.
        anchor : list or tuple of float, optional
            The coordinates of the lower (left-bottom) corner of the domain.
            Defaults to [0.0, 0.0, 0.0] if not provided.
        boundary : list of str, optional
            Specifies the type for each boundary. Options are "free", "periodic", or "reflect".
            Default is None.
        boundary_tags : any, optional
            Additional labels to be associated with the domain boundaries when saving in VTK format.
            Default is None.
        """
        boundary_type = {"free":0, "periodic":1, "reflect":2}
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. got dim={dim}")
        self._dim = dim

        if length is None:
            length = jnp.ones(self._dim, dtype=float)

        if self._dim != len(length):
            raise ValueError(f"dim and len(length) dont match. got dim={dim}. len(length)={len(length)}")

        self._length = jnp.array(length, dtype=float)

        if anchor is None:
            anchor = jnp.zeros(self._dim, dtype=float)
        
        if self._dim != len(anchor):
            raise ValueError(f"dim and len(anchor) dont match. got dim={dim}. len(anchor)={len(anchor)}")

        self._anchor = jnp.array(anchor, dtype=float)

        if boundary is None:
            boundary = ["free"]*self._dim

        if self._dim != len(boundary):
            raise ValueError(f"dim and len(boundary) dont match. got dim={dim}. len(boundary)={len(boundary)}")

        self._boundary = jnp.array([boundary_type[b] for b in boundary], dtype=int)

    def save(self, filename: str, binary: bool = True):
        """
        Save the domain as a XML file, compatible with Paraview.

        Parameters
        ----------
        filename : str
            The name (or path) of the file where the domain will be saved. DONT INCLUDE THE FILE EXTENSION. 
            Paraview expectes the .vtp file extension. We will add it for you. 

        binary : bool, optional
            Whether or not to save the data using binary format.

        # TO DO: ADD OPTION TO ADD TAGS TO BOUNDARIES AND BOUNDARY INFORMATION.
        """
        center = self._anchor + self._length/2
        center.block_until_ready()

        cube = vtk.vtkCubeSource()
        cube.SetXLength(self._length[0])
        cube.SetYLength(self._length[1])

        if self._dim == 3:
            cube.SetZLength(self._length[2])
            cube.SetCenter(center[0], center[1], center[2])
        elif self._dim == 2:
            cube.SetZLength(0.0)
            cube.SetCenter(center[0], center[1], 0.0)
        
        cube.Update()
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename+".vtp")
        writer.SetInputData(cube.GetOutput())
        if binary:
            writer.SetDataModeToBinary()
        writer.Write()

class System(object):
    """
    This class handles the simulation and encapsulates all the information for running the simulation.
    Atributs of this class should not be modified directly. Instead, use the configuration functions.

    Attributes
    ----------
    XXXXXXXXXXXXXXXXX
    """

    def __init__(self, dim:int = 3, domain:Domain = Domain(), dt:float = 0.01, gravity = jnp.array([0.0, 0.0, 0.0], dtype=float)):
        """
        Initialize the simulation

        Parameters
        ----------
        domain : Domain
            The simulation domain. An instance of the Domain class.
        dt : float, optional
            The simulation time step (default is 0.01).
        gravity : float, optional
            The gravitational acceleration (default is 9.81).
        **kwargs : dict
            Any additional parameters you wish to store.
        """
        self.dim = dim
        self.N_spheres = 0
        self.N_particles = 0

        self.domain = domain
        self.dt = dt
        self.gravity = jnp.array(gravity, dtype=float)


        self.pos = jnp.zeros_like(positions)
        self.vel = jnp.zeros_like(positions)
        self.accel = jnp.zeros_like(positions)
        self.particleHash = jnp.zeros(N, dtype=int)
        self.sorted_indices = jnp.zeros(N, dtype=int)


        if dim != gravity.shape[0]:
            raise Exception(f"The dimension of the simulation doesnt match the dimension of gravity. dim = {dim}, gravity.shape[0] = {gravity.shape[0]}")

        if dim != Domain.dim:
            raise Exception(f"The dimension of the simulation doesnt match the dimension of the domain. dim = {dim}, Domain.dim = {Domain.dim}")