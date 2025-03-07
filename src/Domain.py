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
    boundary : list of str
        The type for each domain boundary. Options include "free", "periodic", or "reflect".
    boundary_tags :
        Additional labels associated with the domain boundaries (used when saving in XML format).

    # TO DO: CREATE FUNCTIONS FOR MODIFYING CLASS DATA, HANDLE BOUNDARYS AND TAGS 
    """
    def __init__(self, dim:int=3, length=None, anchor=None, boundary=None, boundary_tags=None):
        """
        Initialize a simulation domain.

        Parameters
        ----------
        dim : int, optional
            Dimension of the domain. Must be 2 or 3.
        length : list or tuple of float, optional
            The length(s) of the domain boundaries.
            Default is ones(dim).
        anchor : list or tuple of float, optional
            The coordinates of the lower (left-bottom) corner of the domain.
            Default is zeros(dim).
        boundary : list of str, optional
            Specifies the type for each boundary. Options are "free", "periodic", or "reflect".
            Default is None.
        boundary_tags : any, optional
            Additional labels to be associated with the domain boundaries when saving in VTK format.
            Default is None.

        TO DO: Handle boundary information and boundary tags
        """
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D domains are supported. Got dim={dim}")
        self._dim = jnp.asarray(dim, dtype=int)

        if length is None:
            length = jnp.ones(self._dim, dtype=float)

        if self._dim != length.shape[0]:
            raise ValueError(f"dim and len(length) dont match. Got dim={self._dim} and len(length)={length.shape[0]}")

        self._length = jnp.array(length, dtype=float)

        if anchor is None:
            anchor = jnp.zeros(self._dim, dtype=float)
        
        if self._dim != anchor.shape[0]:
            raise ValueError(f"dim and len(anchor) dont match. Got dim={self._dim} and len(anchor)={anchor.shape[0]}")

        self._anchor = jnp.array(anchor, dtype=float)

    def save(self, filename: str, binary: bool = True):
        """
        Save the domain as a XML file, compatible with Paraview.

        Parameters
        ----------
        filename : str
            The file's name (or path) where the domain will be saved. DON'T INCLUDE THE FILE EXTENSION. 
            Paraview expects the .vtp file extension. We will add it for you. 

        binary : bool, optional
            Whether or not to save the data using binary format.

        # TO DO: ADD OPTION TO ADD TAGS TO BOUNDARIES AND BOUNDARY INFORMATION.
        """
        center = self._anchor + self._length/2
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

    @property
    @partial(jax.jit, static_argnums=(0,))
    def dim(self):
        return self._dim

    @property
    @partial(jax.jit, static_argnums=(0,))
    def length(self):
        return self._length

    @property
    @partial(jax.jit, static_argnums=(0,))
    def anchor(self):
        return self._anchor

    @property
    @partial(jax.jit, static_argnums=(0,))
    def boundary(self):
        return self._boundary

    @property
    def boundary_tags(self):
        return self._boundary_tags