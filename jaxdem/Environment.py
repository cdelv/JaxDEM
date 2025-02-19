import jax 
import jax.numpy as jnp
import vtk

import jax
import jax.numpy as jnp
import vtk

import vtk
import jax
import jax.numpy as jnp

class Domain:
    """
    Class for defining a simulation domain.

    Attributes
    ----------
    dim : int
        Dimension of the simulation domain (must be 2 or 3).
    length : list of float
        The lengths of the domain boundaries along the x, y, and z axes (if aplicable).
        If a single float is provided at initialization, it is used for all axes.
    anchor : list of float
        The coordinates of the lower (left-bottom) corner of the domain.
        Defaults to [0.0, 0.0, 0.0] if not specified.
    boundary : list of str or None
        The type for each domain boundary. Options include "free", "periodic", or "reflect".
    boundary_tags : any
        Additional labels associated with the domain boundaries (used when saving in VTK format).
    """

    def __init__(self, dim: int = 3, length=1.0, anchor=None, boundary=None, boundary_tags=None):
        """
        Initialize a 3D simulation domain.

        Parameters
        ----------
        dim : int, optional
            Dimension of the domain. Must be 3 (default is 3).
        length : float or list of float, optional
            The length(s) of the domain boundaries. This can be a single float or an iterable of three floats.
            If a single float is provided, it is applied to the x, y, and z axes. Default is 1.0.
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
        if dim != 3:
            raise ValueError("Only 3D domains are supported.")
        self.dim = dim

        # Process the 'length' parameter.
        if isinstance(length, (int, float)):
            self.length = [float(length)] * 3
        else:
            try:
                length_list = list(length)
            except TypeError:
                raise ValueError("length must be a float or an iterable of three floats.")
            if len(length_list) != 3:
                raise ValueError("For a 3D domain, length must have exactly three elements.")
            self.length = [float(l) for l in length_list]

        # Process the 'anchor' parameter.
        if anchor is None:
            self.anchor = [0.0, 0.0, 0.0]
        else:
            try:
                anchor_list = list(anchor)
            except TypeError:
                raise ValueError("anchor must be an iterable of three floats.")
            if len(anchor_list) != 3:
                raise ValueError("For a 3D domain, anchor must have exactly three elements.")
            self.anchor = [float(a) for a in anchor_list]

        self.boundary = boundary
        self.boundary_tags = boundary_tags

    def save(self, filename: str):
        """
        Save the domain as a VTK file.

        The domain is saved as a cube, with the cube's center computed as:
            center = anchor + (length / 2)

        Parameters
        ----------
        filename : str
            The name (or path) of the file where the domain will be saved.
        """
        cube = vtk.vtkCubeSource()
        cube.SetXLength(self.length[0])
        cube.SetYLength(self.length[1])
        cube.SetZLength(self.length[2])
        
        # Compute the center as anchor + half of the length in each dimension.
        center = jnp.array(self.anchor) + jnp.array(self.length) / 2.0
        center_np = jax.device_get(center)
        cube.SetCenter(center_np[0], center_np[1], center_np[2])
        cube.Update()

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(cube.GetOutput())
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