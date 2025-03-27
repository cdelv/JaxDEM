import numpy as np
import vtk
import vtk.util.numpy_support as numpy_support

import os
import shutil
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
import concurrent.futures

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System

class VTKBaseWriter(Factory, ABC):
    @staticmethod
    @abstractmethod
    def write(state: 'State', system: 'System', save_counter: int, data_dir: str, binary: bool) -> int:
        """
        Abstract method to write simulation data.
        
        Parameters
        ----------
        state : State
            The current simulation state.
        system : System
            The current simulation system configuration.
        save_counter : int
            The current save counter to use in filename.
        data_dir : str
            Directory to save the output files.
        binary : bool
            Whether to save in binary mode.
        
        Returns
        -------
        int
            The updated save counter.
        """
        ...

@dataclass(kw_only=True)
class VTKWriter(VTKBaseWriter):
    """
    A configurable writer class for saving simulation data.
    
    Attributes
    ----------
    writers : Optional[List[str]], optional
        List of writer types to use. If None, uses all registered writers.
    data_dir : str, default "frames"
        Directory to save output files.
    binary : bool, default True
        Whether to save files in binary mode.
    empty : bool, default False
        Whether to empty data_dir at the start
    save_counter : int
        Counter to track the number of saves performed.
    """
    writers: Optional[List[str]] = None
    data_dir: str = "frames"
    binary: bool = True
    empty: bool = False
    save_counter: int = 0

    def __post_init__(self):
        """
        Initialize the writer after object creation.
        If no writers are specified, use all registered writers.
        Create the data directory if it doesn't exist.
        """
        if self.writers is None:
            self.writers = list(self._registry.keys())

        for writer_name in self.writers:
            if writer_name not in self._registry:
                available = list(self._registry.keys())
                raise KeyError(f"No class registered under '{writer_name}'. Available: {available}")
        
        os.makedirs(self.data_dir, exist_ok=True)

        if self.empty:
            for filename in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def save(self, state: 'State', system: 'System') -> int:
        """
        Save simulation data using configured writers concurrently.
        
        Parameters
        ----------
        state : State
            The current simulation state.
        system : System
            The current simulation system configuration.
        
        Returns
        -------
        int
            The updated save counter.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.writers)) as executor:
            futures = [
                executor.submit(
                    self._registry[writer_name].write, 
                    state, system, self.save_counter, self.data_dir, self.binary
                ) 
                for writer_name in self.writers
            ]
            done, _ = concurrent.futures.wait(futures)
            self.save_counter = max(future.result() for future in done)
        
        return self.save_counter

    @staticmethod
    def write(state: 'State', system: 'System', save_counter: int, data_dir: str, binary: bool) -> int:
        ...

@VTKWriter.register("spheres")
class VTKSpheresWriter(VTKBaseWriter):
    """
    Writer for saving sphere particle data in VTK format.
    """
    @staticmethod
    def write(state: 'State', system: 'System', save_counter: int, data_dir: str, binary: bool) -> int:
        """
        Write sphere particle data to a VTK file.
        """
        filename = os.path.join(data_dir, f"spheres_{save_counter:08d}.vtp")
        Pos = np.asarray(state.pos)
        Rad = np.asarray(state.rad)

        if state.dim == 2:
            Pos = np.pad(Pos, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(Pos))
        radius_array = numpy_support.numpy_to_vtk(Rad)
        radius_array.SetName("Radius")
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(radius_array)
        polydata.GetPointData().SetActiveScalars("Radius")
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(polydata)
        if binary:
            writer.SetDataModeToBinary()
        writer.Write()

        return save_counter + 1

@VTKWriter.register("domain")
class VTKDomainWriter(VTKBaseWriter):
    """
    Writer for saving domain geometry in VTK format.
    """
    @staticmethod
    def write(state: 'State', system: 'System', save_counter: int, data_dir: str, binary: bool) -> int:
        """
        Write domain geometry to a VTK file.
        """
        filename = os.path.join(data_dir, f"domain_{save_counter:08d}.vtp")
        Box = np.asarray(system.domain.box_size)
        Anchor = np.asarray(system.domain.anchor)
        
        if Box.size == 2:
            Box = np.pad(Box, (0, 1), mode='constant', constant_values=0)

        if Anchor.size == 2:
            Anchor = np.pad(Anchor, (0, 1), mode='constant', constant_values=0)
        
        cube = vtk.vtkCubeSource()
        cube.SetXLength(Box[0])
        cube.SetYLength(Box[1])
        cube.SetZLength(Box[2])
        cube.SetCenter(Anchor + Box/2)
        cube.Update()
        
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(cube.GetOutput())
        if binary:
            writer.SetDataModeToBinary()
        writer.Write()

        return save_counter + 1

@VTKWriter.register("grid")
class VTKGridWriter(VTKBaseWriter):
    """
    Writer for saving grid data in VTK format.
    Saves the grid's structure and grid hash for visualization.

    TO DO: improve
    """
    @staticmethod
    def write(state: 'State', system: 'System', save_counter: int, data_dir: str, binary: bool) -> int:        
        # Build the filename for the .vti file
        filename = os.path.join(data_dir, f"grid_{save_counter:08d}.vti")
        
        grid_obj = system.grid
        cell_size = grid_obj.cell_size
        # Convert n_cells to a NumPy array for processing.
        n_cells = np.array(system.grid.n_cells)
        periodic = system.grid.periodic
        
        # Determine grid dimensions: For a uniform grid, dimensions are n_cells + 1.
        if state.dim == 2:
            dims = (int(n_cells[0] + 1), int(n_cells[1] + 1), 1)
            spacing = (cell_size, cell_size, 1.0)
        else:
            dims = (int(n_cells[0] + 1), int(n_cells[1] + 1), int(n_cells[2] + 1))
            spacing = (cell_size, cell_size, cell_size)
        
        # Create the vtkImageData object representing the grid.
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(dims)
        imageData.SetSpacing(spacing)
        
        # Set the origin using the domain's anchor.
        origin = np.asarray(system.domain.anchor, dtype=float)
        if state.dim == 2:
            origin = np.pad(origin, (0, 1), mode='constant', constant_values=0)
        imageData.SetOrigin(origin)
        
        # ---------------------------------------------------------------------
        # Compute grid hash for each grid point.
        # For visualization purposes, we compute a hash based on the grid indices.
        # Here, we assume that the grid hash is computed as the dot product
        # of the grid point index (possibly reduced modulo n_cells for periodicity)
        # with a weight vector. For 2D, we use weights = [1, n_cells[1]].
        # For 3D, weights = [1, n_cells[1], n_cells[2]].
        # ---------------------------------------------------------------------
        if state.dim == 2:
            ix = np.arange(dims[0])
            iy = np.arange(dims[1])
            grid_x, grid_y = np.meshgrid(ix, iy, indexing='ij')
            # Stack the indices into an array of shape (num_points, 2)
            grid_indices = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
            weights = np.array([1, n_cells[1]])
            if periodic:
                # Apply modulo for periodic domains.
                grid_indices = grid_indices % n_cells[:2]
            # Compute hash as the dot product.
            hash_values = np.dot(grid_indices, weights)
        else:
            ix = np.arange(dims[0])
            iy = np.arange(dims[1])
            iz = np.arange(dims[2])
            grid_x, grid_y, grid_z = np.meshgrid(ix, iy, iz, indexing='ij')
            grid_indices = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
            weights = np.array([1, n_cells[1], n_cells[2]])
            if periodic:
                grid_indices = grid_indices % n_cells
            hash_values = np.dot(grid_indices, weights)
        
        # Convert the hash values to a VTK array and assign them as scalars.
        vtk_hash_array = numpy_support.numpy_to_vtk(hash_values, deep=True)
        vtk_hash_array.SetName("GridHash")
        imageData.GetPointData().SetScalars(vtk_hash_array)
        
        # Write the vtkImageData to a file using vtkXMLImageDataWriter.
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(imageData)
        if binary:
            writer.SetDataModeToBinary()
        writer.Write()
        
        return save_counter + 1


