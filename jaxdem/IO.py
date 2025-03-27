import numpy as np
import vtk
import vtk.util.numpy_support as numpy_support
import jax
import jax.numpy as jnp

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
        if state.pos.ndim <= 2:
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
        else:
            self.save_counter = self.save_batch(state, system)

        return self.save_counter

    def save_batch(self, state: 'State', system: 'System') -> int:
        """
        Save simulation data for a batched simulation.
        For each element in the batch (assumed along axis 0 of state.pos),
        a separate subdirectory is created within self.data_dir. Then for each
        subdirectory, all configured writers are submitted concurrently.
        All futures are waited on after the for-loop, and the save counter is updated
        with the maximum value returned by the writers.

        Parameters
        ----------
        state : State
            The current simulation state; if batched, arrays have shape (B, ...).
        system : System
            The current simulation system configuration; if batched, arrays have shape (B, ...).

        Returns
        -------
        int
            The updated save counter.
        """
        batch_size = state.pos.shape[0]
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers = len(self.writers)) as executor:
            for i in range(batch_size):
                state_dir = os.path.join(self.data_dir, f"state_{i:08d}")
                os.makedirs(state_dir, exist_ok=True)
                state_i = jax.tree_map(
                    lambda x: x[i] if (isinstance(x, jnp.ndarray) and x.shape[0] == batch_size) else x,
                    state
                )
                system_i = jax.tree_map(
                    lambda x: x[i] if (isinstance(x, jnp.ndarray) and x.shape[0] == batch_size) else x,
                    system
                )
                for writer_name in self.writers:
                    futures.append(
                        executor.submit(
                            self._registry[writer_name].write,
                            state_i, system_i, self.save_counter, state_dir, self.binary
                        )
                    )
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        self.save_counter = max(results)
        return self.save_counter + 1




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