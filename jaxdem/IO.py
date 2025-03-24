# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import os
import numpy as np
import vtk
import vtk.util.numpy_support as numpy_support

from jaxdem.State import state

def save_spheres(current_state: 'state', save_counter: int = 0, data_dir: str = "frames", binary: bool = True) -> int:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = os.path.join(data_dir, f"spheres_{save_counter:08d}.vtp")
    Pos = np.asarray(current_state.pos)
    Rad = np.asarray(current_state.rad)

    if current_state.dim == 2:
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

"""
def save_domain(box: jax.Array, save_counter: int = 0, data_dir: str = "frames", binary: bool = True) -> int:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = os.path.join(data_dir, f"domain_{save_counter:08d}.vtp")
    Box = np.asarray(box)
    if Box.size == 2:
        Box = np.pad(Box, (0, 1), mode='constant', constant_values=0)
    cube = vtk.vtkCubeSource()
    cube.SetXLength(Box[0])
    cube.SetYLength(Box[1])
    cube.SetZLength(Box[2])
    cube.SetCenter(Box/2)
    cube.Update()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(cube.GetOutput())
    if binary:
        writer.SetDataModeToBinary()
    writer.Write()

    return save_counter + 1
"""