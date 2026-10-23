# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""The exported cell follows the domain's physical periodic lattice."""

import jax.numpy as jnp

from itertools import permutations, product

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import jaxdem as jd


def write_domain(tmp_path, dim, domain_type, *, binary=True, **domain_kw):
    box = np.array([2.0, 3.0, 4.0])[:dim]
    anchor = np.array([-1.25, 2.5, -0.75])[:dim]
    state = jd.State.create(
        pos=jnp.asarray((anchor + 0.25 * box)[None]), rad=jnp.array([0.1])
    )
    system = jd.System.create(
        state=state,
        collider_type="",
        domain_type=domain_type,
        domain_kw={"box_size": box, "anchor": anchor, **domain_kw},
    )
    with jd.VTKWriter(directory=tmp_path, writers=["domain"], binary=binary) as writer:
        writer.save(state, system)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(next(tmp_path.rglob("domain_*.vtp"))))
    reader.Update()
    return reader.GetOutput(), box, anchor


@pytest.mark.parametrize(
    "dim,alpha,beta",
    [
        (dim, alpha, beta)
        for dim in (2, 3)
        for alpha, beta in permutations(range(dim), 2)
    ],
)
@pytest.mark.parametrize("gamma", [-0.4, 0.65])
def test_lees_edwards_domain_exports_the_sheared_periodic_cell(
    tmp_path, dim, alpha, beta, gamma
):
    poly, box, anchor = write_domain(
        tmp_path, dim, "leesedwards", gamma=gamma, alpha=alpha, beta=beta
    )
    points = np.unique(vtk_to_numpy(poly.GetPoints().GetData()), axis=0)

    # The gradient edge joins neighboring shear-periodic images. The other
    # lattice edges and the anchor do not change under this imposed shear.
    lattice_edges = np.diag(box)
    lattice_edges[beta, alpha] = gamma * box[beta]
    corners = np.array(
        [
            anchor + np.asarray(image) @ lattice_edges
            for image in product((0, 1), repeat=dim)
        ]
    )
    corners = np.pad(corners, ((0, 0), (0, 3 - dim)))
    np.testing.assert_allclose(points, np.unique(corners, axis=0), atol=2e-6)

    # A transformed surface needs transformed normals, not just tilted points.
    # Check each nondegenerate face against its actual exported edge vectors.
    normals = vtk_to_numpy(poly.GetPointData().GetNormals())
    for index in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(index)
        face = vtk_to_numpy(cell.GetPoints().GetData())
        edge_a, edge_b = face[1] - face[0], face[-1] - face[0]
        if np.linalg.norm(np.cross(edge_a, edge_b)) == 0:
            continue
        normal = normals[cell.GetPointId(0)]
        np.testing.assert_allclose([normal @ edge_a, normal @ edge_b], 0, atol=2e-6)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "domain_type", ["free", "periodic", "reflect", "reflectsphere"]
)
@pytest.mark.parametrize("binary", [False, True])
def test_orthogonal_domain_export_preserves_box_and_anchor(
    tmp_path, dim, domain_type, binary
):
    poly, box, anchor = write_domain(tmp_path, dim, domain_type, binary=binary)
    points = np.unique(vtk_to_numpy(poly.GetPoints().GetData()), axis=0)
    corners = np.array(
        [anchor + np.asarray(image) * box for image in product((0, 1), repeat=dim)]
    )
    corners = np.pad(corners, ((0, 0), (0, 3 - dim)))
    np.testing.assert_allclose(points, np.unique(corners, axis=0), atol=2e-6)
