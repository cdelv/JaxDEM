# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Bond exclusions select the exact forces and energy of an asymmetric triangle."""

import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils import get_contacts


@pytest.mark.parametrize(
    "backend", ["naive", "CellList", "MultiCellList", "NeighborList"]
)
@pytest.mark.parametrize("include_bonds", [False, True])
def test_excluded_pairs(backend, include_bonds):
    pos = np.array([[0.0, 0.0], [0.55, 0.25], [1.2, -0.1]])
    state = jd.State.create(
        pos=jnp.asarray(pos), rad=jnp.full(3, 0.8), bond_id=[[1], [0, 2], [1]]
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("elastic", young=1.0, poisson=0.25, density=1.0)]
    )
    system = jd.System.create(
        state=state,
        mat_table=table,
        collider_type=backend,
        collider_kw=(
            {
                "max_neighbors": 3,
                "cutoff": 1.6,
                "secondary_collider_type": "CellList",
            }
            if backend == "NeighborList"
            else {}
        ),
        interact_same_bond_id=include_bonds,
    )
    pairs = [(0, 1), (0, 2), (1, 2)] if include_bonds else [(0, 2)]
    expected_force, expected_energy = np.zeros_like(pos), 0.0
    for i, j in pairs:
        delta = pos[i] - pos[j]
        distance = np.linalg.norm(delta)
        overlap = 1.6 - distance
        force = overlap * delta / distance  # k=1, independent of collider/law code.
        expected_force[i] += force
        expected_force[j] -= force
        expected_energy += 0.5 * overlap**2

    evaluated, updated = jd.System.evaluate_forces(state, system)
    np.testing.assert_allclose(evaluated.force, expected_force, rtol=2e-6, atol=1e-7)
    _, _, energy = updated.collider.compute_potential_energy(state, updated)
    np.testing.assert_allclose(energy, expected_energy, rtol=2e-6)
    _, _, contacts = get_contacts(state, updated)
    actual_pairs = {tuple(pair) for pair in np.asarray(contacts.pair_ids)}
    assert contacts.pair_ids.shape[0] == 2 * len(pairs)
    assert actual_pairs == {(i, j) for a, b in pairs for i, j in [(a, b), (b, a)]}
