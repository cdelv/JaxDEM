# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Pair-force diagnostics cover the configured interaction range."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils import (
    compute_contact_pressure,
    compute_contact_stress_tensor,
    get_contacts,
)


@pytest.mark.parametrize(
    "collider",
    ["Naive", "CellList", "MultiCellList", "neighbor-cell", "neighbor-multi"],
)
@pytest.mark.parametrize("composition", ["single", "combiner", "router"])
@pytest.mark.parametrize("cutoff_ratio,distance", [(2.5, 2.0), (4.0, 3.2)])
def test_diagnostics_follow_force_reach(collider, composition, cutoff_ratio, distance):
    direction = np.array([0.6, 0.8])
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], distance * direction]),
        rad=jnp.array([0.4, 0.6]),
        species_id=jnp.array([0, 1]),
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("lj", density=1.0, epsilon=1.0)]
    )
    law = jd.ForceModel.create("lennardjones", cutoff_ratio=cutoff_ratio)
    if composition == "combiner":
        law = jd.LawCombiner(laws=(law, jd.ForceModel.create("wca")))
    elif composition == "router":
        law = jd.ForceRouter.from_dict(2, {(0, 1): law})
    collider_kw = {}
    if collider.startswith("neighbor-"):
        collider_kw = {
            "secondary_collider_type": (
                "CellList" if collider == "neighbor-cell" else "MultiCellList"
            ),
            "cutoff": 0.5,
            "skin": 0.1,
            "max_neighbors": 3,
        }
        collider = "NeighborList"
    system = jd.System.create(
        state=state,
        force_model=law,
        mat_table=table,
        collider_type=collider,
        collider_kw=collider_kw,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 16.0)},
    )

    _, updated, contacts = get_contacts(state, system)
    np.testing.assert_array_equal(contacts.pair_ids, [[0, 1], [1, 0]])
    totals = np.zeros((2, 2))
    np.add.at(totals, np.asarray(contacts.pair_ids)[:, 0], np.asarray(contacts.forces))
    force_on_zero = (
        -24.0 / distance * (2.0 / distance**12 - 1.0 / distance**6) * direction
    )
    np.testing.assert_allclose(totals, [force_on_zero, -force_on_zero], rtol=3e-6)
    evaluated, _ = jd.System.evaluate_forces(state, updated)
    np.testing.assert_allclose(totals, evaluated.force, rtol=3e-6)

    _, _, stress = compute_contact_stress_tensor(state, updated)
    expected_stress = np.outer(-distance * direction, force_on_zero) / 16.0**2
    np.testing.assert_allclose(stress, expected_stress, rtol=3e-6)
    _, _, pressure = compute_contact_pressure(state, updated)
    np.testing.assert_allclose(pressure, np.trace(expected_stress) / 2, rtol=3e-6)
