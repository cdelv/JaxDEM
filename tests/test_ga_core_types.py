# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""GA composition follows the supplied solid members and optional central core."""

import numpy as np
import pytest

from jaxdem.utils import particle_creation


def _create(**kwargs):
    return particle_creation.create_ga_state(
        N=1,
        nv=4,
        dim=2,
        particle_radius=1.0,
        asperity_radius=0.1,
        particle_mass=2.0,
        n_samples=64,
        seed=0,
        mesh_type="fibonacci",
        **kwargs,
    )


@pytest.mark.parametrize("particle_type", ["clump", "dp"])
def test_default_core_uses_only_surface_members(particle_type):
    default = _create(particle_type=particle_type)
    explicit = _create(particle_type=particle_type, core_type=None)

    assert default.N == 4
    np.testing.assert_allclose(np.linalg.norm(default.pos, axis=-1), 0.9, rtol=2e-6)
    np.testing.assert_allclose(default.rad, 0.1, rtol=2e-6)
    for name in ("pos", "rad", "mass", "volume", "inertia"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))

    # Four separate solid disks: omitting a center leaves the center empty.
    expected_volume = 4 * np.pi * 0.1**2
    actual_volume = (
        default.volume[0] if particle_type == "clump" else default.volume.sum()
    )
    np.testing.assert_allclose(actual_volume, expected_volume, rtol=2e-6)


def test_solid_core_adds_geometry_and_phantom_keeps_its_mass_properties():
    surface = _create()
    solid = _create(core_type="solid")
    phantom = _create(core_type="phantom")

    assert solid.N == surface.N + 1
    assert phantom.N == surface.N
    np.testing.assert_allclose(solid.pos[-1], 0.0, atol=2e-7)
    np.testing.assert_allclose(solid.rad[-1], 0.9, rtol=2e-6)
    np.testing.assert_allclose(phantom.pos, solid.pos[:-1], atol=2e-7)
    np.testing.assert_allclose(phantom.rad, solid.rad[:-1], rtol=2e-6)
    for name in ("mass", "volume", "inertia", "pos_c"):
        np.testing.assert_allclose(getattr(phantom, name), getattr(solid, name)[:-1])
    assert solid.volume[0] > surface.volume[0]


def test_hollow_core_option_is_rejected():
    with pytest.raises(ValueError, match="core_type"):
        _create(core_type="hollow")

    with pytest.raises(ValueError, match="core_type"):
        particle_creation.build_ga_system(
            particle_radii=[1.0],
            vertex_counts=[4],
            asperity_radius=0.1,
            phi=0.1,
            dim=2,
            core_type="hollow",
            n_property_samples=64,
            mesh_type="fibonacci",
            seed=0,
        )


def test_system_builder_defaults_to_solid_surface_members_only():
    # Four radius-0.1 disks have 4% of their unit bounding disk's area.
    # Start at the desired true packing fraction so this checks composition
    # through placement and system construction without a compression loop.
    target_phi = 0.01 * 4 * 0.1**2
    state, system = particle_creation.build_ga_system(
        particle_radii=[1.0, 1.0],
        vertex_counts=[4, 4],
        asperity_radius=0.1,
        phi=target_phi,
        initial_phi_bb=0.01,
        dim=2,
        n_property_samples=64,
        mesh_type="fibonacci",
        collider_type="naive",
        max_n_min_steps_per_outer=32,
        seed=0,
    )

    assert state.N == 8
    np.testing.assert_array_equal(
        np.unique(state.clump_id, return_counts=True)[1], [4, 4]
    )
    np.testing.assert_allclose(state.rad, 0.1, rtol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(state.pos_p, axis=-1), 0.9, rtol=2e-6)
    np.testing.assert_allclose(state.volume, 4 * np.pi * 0.1**2, rtol=2e-6)
    np.testing.assert_allclose(
        2 * state.volume[0] / np.prod(system.domain.box_size), target_phi, rtol=2e-6
    )
