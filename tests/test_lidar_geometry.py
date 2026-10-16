"""Exact nearest-center proximity, angular sectors, ties and periodic images."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils import lidar_2d, lidar_3d, cross_lidar_2d, cross_lidar_3d


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("cross", [False, True])
def test_lidar_reports_nearest_center_and_target_identity(dim, cross):
    targets = np.array([[2.0, 2.0], [1.0, 1.0], [-1.0, 1.0], [20.0, 20.0], [1.0, 1.0]])
    if dim == 3:
        targets = np.column_stack((targets, [2.0, 1.0, -1.0, 20.0, 1.0]))
    state = jd.State.create(pos=jnp.asarray(np.vstack((np.zeros(dim), targets))))
    system = jd.System.create(state=state, collider_type="")
    kwargs = {"n_bins": 4} if dim == 2 else {"n_azimuth": 4, "n_elevation": 4}
    if cross:
        fn = cross_lidar_2d if dim == 2 else cross_lidar_3d
        proximity, ids, overflow = fn(
            state.pos[:1], state.pos[1:], system, lidar_range=5.0, **kwargs
        )
    else:
        fn = lidar_2d if dim == 2 else lidar_3d
        _, _, proximity, ids, overflow = fn(state, system, lidar_range=5.0, **kwargs)
    bins = [2, 3] if dim == 2 else [10, 13]
    expected = np.zeros(4 if dim == 2 else 16)
    expected[bins] = 5 - np.sqrt(dim)
    expected_ids = np.full(len(expected), -1 if cross else 0)
    expected_ids[bins] = np.array([1, 2]) + (0 if cross else 1)
    np.testing.assert_allclose(proximity[0], expected, atol=1e-6)
    np.testing.assert_array_equal(ids[0], expected_ids)
    assert not overflow


@pytest.mark.parametrize("dim", [2, 3])
def test_lidar_uses_the_periodic_image_direction(dim):
    sensor, target = jnp.ones((1, dim)), jnp.ones((1, dim))
    sensor = sensor.at[0, 0].set(0.5)
    target = target.at[0, 0].set(9.5)
    state = jd.State.create(pos=jnp.concatenate((sensor, target)))
    system = jd.System.create(
        state=state,
        collider_type="",
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0)},
    )
    if dim == 2:
        proximity, ids, _ = cross_lidar_2d(sensor, target, system, 5.0, 4)
        bin_index = 0
    else:
        proximity, ids, _ = cross_lidar_3d(sensor, target, system, 5.0, 4, 4)
        bin_index = 2
    expected = np.zeros(proximity.shape)
    expected[0, bin_index] = 4.0
    np.testing.assert_allclose(proximity, expected)
    np.testing.assert_array_equal(ids[0, bin_index], 0)
