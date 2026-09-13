"""Incomplete neighbor lists must never certify mechanical equilibrium."""

from dataclasses import replace

import jax.numpy as jnp
import jaxdem as jd
import optax
import pytest

from jaxdem.minimizers.routines import COLLIDER_OVERFLOW, minimize
from jaxdem.utils.jamming import bisection_jam


def make_system(x):
    st = jd.State.create(pos=jnp.column_stack((jnp.asarray(x), jnp.zeros(len(x)))),
                         rad=jnp.full(len(x), 0.5))
    mat = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    sy = jd.System.create(
        state=st, dt=0.01, minimizer=jd.minimizers.fire,
        domain_type="periodic", domain_kw={"box_size": jnp.full(2, 10.0)},
        force_model_type="spring", collider_type="neighborlist",
        collider_kw={"cutoff": 1.0, "skin": 0.05, "max_neighbors": 1,
                     "secondary_collider_type": "naive"},
        mat_table=jd.MaterialTable.from_materials([mat]),
    )
    return st, sy


@pytest.mark.parametrize("max_steps", [0, 10])
def test_overflow_rejected_even_if_force_check_passes(max_steps):
    st, sy = make_system([0, 0.2, 0.4])
    out = minimize(st, sy, max_steps=max_steps, force_tol=1e6, torque_tol=1e6, return_info=True)
    assert out[-1].finite
    assert not out[-1].converged
    assert int(out[-1].status) == COLLIDER_OVERFLOW
    assert int(out[2]) == 0


def test_overflow_during_relaxation_stops_at_first_bad_evaluation():
    st, sy = make_system([0, 0.5, 1.6])
    initial = minimize(st, sy, max_steps=0, return_info=True)
    assert not initial[1].collider.overflow
    sy = replace(sy, minimizer=optax.sgd(0.2))
    out = minimize(st, sy, max_steps=10, return_info=True)
    assert int(out[2]) == 1
    assert out[1].collider.overflow
    assert int(out[-1].status) == COLLIDER_OVERFLOW
    assert not out[-1].converged


def test_jamming_propagates_overflow_failure():
    st, sy = make_system([0, 0.2, 0.4])
    result, info = bisection_jam(st, sy, n_minimization_steps=10,
                               n_jamming_steps=2, verbose=False, return_info=True)
    assert not result.converged and not info.converged
    assert int(info.minimization.status) == COLLIDER_OVERFLOW
    assert int(info.max_minimization_steps) == 0
