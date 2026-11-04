# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check contact-history payload shapes in composite force laws."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace

import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.forces import ForceModel, ForceRouter, LawCombiner


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MatrixHistoryLaw(ForceModel):
    def history_shape(self, dim: int) -> tuple[int, ...]:
        return (2, 2)

    @staticmethod
    def force(
        i, j, pos, state, system, history, *, advance_history=True, quasistatic=False
    ):
        strength = jnp.sum(history * jnp.array([[1.0, 2.0], [3.0, 4.0]]), axis=(-2, -1))
        force = strength[..., None] * (pos[j] - pos[i])
        torque = jnp.zeros(jnp.shape(j) + state.torque[i].shape)
        return (
            force,
            torque,
            history + advance_history * jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        )

    @staticmethod
    def energy(i, j, pos, state, system, history):
        return history[..., 0, 1] + 10 * history[..., 1, 0]


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ScalarHistoryLaw(MatrixHistoryLaw):
    def history_shape(self, dim: int) -> tuple[int, ...]:
        return ()

    @staticmethod
    def force(
        i, j, pos, state, system, history, *, advance_history=True, quasistatic=False
    ):
        return (
            history[..., None] * (pos[j] - pos[i]),
            jnp.zeros(jnp.shape(j) + state.torque[i].shape),
            history + 5 * advance_history,
        )

    @staticmethod
    def energy(i, j, pos, state, system, history):
        return 7 * history


def _state_system(force_model):
    state = jdem.State.create(
        pos=jnp.asarray([[0.0, 0.0], [0.5, 0.0], [0.25, 0.5]]),
        rad=jnp.full(3, 0.5),
        species_id=jnp.array([0, 0, 1]),
    )
    system = jdem.System.create(state=state, collider_type="naive")
    return state, replace(system, force_model=force_model)


@pytest.mark.parametrize("advance", [False, True])
def test_combiner_flattens_and_restores_arbitrary_child_history_shapes(advance):
    law = LawCombiner(laws=(MatrixHistoryLaw(), ScalarHistoryLaw()))
    state, system = _state_system(law)
    history = jnp.array([[1.0, 2.0, 3.0, 4.0, 11.0]])

    force, _, updated = law.force(
        0, jnp.asarray([1]), state.pos, state, system, history, advance_history=advance
    )

    assert law.history_shape(state.dim) == (5,)
    assert history.shape == (1, 5)
    assert updated.shape == (1, 5)
    np.testing.assert_array_equal(
        updated, history + advance * jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    )
    np.testing.assert_allclose(force, [[20.5, 0.0]])
    np.testing.assert_allclose(
        law.energy(0, jnp.array([1]), state.pos, state, system, history), [109.0]
    )


@pytest.mark.parametrize("advance", [False, True])
def test_router_restores_matrix_history_and_keeps_flat_storage(advance):
    law = ForceRouter(
        table=(
            (MatrixHistoryLaw(), ScalarHistoryLaw()),
            (ScalarHistoryLaw(), MatrixHistoryLaw()),
        )
    )
    state, system = _state_system(law)
    history = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0, 11.0, 21.0, 22.0, 23.0, 24.0],
            [101.0, 102.0, 103.0, 104.0, 111.0, 121.0, 122.0, 123.0, 124.0],
        ]
    )

    force, _, updated = law.force(
        0, jnp.array([1, 2]), state.pos, state, system, history, advance_history=advance
    )

    assert law.history_shape(state.dim) == (9,)
    increment = np.zeros((2, 9))
    increment[0, :4], increment[1, 4] = [1.0, 2.0, 3.0, 4.0], 5.0
    np.testing.assert_array_equal(updated, history + advance * increment)
    np.testing.assert_allclose(force, [[15.0, 0.0], [27.75, 55.5]])
    np.testing.assert_allclose(
        law.energy(0, jnp.array([1, 2]), state.pos, state, system, history),
        [32.0, 777.0],
    )


@pytest.mark.parametrize("pair_shape", [(0,), (2, 0)])
def test_composite_history_initialization_accepts_empty_pair_axes(pair_shape):
    for law in (
        LawCombiner(laws=(MatrixHistoryLaw(), ScalarHistoryLaw())),
        ForceRouter(table=((MatrixHistoryLaw(),),)),
    ):
        history = jax.jit(lambda: law.init_history(pair_shape, 2))()
        assert history.shape == (*pair_shape, *law.history_shape(2))
