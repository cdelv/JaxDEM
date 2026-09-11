from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jdem
from jaxdem.forces import ForceModel, ForceRouter, LawCombiner


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MatrixHistoryLaw(ForceModel):
    def history_shape(self, dim: int) -> tuple[int, ...]:
        return (2, 2)

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        shape = jnp.shape(j) + (state.dim,)
        return jnp.zeros(shape), jnp.zeros(shape), history + advance_history

    @staticmethod
    def energy(i, j, pos, state, system):
        return jnp.zeros(jnp.shape(j))


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ScalarHistoryLaw(MatrixHistoryLaw):
    def history_shape(self, dim: int) -> tuple[int, ...]:
        return ()


def _state_system(force_model):
    state = jdem.State.create(
        pos=jnp.asarray([[0.0, 0.0], [0.5, 0.0]]), rad=jnp.full(2, 0.5)
    )
    system = jdem.System.create(state=state, collider_type="naive")
    return state, replace(system, force_model=force_model)


def test_combiner_flattens_and_restores_arbitrary_child_history_shapes():
    law = LawCombiner(laws=(MatrixHistoryLaw(), ScalarHistoryLaw()))
    state, system = _state_system(law)
    history = law.init_history((1,), state.dim)

    _, _, updated = law.force(0, jnp.asarray([1]), state.pos, state, system, history)

    assert law.history_shape(state.dim) == (5,)
    assert history.shape == (1, 5)
    assert updated.shape == (1, 5)
    assert jnp.all(updated == 1)


def test_router_restores_matrix_history_and_keeps_flat_storage():
    law = ForceRouter(table=((MatrixHistoryLaw(),),))
    state, system = _state_system(law)
    history = law.init_history((1,), state.dim)

    _, _, updated = law.force(0, jnp.asarray([1]), state.pos, state, system, history)

    assert law.history_shape(state.dim) == (4,)
    assert history.shape == (1, 4)
    assert updated.shape == (1, 4)
    assert jnp.all(updated == 1)


@pytest.mark.parametrize("pair_shape", [(0,), (2, 0)])
def test_composite_history_initialization_accepts_empty_pair_axes(pair_shape):
    for law in (
        LawCombiner(laws=(MatrixHistoryLaw(), ScalarHistoryLaw())),
        ForceRouter(table=((MatrixHistoryLaw(),),)),
    ):
        history = jax.jit(lambda: law.init_history(pair_shape, 2))()
        assert history.shape == (*pair_shape, *law.history_shape(2))
