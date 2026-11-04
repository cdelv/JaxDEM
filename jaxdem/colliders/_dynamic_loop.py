# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Dynamic loops with one bound across explicit environment batch axes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


def shared_bounded_loop(body: Callable[..., Any], *, first_step: bool = False) -> Any:
    """Return a loop over ``body(index, carry, operands)``.

    Each explicit ``vmap`` axis shares the maximum bound. The body must mask
    exhausted work. A prototype call supplies carry variance, while its values
    remain unused. With the default option, zero bounds skip the body. Manual
    mesh axes retain independent bounds.

    ``first_step=True`` executes the first step before the loop. The body must
    leave its carry unchanged when the bound is zero. This option preserves
    the first step's values and manual-axis variance.
    """

    @custom_vmap
    def dispatch(steps: Any, initial: Any, operands: Any) -> Any:
        if first_step:
            initial = body(jnp.asarray(0, steps.dtype), initial, operands)
            start = 1
        else:
            prototype = body(jnp.asarray(0, steps.dtype), initial, operands)
            initial = jax.tree.map(
                lambda start, result: start + jnp.zeros_like(result),
                initial,
                prototype,
            )
            start = 0
        return jax.lax.fori_loop(
            jnp.asarray(start, steps.dtype),
            steps,
            lambda index, carry: body(index, carry, operands),
            initial,
        )

    @dispatch.def_vmap
    def mapped(
        axis_size: int, in_batched: Any, steps: Any, initial: Any, operands: Any
    ) -> Any:
        _, carry_batched, operands_batched = in_batched
        initial = jax.tree.map(
            lambda value, batched: (
                value
                if batched
                else jnp.broadcast_to(value, (axis_size,) + value.shape)
            ),
            initial,
            carry_batched,
        )
        carry_axes = jax.tree.map(lambda _: 0, initial)
        operand_axes = jax.tree.map(
            lambda batched: 0 if batched else None, operands_batched
        )

        def batch_body(index: Any, carry: Any, arguments: Any) -> Any:
            return jax.vmap(
                lambda local_carry, local_arguments: body(
                    index, local_carry, local_arguments
                ),
                in_axes=(carry_axes, operand_axes),
                axis_size=axis_size,
            )(carry, arguments)

        result = shared_bounded_loop(batch_body, first_step=first_step)(
            jnp.max(steps, initial=0), initial, operands
        )
        return result, jax.tree.map(lambda _: True, result)

    return dispatch
