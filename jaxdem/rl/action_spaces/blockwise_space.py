# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Apply action-space bijectors to contiguous parts of one action vector."""

import jax
import jax.numpy as jnp

from collections.abc import Sequence
from functools import partial
from itertools import accumulate
from typing import cast

import distrax  # type: ignore[import-untyped]
import numpy as np
from distrax._src.bijectors.bijector import Array  # type: ignore[import-untyped]

from . import ActionSpace


@ActionSpace.register("Blockwise")
class BlockwiseSpace(distrax.Bijector, ActionSpace):  # type: ignore[misc]
    r"""Apply separate constraints to explicit slices of a flat action vector.

    Parameters
    ----------
    blocks : sequence of (int, ActionSpace)
        Positive block sizes and their scalar or vector bijectors, in order.
        Sizes must sum to the input vector's length. Each child must preserve
        its event shape and implement :meth:`ActionSpace.log_det_expectation <jaxdem.rl.action_spaces.ActionSpace.log_det_expectation>`
        to support Gaussian policy entropy. Blocks may themselves be
        :class:`BlockwiseSpace <jaxdem.rl.action_spaces.blockwise_space.BlockwiseSpace>` instances.

    The Jacobian is block diagonal, so its log-determinant is the sum of the
    children's log-determinants. The Gaussian entropy correction is likewise
    the sum of their expectations over the corresponding marginal Gaussians.
    Leading sample and batch axes are preserved.

    Example
    -------
    Constrain force and torque separately, followed by two bounded scalars::

        BlockwiseSpace(blocks=(
            (3, MaxNormSpace(max_norm=force_limit)),
            (3, MaxNormSpace(max_norm=torque_limit)),
            (2, BoxSpace(-1.0, 1.0)),
        ))
    """

    __slots__ = ()

    def __init__(self, blocks: Sequence[tuple[int, ActionSpace]]):
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("blocks must not be empty")
        bijectors = []
        for size, space in blocks:
            if (
                isinstance(size, bool)
                or not isinstance(size, (int, np.integer))
                or size <= 0
            ):
                raise ValueError("block sizes must be positive integers")
            if not isinstance(space, ActionSpace) or not isinstance(
                space, distrax.Bijector
            ):
                raise TypeError("each block must contain an ActionSpace bijector")
            if (
                space.event_ndims_in not in (0, 1)
                or space.event_ndims_out != space.event_ndims_in
            ):
                raise ValueError(
                    "block bijectors must preserve scalar or vector event shapes"
                )
            bijectors.append(distrax.Block(space, ndims=1 - space.event_ndims_in))

        self._sizes = tuple(int(size) for size, _ in blocks)
        self._split_indices = tuple(accumulate(self._sizes[:-1]))
        self._bijectors = tuple(bijectors)
        super().__init__(
            event_ndims_in=1,
            is_constant_jacobian=all(b.is_constant_jacobian for b in self._bijectors),
            is_constant_log_det=all(b.is_constant_log_det for b in self._bijectors),
        )

    @property
    def blocks(self) -> tuple[tuple[int, ActionSpace], ...]:
        """Block sizes and original children, retained for factory serialization."""
        return tuple(
            (size, cast(ActionSpace, bijector.bijector))
            for size, bijector in zip(self._sizes, self._bijectors, strict=True)
        )

    @property
    def size(self) -> int:
        """Number of coordinates in one input or output event."""
        return sum(self._sizes)

    def _split(self, x: Array) -> list[jax.Array]:
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != self.size:
            raise ValueError(
                f"Expected a vector of length {self.size}, got shape {x.shape}"
            )
        return jnp.split(x, self._split_indices, axis=-1)

    @partial(jax.named_call, name="BlockwiseSpace.forward_and_log_det")
    def forward_and_log_det(self, x: Array) -> tuple[jax.Array, jax.Array]:
        values, log_dets = zip(
            *(
                bijector.forward_and_log_det(part)
                for bijector, part in zip(self._bijectors, self._split(x), strict=True)
            )
        )
        return jnp.concatenate(values, axis=-1), sum(log_dets)

    @partial(jax.named_call, name="BlockwiseSpace.inverse_and_log_det")
    def inverse_and_log_det(self, y: Array) -> tuple[jax.Array, jax.Array]:
        values, log_dets = zip(
            *(
                bijector.inverse_and_log_det(part)
                for bijector, part in zip(self._bijectors, self._split(y), strict=True)
            )
        )
        return jnp.concatenate(values, axis=-1), sum(log_dets)

    @partial(jax.named_call, name="BlockwiseSpace.log_det_expectation")
    def log_det_expectation(self, mean: jax.Array, std: jax.Array) -> jax.Array:
        """Sum the children's Gaussian log-Jacobian expectations."""
        mean, std = jnp.broadcast_arrays(mean, std)
        corrections = (
            bijector.bijector.log_det_expectation(mu, sigma)
            for bijector, mu, sigma in zip(
                self._bijectors, self._split(mean), self._split(std), strict=True
            )
        )
        return sum(corrections, start=jnp.zeros(mean.shape[:-1], dtype=mean.dtype))


__all__ = ["BlockwiseSpace"]
