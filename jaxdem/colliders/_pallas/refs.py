# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Trace-time named buffer packing, Triton load/store views, and vector math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from .config import pltriton


@dataclass(frozen=True)
class PackLayout:
    """Static name-to-position map resolved completely at trace time."""

    index: dict[str, int]
    slices: dict[str, slice]


class ArgPack:
    """Build a flat kernel argument list with named slots instead of magic indices."""

    def __init__(self) -> None:
        self._args: list[Any] = []
        self._index: dict[str, int] = {}
        self._slices: dict[str, slice] = {}

    def add(self, name: str, arr: jax.Array) -> None:
        self._index[name] = len(self._args)
        self._args.append(arr)

    def add_group(self, name: str, arrs: tuple[jax.Array, ...] | list[jax.Array]) -> None:
        start = len(self._args)
        self._args.extend(arrs)
        self._slices[name] = slice(start, len(self._args))

    def build(self) -> tuple[tuple[jax.Array, ...], PackLayout]:
        return tuple(self._args), PackLayout(dict(self._index), dict(self._slices))


def ref_at(ref: Any, b: Any, is_batched: bool, idx: Any) -> Any:
    return ref.at[b, idx] if is_batched else ref.at[idx]


def scalar_at(ref: Any, b: Any, is_batched: bool, idx: int) -> Any:
    return ref[b, idx] if is_batched else ref[idx]


def load1_ref(
    ref: Any,
    b: Any,
    is_batched: bool,
    idx: Any,
    mask: Any,
    other: Any = 0.0,
    eviction_policy: str | None = None,
) -> jax.Array:
    if eviction_policy is not None:
        return pltriton.load(
            ref_at(ref, b, is_batched, idx),
            mask=mask,
            other=other,
            eviction_policy=eviction_policy,
        )
    return pltriton.load(ref_at(ref, b, is_batched, idx), mask=mask, other=other)


def store1(
    ref: Any, b: Any, has_batch: bool, idx: Any, val: jax.Array, mask: Any
) -> None:
    pltriton.store(ref_at(ref, b, has_batch, idx), val, mask=mask)


def store_vec(
    ref: Any,
    b: Any,
    has_batch: bool,
    idx: Any,
    dim: int,
    vals: tuple[jax.Array, ...],
    mask: Any,
) -> None:
    base = idx * dim
    for d in range(dim):
        pltriton.store(ref_at(ref, b, has_batch, base + d), vals[d], mask=mask)


class RefView:
    """Named accessor over Pallas input refs inside a Triton kernel body."""

    def __init__(
        self,
        in_refs: tuple[Any, ...],
        batched: tuple[bool, ...],
        layout: PackLayout,
        b: Any,
        has_batch: bool,
    ) -> None:
        self._refs = in_refs
        self._batched = batched
        self._layout = layout
        self.b = b
        self.has_batch = has_batch

    def has(self, name: str) -> bool:
        return name in self._layout.index

    def ref_and_batched(self, name: str) -> tuple[Any, bool]:
        i = self._layout.index[name]
        return self._refs[i], self._batched[i]

    def group(self, name: str) -> tuple[tuple[Any, ...], tuple[bool, ...]]:
        sl = self._layout.slices[name]
        return self._refs[sl], self._batched[sl]

    def scalar(self, name: str, idx: int = 0) -> jax.Array:
        ref, is_b = self.ref_and_batched(name)
        return scalar_at(ref, self.b, is_b, idx)

    def load1(
        self,
        name: str,
        idx: Any,
        mask: Any,
        other: Any = 0.0,
        eviction_policy: str | None = None,
    ) -> jax.Array:
        ref, is_b = self.ref_and_batched(name)
        return load1_ref(
            ref, self.b, is_b, idx, mask, other=other, eviction_policy=eviction_policy
        )

    def load_vec(
        self,
        name: str,
        idx: Any,
        dim: int,
        mask: Any,
        other: float = 0.0,
    ) -> tuple[jax.Array, ...]:
        ref, is_b = self.ref_and_batched(name)
        base = idx * dim
        return tuple(
            load1_ref(ref, self.b, is_b, base + d, mask, other=other)
            for d in range(dim)
        )


def batched_pallas(
    run_fn: Callable[[tuple[Any, ...], tuple[bool, ...], int], Any],
    *,
    single_output: bool = False,
) -> Callable[..., Any]:
    """Wrap a 1D/2D Pallas launcher in ``@custom_vmap`` for 2D batched grids."""

    @custom_vmap
    def _fn(*kargs: Any) -> Any:
        return run_fn(kargs, (False,) * len(kargs), 0)

    @_fn.def_vmap
    def _fn_vmap(axis_size: int, in_batched: Any, *kargs: Any) -> Any:
        res = run_fn(kargs, tuple(bool(b) for b in in_batched), axis_size)
        if single_output:
            return res, True
        return res, tuple(True for _ in res)

    return _fn


# ---------------------------------------------------------------------------
# Triton scalar/vector register helpers (traced inside pallas_call kernels)
# ---------------------------------------------------------------------------


def triton_rint(x: jax.Array) -> jax.Array:
    """IEEE round-to-nearest-even without relying on Triton's ``round`` lowering."""
    r = jnp.floor(x + 0.5)
    return jnp.where(
        ((r - x) == 0.5) & ((r - 2.0 * jnp.floor(r * 0.5)) != 0.0), r - 1.0, r
    )


def triton_dot(a: tuple[jax.Array, ...], b: tuple[jax.Array, ...]) -> jax.Array:
    out = a[0] * b[0]
    for x, y in zip(a[1:], b[1:]):
        out = out + x * y
    return out


def triton_cross(
    dim: int, a: tuple[jax.Array, ...], b: tuple[jax.Array, ...]
) -> tuple[jax.Array, ...]:
    if dim == 2:
        return (a[0] * b[1] - a[1] * b[0],)
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def triton_cross_w_r(
    dim: int, w: tuple[jax.Array, ...], r: tuple[jax.Array, ...]
) -> tuple[jax.Array, ...]:
    if dim == 2:
        return (-w[0] * r[1], w[0] * r[0])
    return (
        w[1] * r[2] - w[2] * r[1],
        w[2] * r[0] - w[0] * r[2],
        w[0] * r[1] - w[1] * r[0],
    )


def triton_scaled_unit_and_norm(
    comps: tuple[jax.Array, ...],
) -> tuple[tuple[jax.Array, ...], jax.Array]:
    """Overflow-safe unit vector and norm matching ``linalg.unit_and_norm``."""
    scale = jnp.abs(comps[0])
    for c in comps[1:]:
        scale = jnp.maximum(scale, jnp.abs(c))
    zero = scale == 0.0
    safe_s = jnp.where(zero, 1.0, scale)
    s_comps = tuple(c / safe_s for c in comps)
    n2 = triton_dot(s_comps, s_comps)
    r = jnp.sqrt(jnp.where(zero, 1.0, n2))
    inv_r = jnp.where(zero, 0.0, 1.0 / r)
    return tuple(c * inv_r for c in s_comps), jnp.where(zero, 0.0, r * scale)


def triton_scaled_norm(comps: tuple[jax.Array, ...]) -> jax.Array:
    scale = jnp.abs(comps[0])
    for c in comps[1:]:
        scale = jnp.maximum(scale, jnp.abs(c))
    zero = scale == 0.0
    safe_s = jnp.where(zero, 1.0, scale)
    s_comps = tuple(c / safe_s for c in comps)
    n2 = triton_dot(s_comps, s_comps)
    return jnp.where(zero, 0.0, jnp.sqrt(jnp.where(zero, 1.0, n2)) * scale)


def triton_bond_ok(
    bond_w: int,
    rv: RefView,
    idx_a: jax.Array,
    id_b: jax.Array,
    mask: jax.Array,
    isame: jax.Array,
) -> jax.Array:
    """Return ``True`` where pair ``(idx_a, id_b)`` is allowed by bond exclusions."""
    if bond_w == 0:
        return jnp.ones_like(mask)
    any_match = jnp.zeros_like(mask)
    base = idx_a * bond_w
    for c in range(bond_w):
        bid = rv.load1("bond", base + c, mask, other=-1)
        any_match = any_match | (bid == id_b)
    return jnp.where(isame, True, ~any_match)
