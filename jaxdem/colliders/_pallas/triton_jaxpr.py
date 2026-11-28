# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Generic jaxpr-to-Triton scalarizer for arbitrary ForceModels and Domains.

Pallas Triton requires DRAM reads to use masked ``pltriton.load`` on 1D refs
and forbids non-power-of-2 tensor dimensions (such as 3D vectors). This module
bridges those two constraints generically: it traces any per-pair JAX function
(``ForceModel.force`` or ``Domain.displacement``) with ``jax.make_jaxpr``,
validates via a single-source-of-truth ``jax.eval_shape`` dry run, hoists
loop-invariant equations before the neighbor loop, packs only live leaves into
``ArgPack``, and lowers small per-pair register tensors as ``np.ndarray``
objects of ``(BLOCK,)`` lanes while keeping large arrays as zero-copy strided
DRAM views.

Any function using unsupported primitives or oversized register intermediates
returns ``None`` at trace time so callers fall back cleanly to Pure-JAX.
"""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Var

from .refs import ArgPack, RefView, store1, triton_rint

if TYPE_CHECKING:  # pragma: no cover
    from ...system import System

_MAX_REG_ELEMS = 64
_MAX_UNROLL_SCAN = 16
_MAX_COND_BRANCHES = 16


class _UnsupportedTritonOp(Exception):
    """Raised during dry-run validation when a jaxpr cannot lower to Triton."""


_ELEMENTWISE_PRIMS: frozenset[str] = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "add",
        "and",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "cbrt",
        "ceil",
        "clz",
        "cos",
        "cosh",
        "div",
        "eq",
        "erf",
        "erfc",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "ge",
        "gt",
        "is_finite",
        "le",
        "log",
        "log1p",
        "lt",
        "max",
        "min",
        "mul",
        "ne",
        "neg",
        "not",
        "or",
        "pow",
        "rem",
        "rsqrt",
        "shift_left",
        "shift_right_arithmetic",
        "shift_right_logical",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "square",
        "sub",
        "tan",
        "tanh",
        "xor",
    }
)

_REDUCE_OPS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "reduce_sum": lambda a, b: a + b,
    "reduce_prod": lambda a, b: a * b,
    "reduce_max": jnp.maximum,
    "reduce_min": jnp.minimum,
    "reduce_and": lambda a, b: a & b,
    "reduce_or": lambda a, b: a | b,
}

_SCATTER_OPS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "scatter": lambda _old, upd: upd,
    "scatter-add": lambda old, upd: old + upd,
    "scatter-mul": lambda old, upd: old * upd,
    "scatter-min": jnp.minimum,
    "scatter-max": jnp.maximum,
}


# ---------------------------------------------------------------------------
# Core types: zero-copy DRAM views and NumPy-backed register tensors
# ---------------------------------------------------------------------------


def _c_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return tuple(strides)


def _ravel(coords: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return sum(c * s for c, s in zip(coords, strides))


@dataclass(frozen=True)
class _MemView:
    """Strided view into a packed 1D DRAM ref (not yet loaded into registers)."""

    ref_name: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    dtype: Any
    is_bool: bool = False

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def is_contiguous(self) -> bool:
        return self.strides == _c_strides(self.shape)


@dataclass(frozen=True)
class _RegTensor:
    """Small per-pair tensor stored as an object ndarray of ``(BLOCK,)`` lanes."""

    dtype: Any
    lanes: np.ndarray  # dtype=object, shape == logical tensor shape

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.lanes.shape)

    @property
    def size(self) -> int:
        return int(self.lanes.size)

    def flat(self) -> list[jax.Array]:
        return list(self.lanes.ravel())

    def scalar(self) -> jax.Array:
        return self.lanes.item()  # type: ignore[no-any-return]


def _check_reg_size(shape: tuple[int, ...]) -> None:
    if math.prod(shape) > _MAX_REG_ELEMS:
        raise _UnsupportedTritonOp(f"register tensor too large: {shape}")


def _reg(shape: tuple[int, ...], dtype: Any, elems: list[jax.Array]) -> _RegTensor:
    _check_reg_size(shape)
    arr = np.empty(len(elems), dtype=object)
    for k, e in enumerate(elems):
        arr[k] = e
    return _RegTensor(dtype, arr.reshape(shape))


def _reg_scalar(dtype: Any, val: jax.Array) -> _RegTensor:
    arr = np.empty((), dtype=object)
    arr[()] = val
    return _RegTensor(dtype, arr)


@dataclass(frozen=True)
class _InBinding:
    """How one jaxpr input or const var maps into kernel registers/DRAM refs."""

    kind: str  # "arg" | "mem" | "const_inline" | "empty"
    arg_slot: int = -1
    ref_name: str = ""
    shape: tuple[int, ...] = ()
    strides: tuple[int, ...] = ()
    offset: int = 0
    dtype: Any = None
    is_bool: bool = False
    const_val: Any = None


@dataclass(frozen=True)
class CompiledJaxpr:
    """Validated, loop-invariant-split jaxpr ready for Triton evaluation."""

    closed_jaxpr: ClosedJaxpr
    pre_eqns: tuple[Any, ...]
    loop_eqns: tuple[Any, ...]
    const_bindings: tuple[_InBinding, ...]
    in_bindings: tuple[_InBinding, ...]


@dataclass(frozen=True)
class ForceSpec:
    """Compiled Triton force specification for any ``ForceModel``."""

    compiled: CompiledJaxpr
    dim: int
    ang_dim: int
    hist_shape: tuple[int, ...]
    hist_width: int
    has_torque: bool
    has_history: bool
    advance_history: bool
    only_spheres: bool
    out_dtype: Any


@dataclass(frozen=True)
class DisplacementSpec:
    """Compiled Triton displacement specification for any ``Domain``."""

    compiled: CompiledJaxpr
    dim: int
    dtype: Any


# ---------------------------------------------------------------------------
# DCE, loop-invariant code motion, and single-source-of-truth validation
# ---------------------------------------------------------------------------


def _dce_jaxpr(jaxpr: Jaxpr) -> Jaxpr:
    live: set[Var] = {v for v in jaxpr.outvars if isinstance(v, Var)}
    kept = []
    for eqn in reversed(jaxpr.eqns):
        if any(v in live for v in eqn.outvars):
            kept.append(eqn)
            for a in eqn.invars:
                if isinstance(a, Var):
                    live.add(a)
    kept.reverse()
    jaxpr_any: Any = jaxpr
    result: Jaxpr = jaxpr_any.replace(eqns=kept)
    return result


def _split_loop_invariant(
    jaxpr: Jaxpr, variant_indices: set[int]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    variant: set[Var] = {jaxpr.invars[idx] for idx in variant_indices}
    pre_eqns: list[Any] = []
    loop_eqns: list[Any] = []
    for eqn in jaxpr.eqns:
        if any(isinstance(a, Var) and a in variant for a in eqn.invars):
            variant.update(eqn.outvars)
            loop_eqns.append(eqn)
        else:
            pre_eqns.append(eqn)
    return tuple(pre_eqns), tuple(loop_eqns)


def _used_invar_indices(jaxpr: Jaxpr) -> set[int]:
    var_to_idx = {v: idx for idx, v in enumerate(jaxpr.invars)}
    used: set[int] = set()
    for eqn in jaxpr.eqns:
        for a in eqn.invars:
            if isinstance(a, Var) and a in var_to_idx:
                used.add(var_to_idx[a])
    for a in jaxpr.outvars:
        if isinstance(a, Var) and a in var_to_idx:
            used.add(var_to_idx[a])
    return used


class _DryRunRefView:
    """Minimal ``RefView`` stand-in used under ``jax.eval_shape`` for validation."""

    def __init__(self, ref_dtypes: dict[str, Any], block: int) -> None:
        self._dtypes = ref_dtypes
        self._block = block

    def scalar(self, name: str, idx: int = 0) -> jax.Array:
        return jnp.zeros((), dtype=self._dtypes[name])

    def load1(
        self,
        name: str,
        idx: Any,
        mask: Any,
        other: Any = 0.0,
        eviction_policy: str | None = None,
    ) -> jax.Array:
        return jnp.zeros((self._block,), dtype=self._dtypes[name])


def _compile_jaxpr(
    fn: Callable[..., Any],
    sample_args: tuple[Any, ...],
    *,
    arg_indices: dict[int, int],
    variant_indices: set[int],
    pack: ArgPack,
    ref_prefix: str,
    aliases: dict[int, _InBinding] | None = None,
) -> CompiledJaxpr | None:
    """Trace ``fn(*sample_args)``, validate via dry run, LICM-split, and pack live leaves."""
    try:
        closed = jax.make_jaxpr(fn)(*sample_args)
    except Exception:
        return None
    closed = ClosedJaxpr(_dce_jaxpr(closed.jaxpr), closed.consts)

    flat_inputs, _ = jax.tree_util.tree_flatten(sample_args)
    if len(flat_inputs) != len(closed.jaxpr.invars):
        return None

    used = _used_invar_indices(closed.jaxpr)
    alias_map = aliases or {}
    in_bindings: list[_InBinding] = []
    packed_by_id: dict[int, _InBinding] = {}
    ref_dtypes: dict[str, Any] = {}
    mem_counter = 0

    for idx, leaf in enumerate(flat_inputs):
        shape = tuple(leaf.shape)
        dtype = leaf.dtype
        if idx in arg_indices:
            if math.prod(shape) > _MAX_REG_ELEMS:
                return None
            in_bindings.append(
                _InBinding("arg", arg_slot=arg_indices[idx], shape=shape, dtype=dtype)
            )
        elif idx not in used or math.prod(shape) == 0:
            in_bindings.append(_InBinding("empty", shape=shape, dtype=dtype))
        elif id(leaf) in alias_map:
            b = alias_map[id(leaf)]
            ref_dtypes[b.ref_name] = jnp.int8 if b.is_bool else b.dtype
            in_bindings.append(b)
        elif id(leaf) in packed_by_id:
            in_bindings.append(packed_by_id[id(leaf)])
        else:
            ref_name = f"{ref_prefix}_{mem_counter}"
            mem_counter += 1
            is_bool = dtype == jnp.bool_
            arr = leaf.astype(jnp.int8) if is_bool else leaf
            pack.add(ref_name, arr.reshape(-1))
            ref_dtypes[ref_name] = arr.dtype
            b = _InBinding(
                "mem",
                ref_name=ref_name,
                shape=shape,
                strides=_c_strides(shape),
                offset=0,
                dtype=dtype,
                is_bool=is_bool,
            )
            packed_by_id[id(leaf)] = b
            in_bindings.append(b)

    const_bindings: list[_InBinding] = []
    for c_idx, cval in enumerate(closed.consts):
        shape = tuple(np.shape(cval))
        dtype = jax.dtypes.canonicalize_dtype(np.asarray(cval).dtype)
        if math.prod(shape) <= _MAX_REG_ELEMS:
            const_bindings.append(
                _InBinding("const_inline", shape=shape, dtype=dtype, const_val=cval)
            )
        else:
            ref_name = f"{ref_prefix}_c_{c_idx}"
            is_bool = dtype == np.bool_
            arr = jnp.asarray(cval, dtype=jnp.int8 if is_bool else dtype)
            pack.add(ref_name, arr.reshape(-1))
            ref_dtypes[ref_name] = arr.dtype
            const_bindings.append(
                _InBinding(
                    "mem",
                    ref_name=ref_name,
                    shape=shape,
                    strides=_c_strides(shape),
                    offset=0,
                    dtype=dtype,
                    is_bool=is_bool,
                )
            )

    pre_eqns, loop_eqns = _split_loop_invariant(closed.jaxpr, variant_indices)
    compiled = CompiledJaxpr(
        closed_jaxpr=closed,
        pre_eqns=pre_eqns,
        loop_eqns=loop_eqns,
        const_bindings=tuple(const_bindings),
        in_bindings=tuple(in_bindings),
    )

    # Single-source-of-truth validation: dry-run the exact interpreter under eval_shape.
    def _dry_run() -> tuple[jax.Array, ...]:
        block = 32
        dry_rv: Any = _DryRunRefView(ref_dtypes, block)
        mask = jnp.ones((block,), dtype=jnp.bool_)
        arg_vals: dict[int, _RegTensor] = {}
        for b in in_bindings:
            if b.kind == "arg":
                n_e = math.prod(b.shape)
                elems = [jnp.zeros((block,), dtype=b.dtype) for _ in range(n_e)]
                arg_vals[b.arg_slot] = _reg(b.shape, b.dtype, elems)
        env = _eval_pre_env(compiled, dry_rv, arg_vals, mask, block)
        _eval_eqns(compiled.loop_eqns, env, dry_rv, mask, block)
        outs: list[jax.Array] = []
        for a in closed.jaxpr.outvars:
            r = (
                _const_to_reg(a.val)
                if isinstance(a, Literal)
                else _materialize(env[a], dry_rv)
            )
            outs.extend(r.flat())
        return tuple(outs)

    try:
        jax.eval_shape(_dry_run)
    except _UnsupportedTritonOp:
        return None
    return compiled


def compile_force_spec(
    force_model: Any,
    state: Any,
    system: System,
    pos: jax.Array,
    *,
    advance_history: bool,
    quasistatic: bool,
    pack: ArgPack,
    pr_pos_ref: str | None = None,
    pr_rad_ref: str | None = None,
) -> ForceSpec | None:
    """Compile any ``ForceModel.force`` into a Triton-ready ``ForceSpec``."""
    from ...state import _suspend_rotation_cache_refresh

    dim = int(pos.shape[-1])
    if dim not in (2, 3):
        return None
    ang_dim = 3 if dim == 3 else 1
    hist_shape = force_model.history_shape(dim)
    hist_width = math.prod(hist_shape)
    if hist_width > _MAX_REG_ELEMS:
        return None

    i0 = jnp.asarray(0, dtype=int)
    j0 = jnp.asarray(0, dtype=int)
    h0 = force_model.init_history((), dim)

    def _single_pair(
        i: jax.Array,
        j: jax.Array,
        h_arg: jax.Array,
        pos_arg: jax.Array,
        st_arg: Any,
        sy_arg: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return sy_arg.force_model.force(
            i,
            j,
            pos_arg,
            st_arg,
            sy_arg,
            h_arg,
            advance_history=advance_history,
            quasistatic=quasistatic,
        )

    aliases: dict[int, _InBinding] = {}
    if pr_pos_ref is not None and tuple(pos.shape) == (pos.shape[0], 3):
        aliases[id(pos)] = _InBinding(
            "mem",
            ref_name=pr_pos_ref,
            shape=tuple(pos.shape),
            strides=(4, 1),
            offset=0,
            dtype=pos.dtype,
        )
    if pr_rad_ref is not None and tuple(state.rad.shape) == (pos.shape[0],):
        aliases[id(state.rad)] = _InBinding(
            "mem",
            ref_name=pr_rad_ref,
            shape=tuple(state.rad.shape),
            strides=(4,),
            offset=3,
            dtype=state.rad.dtype,
        )

    with _suspend_rotation_cache_refresh():
        compiled = _compile_jaxpr(
            _single_pair,
            (i0, j0, h0, pos, state, system),
            arg_indices={0: 0, 1: 1, 2: 2},
            variant_indices={1, 2},
            pack=pack,
            ref_prefix="f_in",
            aliases=aliases,
        )
    if compiled is None:
        return None
    outvars = compiled.closed_jaxpr.jaxpr.outvars
    if len(outvars) != 3:
        return None
    f_aval, t_aval, h_aval = (v.aval for v in outvars)
    if (
        tuple(getattr(f_aval, "shape", ())) != (dim,)
        or tuple(getattr(t_aval, "shape", ())) != (ang_dim,)
        or tuple(getattr(h_aval, "shape", ())) != hist_shape
    ):
        return None

    return ForceSpec(
        compiled=compiled,
        dim=dim,
        ang_dim=ang_dim,
        hist_shape=hist_shape,
        hist_width=hist_width,
        has_torque=bool(force_model.has_pair_torque),
        has_history=hist_width > 0,
        advance_history=advance_history,
        only_spheres=bool(state.only_spheres),
        out_dtype=getattr(f_aval, "dtype", pos.dtype),
    )


def compile_displacement_spec(
    system: System,
    dim: int,
    dtype: Any,
    *,
    pack: ArgPack,
    ref_prefix: str = "d_in",
) -> DisplacementSpec | None:
    """Compile any ``Domain.displacement`` into a Triton-ready ``DisplacementSpec``."""
    p0 = jnp.zeros((dim,), dtype=dtype)

    def _disp(pi: jax.Array, pj: jax.Array, sy_arg: Any) -> jax.Array:
        return sy_arg.domain.displacement(pi, pj, sy_arg)

    compiled = _compile_jaxpr(
        _disp,
        (p0, p0, system),
        arg_indices={0: 0, 1: 1},
        variant_indices={0, 1},
        pack=pack,
        ref_prefix=ref_prefix,
    )
    if compiled is None:
        return None
    outvars = compiled.closed_jaxpr.jaxpr.outvars
    if len(outvars) != 1 or tuple(getattr(outvars[0].aval, "shape", ())) != (dim,):
        return None
    return DisplacementSpec(compiled=compiled, dim=dim, dtype=dtype)


# ---------------------------------------------------------------------------
# Triton kernel interpreter (traced inside pl.pallas_call or eval_shape dry run)
# ---------------------------------------------------------------------------


def _as_block(x: jax.Array, block: int) -> jax.Array:
    return jnp.broadcast_to(x, (block,)) if x.ndim == 0 else x


def _const_to_reg(cval: Any) -> _RegTensor:
    arr = np.asarray(cval)
    _check_reg_size(tuple(arr.shape))
    dt = jax.dtypes.canonicalize_dtype(arr.dtype)
    return _reg(
        tuple(arr.shape), dt, [jnp.asarray(x.item(), dtype=dt) for x in arr.ravel()]
    )


def _materialize(val: _MemView | _RegTensor, rv: RefView) -> _RegTensor:
    if isinstance(val, _RegTensor):
        return val
    _check_reg_size(val.shape)
    elems: list[jax.Array] = []
    for coord in itertools.product(*map(range, val.shape)):
        v = rv.scalar(val.ref_name, val.offset + _ravel(coord, val.strides))
        elems.append(v != 0 if val.is_bool else v)
    return _reg(val.shape, val.dtype, elems)


def _load_mem_elem(
    view: _MemView, addr: jax.Array, rv: RefView, mask: jax.Array
) -> jax.Array:
    other: Any = 0 if view.is_bool else jnp.asarray(0, dtype=view.dtype)
    v = rv.load1(view.ref_name, addr, mask, other=other)
    return v != 0 if view.is_bool else v


def _lookup_reg_elem(reg: _RegTensor, flat_idx: jax.Array) -> jax.Array:
    elems = reg.flat()
    acc = elems[0]
    for k in range(1, len(elems)):
        acc = jnp.where(flat_idx == k, elems[k], acc)
    return acc


def _clip_index(idx: jax.Array, max_start: int) -> jax.Array:
    return jnp.minimum(jnp.maximum(idx, 0), max(max_start, 0))


def _eval_integer_pow(x: jax.Array, y: int) -> jax.Array:
    if y == 0:
        return jnp.ones_like(x)
    if y < 0:
        return 1.0 / _eval_integer_pow(x, -y)
    res: jax.Array | None = None
    base, exp = x, y
    while exp > 0:
        if exp & 1:
            res = base if res is None else res * base
        exp >>= 1
        if exp > 0:
            base = base * base
    assert res is not None
    return res


def _eval_gather(
    operand: _MemView | _RegTensor,
    indices: _RegTensor,
    params: dict[str, Any],
    out_shape: tuple[int, ...],
    rv: RefView,
    mask: jax.Array,
) -> _RegTensor:
    _check_reg_size(out_shape)
    dnums = params["dimension_numbers"]
    slice_sizes = tuple(params["slice_sizes"])
    offset_dims = tuple(dnums.offset_dims)
    collapsed = set(dnums.collapsed_slice_dims)
    start_index_map = tuple(dnums.start_index_map)
    op_batch_dims = tuple(getattr(dnums, "operand_batching_dims", ()))
    idx_batch_dims = tuple(getattr(dnums, "start_indices_batching_dims", ()))

    offset_set = set(offset_dims)
    batch_out_dims = [d for d in range(len(out_shape)) if d not in offset_set]
    rem_op_dims = [
        d
        for d in range(len(operand.shape))
        if d not in collapsed and d not in op_batch_dims
    ]
    index_vector_dim = len(indices.shape) - 1
    op_strides = (
        operand.strides if isinstance(operand, _MemView) else _c_strides(operand.shape)
    )
    op_offset = operand.offset if isinstance(operand, _MemView) else 0
    idx_lanes = indices.lanes

    out_elems: list[jax.Array] = []
    for out_c in itertools.product(*map(range, out_shape)):
        b_coord = tuple(out_c[d] for d in batch_out_dims)
        starts: list[int | jax.Array] = [0] * len(operand.shape)
        for ob_d, ib_d in zip(op_batch_dims, idx_batch_dims):
            starts[ob_d] = b_coord[ib_d]
        for k, op_d in enumerate(start_index_map):
            idx_c = b_coord[:index_vector_dim] + (k,) + b_coord[index_vector_dim:]
            starts[op_d] = _clip_index(
                idx_lanes[idx_c], operand.shape[op_d] - slice_sizes[op_d]
            )
        for off_pos, op_d in enumerate(rem_op_dims):
            starts[op_d] = starts[op_d] + out_c[offset_dims[off_pos]]

        addr: int | jax.Array = op_offset
        for c_val, st in zip(starts, op_strides):
            if st != 0:
                addr = addr + c_val * st

        if isinstance(operand, _MemView):
            if isinstance(addr, int):
                v = rv.scalar(operand.ref_name, addr)
                out_elems.append(v != 0 if operand.is_bool else v)
            else:
                out_elems.append(_load_mem_elem(operand, addr, rv, mask))
        else:
            out_elems.append(
                operand.flat()[addr]
                if isinstance(addr, int)
                else _lookup_reg_elem(operand, addr)
            )
    return _reg(out_shape, operand.dtype, out_elems)


def _eval_scatter(
    name: str,
    operand: _RegTensor,
    indices: _RegTensor,
    updates: _RegTensor,
    params: dict[str, Any],
) -> _RegTensor:
    dnums = params["dimension_numbers"]
    update_window_dims = tuple(dnums.update_window_dims)
    inserted_dims = set(dnums.inserted_window_dims)
    scatter_dims_map = tuple(dnums.scatter_dims_to_operand_dims)
    op_batch_dims = tuple(getattr(dnums, "operand_batching_dims", ()))
    idx_batch_dims = tuple(getattr(dnums, "scatter_indices_batching_dims", ()))

    win_set = set(update_window_dims)
    upd_batch_dims = [d for d in range(len(updates.shape)) if d not in win_set]
    rem_op_dims = [
        d
        for d in range(len(operand.shape))
        if d not in inserted_dims and d not in op_batch_dims
    ]
    index_vector_dim = len(indices.shape) - 1
    op_strides = _c_strides(operand.shape)
    comb = _SCATTER_OPS[name]
    out_elems = operand.flat()

    for upd_c in itertools.product(*map(range, updates.shape)):
        u_val = updates.lanes[upd_c]
        b_coord = tuple(upd_c[d] for d in upd_batch_dims)
        op_c: list[int | jax.Array] = [0] * len(operand.shape)
        in_bounds: jax.Array | bool = True
        for ob_d, ib_d in zip(op_batch_dims, idx_batch_dims):
            op_c[ob_d] = b_coord[ib_d]
        for k, op_d in enumerate(scatter_dims_map):
            idx_c = b_coord[:index_vector_dim] + (k,) + b_coord[index_vector_dim:]
            iv = indices.lanes[idx_c]
            in_bounds = in_bounds & (iv >= 0) & (iv < operand.shape[op_d])
            op_c[op_d] = iv
        for w_pos, op_d in enumerate(rem_op_dims):
            op_c[op_d] = op_c[op_d] + upd_c[update_window_dims[w_pos]]

        flat_addr: int | jax.Array = 0
        for c_val, st in zip(op_c, op_strides):
            flat_addr = flat_addr + c_val * st
        for k in range(len(out_elems)):
            hit = (flat_addr == k) & in_bounds
            out_elems[k] = jnp.where(hit, comb(out_elems[k], u_val), out_elems[k])
    return _reg(operand.shape, operand.dtype, out_elems)


def _eval_subjaxpr(
    sub: ClosedJaxpr,
    args: list[_MemView | _RegTensor],
    rv: RefView,
    load_mask: jax.Array,
    block: int,
    load_cache: dict[Any, _RegTensor] | None = None,
    arg_nonneg: list[str | None] | None = None,
) -> list[_MemView | _RegTensor]:
    sub_env: dict[Var, _MemView | _RegTensor] = {
        cv: _const_to_reg(cval) for cv, cval in zip(sub.jaxpr.constvars, sub.consts)
    }
    sub_nonneg: dict[Var, str] = {}
    for k, (v, val) in enumerate(zip(sub.jaxpr.invars, args)):
        sub_env[v] = val
        if arg_nonneg is not None and arg_nonneg[k] is not None:
            sub_nonneg[v] = arg_nonneg[k]  # type: ignore[assignment]
    _eval_eqns(sub.jaxpr.eqns, sub_env, rv, load_mask, block, load_cache, sub_nonneg)
    return [
        _const_to_reg(a.val) if isinstance(a, Literal) else sub_env[a]
        for a in sub.jaxpr.outvars
    ]


def _eval_eqns(
    eqns: Any,
    env: dict[Var, _MemView | _RegTensor],
    rv: RefView,
    load_mask: jax.Array,
    block: int,
    load_cache: dict[Any, _RegTensor] | None = None,
    nonneg_tags: dict[Var, str] | None = None,
) -> None:
    """Execute a sequence of jaxpr equations in Triton registers."""

    def read(atom: Any) -> _MemView | _RegTensor:
        return _const_to_reg(atom.val) if isinstance(atom, Literal) else env[atom]

    lt_zero: set[Var] = set()

    for eqn in eqns:
        prim = eqn.primitive
        name = prim.name
        params = eqn.params
        args = [read(a) for a in eqn.invars]
        out_avals = [v.aval for v in eqn.outvars]

        if name in ("pjit", "jit", "custom_jvp_call", "custom_vjp_call"):
            sub: ClosedJaxpr = params.get("jaxpr") or params["call_jaxpr"]
            arg_nn = (
                [
                    None if isinstance(a, Literal) else nonneg_tags.get(a)
                    for a in eqn.invars
                ]
                if nonneg_tags
                else None
            )
            outs: list[_MemView | _RegTensor] = _eval_subjaxpr(
                sub, args, rv, load_mask, block, load_cache, arg_nn
            )

        elif name == "cond":
            branches: tuple[ClosedJaxpr, ...] = params["branches"]
            if len(branches) > _MAX_COND_BRANCHES:
                raise _UnsupportedTritonOp("too many cond branches")
            which = _materialize(args[0], rv).scalar()
            which_i = which.astype(jnp.int32)
            arg_nn = (
                [
                    None if isinstance(a, Literal) else nonneg_tags.get(a)
                    for a in eqn.invars[1:]
                ]
                if nonneg_tags
                else None
            )
            br_vals: list[list[_RegTensor]] = []
            for b_i, br in enumerate(branches):
                br_mask = load_mask & (which_i == b_i)
                br_cache = dict(load_cache) if load_cache is not None else None
                if len(br.jaxpr.eqns) > 16 and not isinstance(rv, _DryRunRefView):
                    out_avals_br: list[Any] = [v.aval for v in br.jaxpr.outvars]
                    out_shapes_br = [tuple(a.shape) for a in out_avals_br]
                    out_dtypes_br = [a.dtype for a in out_avals_br]

                    def _eval_br(
                        _br: ClosedJaxpr = br,
                        _bm: jax.Array = br_mask,
                        _bc: dict[Any, _RegTensor] | None = br_cache,
                    ) -> tuple[jax.Array, ...]:
                        res_list = [
                            _materialize(o, rv)
                            for o in _eval_subjaxpr(
                                _br, args[1:], rv, _bm, block, _bc, arg_nn
                            )
                        ]
                        return tuple(
                            _as_block(e, block) for r in res_list for e in r.flat()
                        )

                    def _zero_br(
                        _shapes: list[tuple[int, ...]] = out_shapes_br,
                        _dtypes: list[Any] = out_dtypes_br,
                    ) -> tuple[jax.Array, ...]:
                        return tuple(
                            jnp.zeros((block,), dtype=dt)
                            for s, dt in zip(_shapes, _dtypes)
                            for _ in range(math.prod(s))
                        )

                    flat_out = jax.lax.cond(
                        jnp.sum(br_mask.astype(jnp.int32)) != 0,
                        _eval_br,
                        _zero_br,
                    )
                    unpacked: list[_RegTensor] = []
                    pos_o = 0
                    for s, dt in zip(out_shapes_br, out_dtypes_br):
                        cnt = math.prod(s)
                        unpacked.append(
                            _reg(s, dt, list(flat_out[pos_o : pos_o + cnt]))
                        )
                        pos_o += cnt
                    br_vals.append(unpacked)
                else:
                    br_vals.append(
                        [
                            _materialize(o, rv)
                            for o in _eval_subjaxpr(
                                br, args[1:], rv, br_mask, block, br_cache, arg_nn
                            )
                        ]
                    )
            outs = []
            for o_idx, b0 in enumerate(br_vals[0]):
                cur = b0.flat()
                for b_i in range(1, len(branches)):
                    sel = which_i == b_i
                    bi = br_vals[b_i][o_idx].flat()
                    cur = [jnp.where(sel, bi[k], cur[k]) for k in range(len(cur))]
                outs.append(_reg(b0.shape, b0.dtype, cur))

        elif name == "scan":
            length = int(params["length"])
            if length > _MAX_UNROLL_SCAN or params.get("reverse", False):
                raise _UnsupportedTritonOp("unsupported scan")
            nc, nk = int(params["num_consts"]), int(params["num_carry"])
            scan_consts = args[:nc]
            carry = [_materialize(a, rv) for a in args[nc : nc + nk]]
            xs_args = args[nc + nk :]
            ys_list: list[list[_RegTensor]] = [[] for _ in range(len(eqn.outvars) - nk)]
            for step in range(length):
                xs_step: list[_MemView | _RegTensor] = [
                    _MemView(
                        x.ref_name,
                        x.shape[1:],
                        x.strides[1:],
                        x.offset + step * x.strides[0],
                        x.dtype,
                        x.is_bool,
                    )
                    if isinstance(x, _MemView)
                    else _RegTensor(x.dtype, x.lanes[step])
                    for x in xs_args
                ]
                step_outs = _eval_subjaxpr(
                    params["jaxpr"], scan_consts + carry + xs_step, rv, load_mask, block
                )
                carry = [_materialize(o, rv) for o in step_outs[:nk]]
                for y_i, o in enumerate(step_outs[nk:]):
                    ys_list[y_i].append(_materialize(o, rv))
            outs = list(carry)
            for y_i, y_steps in enumerate(ys_list):
                out_s = tuple(out_avals[nk + y_i].shape)
                _check_reg_size(out_s)
                if y_steps:
                    outs.append(
                        _RegTensor(
                            y_steps[0].dtype,
                            np.stack([y.lanes for y in y_steps], axis=0),
                        )
                    )
                else:
                    outs.append(_reg(out_s, out_avals[nk + y_i].dtype, []))

        elif name == "stop_gradient":
            outs = [args[0]]

        elif name == "slice":
            a0 = args[0]
            starts = tuple(params["start_indices"])
            limits = tuple(params["limit_indices"])
            steps = params.get("strides") or (1,) * len(starts)
            out_shape = tuple(out_avals[0].shape)
            if isinstance(a0, _MemView):
                view = _MemView(
                    a0.ref_name,
                    out_shape,
                    tuple(bs * st for bs, st in zip(a0.strides, steps)),
                    a0.offset + _ravel(starts, a0.strides),
                    a0.dtype,
                    a0.is_bool,
                )
                outs = [_materialize(view, rv) if view.size <= _MAX_REG_ELEMS else view]
            else:
                _check_reg_size(out_shape)
                sl = tuple(
                    slice(s, lim, st) for s, lim, st in zip(starts, limits, steps)
                )
                outs = [_RegTensor(a0.dtype, a0.lanes[sl])]

        elif name == "squeeze":
            a0 = args[0]
            dims = tuple(params["dimensions"])
            if isinstance(a0, _MemView):
                dim_set = set(dims)
                outs = [
                    _MemView(
                        a0.ref_name,
                        tuple(out_avals[0].shape),
                        tuple(s for d, s in enumerate(a0.strides) if d not in dim_set),
                        a0.offset,
                        a0.dtype,
                        a0.is_bool,
                    )
                ]
            else:
                outs = [_RegTensor(a0.dtype, np.squeeze(a0.lanes, axis=dims))]

        elif name == "transpose":
            a0 = args[0]
            perm = tuple(params["permutation"])
            if isinstance(a0, _MemView):
                outs = [
                    _MemView(
                        a0.ref_name,
                        tuple(out_avals[0].shape),
                        tuple(a0.strides[p] for p in perm),
                        a0.offset,
                        a0.dtype,
                        a0.is_bool,
                    )
                ]
            else:
                outs = [_RegTensor(a0.dtype, np.transpose(a0.lanes, perm))]

        elif name == "broadcast_in_dim":
            a0 = args[0]
            bshape = tuple(params["shape"])
            bdims = tuple(params["broadcast_dimensions"])
            if isinstance(a0, _MemView) and math.prod(bshape) > _MAX_REG_ELEMS:
                nst = [0] * len(bshape)
                for src_d, dst_d in enumerate(bdims):
                    if a0.shape[src_d] == bshape[dst_d]:
                        nst[dst_d] = a0.strides[src_d]
                outs = [
                    _MemView(
                        a0.ref_name, bshape, tuple(nst), a0.offset, a0.dtype, a0.is_bool
                    )
                ]
            else:
                _check_reg_size(bshape)
                r0 = _materialize(a0, rv)
                exp_shape = [1] * len(bshape)
                for src_d, dst_d in enumerate(bdims):
                    exp_shape[dst_d] = r0.shape[src_d]
                outs = [
                    _RegTensor(
                        r0.dtype, np.broadcast_to(r0.lanes.reshape(exp_shape), bshape)
                    )
                ]

        elif name == "reshape":
            a0 = args[0]
            new_shape = tuple(params["new_sizes"])
            if isinstance(a0, _MemView) and math.prod(new_shape) > _MAX_REG_ELEMS:
                if params.get("dimensions") is not None or not a0.is_contiguous():
                    raise _UnsupportedTritonOp("non-contiguous large reshape")
                outs = [
                    _MemView(
                        a0.ref_name,
                        new_shape,
                        _c_strides(new_shape),
                        a0.offset,
                        a0.dtype,
                        a0.is_bool,
                    )
                ]
            else:
                _check_reg_size(new_shape)
                r0 = _materialize(a0, rv)
                dims_opt = params.get("dimensions")
                lanes = (
                    np.transpose(r0.lanes, tuple(dims_opt))
                    if dims_opt is not None
                    else r0.lanes
                )
                outs = [_RegTensor(r0.dtype, lanes.reshape(new_shape))]

        elif name in ("concatenate", "stack"):
            out_shape = tuple(out_avals[0].shape)
            _check_reg_size(out_shape)
            regs = [_materialize(a, rv) for a in args]
            fn_np = np.concatenate if name == "concatenate" else np.stack
            ax_key = "dimension" if name == "concatenate" else "axis"
            outs = [
                _RegTensor(
                    out_avals[0].dtype,
                    fn_np([r.lanes for r in regs], axis=int(params[ax_key])),
                )
            ]

        elif name == "rev":
            r0 = _materialize(args[0], rv)
            outs = [
                _RegTensor(
                    r0.dtype, np.flip(r0.lanes, axis=tuple(params["dimensions"]))
                )
            ]

        elif name == "pad":
            r0 = _materialize(args[0], rv)
            pv = _materialize(args[1], rv).scalar()
            out_shape = tuple(out_avals[0].shape)
            _check_reg_size(out_shape)
            out_arr = np.empty(out_shape, dtype=object)
            out_arr.fill(pv)
            src_slices: list[slice] = []
            dst_slices: list[slice] = []
            empty_pad = False
            for d, (lo, _hi, interior) in enumerate(params["padding_config"]):
                step = interior + 1
                s_lo = max(0, (-lo + step - 1) // step) if lo < 0 else 0
                d_lo = lo + s_lo * step
                if d_lo >= out_shape[d] or s_lo >= r0.shape[d]:
                    empty_pad = True
                    break
                n_copy = min(
                    r0.shape[d] - s_lo, (out_shape[d] - d_lo + step - 1) // step
                )
                src_slices.append(slice(s_lo, s_lo + n_copy))
                dst_slices.append(slice(d_lo, d_lo + n_copy * step, step))
            if not empty_pad:
                out_arr[tuple(dst_slices)] = r0.lanes[tuple(src_slices)]
            outs = [_RegTensor(r0.dtype, out_arr)]

        elif name == "iota":
            out_shape = tuple(params["shape"])
            ax, dt = int(params["dimension"]), params["dtype"]
            elems = [
                jnp.asarray(c[ax], dtype=dt)
                for c in itertools.product(*map(range, out_shape))
            ]
            outs = [_reg(out_shape, dt, elems)]

        elif name == "dynamic_slice":
            op = args[0]
            slice_sizes = tuple(params["slice_sizes"])
            _check_reg_size(slice_sizes)
            ckey: Any = None
            if isinstance(op, _MemView) and load_cache is not None:
                skeys: list[Any] = []
                for a in eqn.invars[1:]:
                    if isinstance(a, Literal):
                        skeys.append(("c", int(a.val)))
                    elif nonneg_tags and a in nonneg_tags:
                        skeys.append(("t", nonneg_tags[a]))
                    else:
                        skeys = []
                        break
                if skeys:
                    ckey = (
                        op.ref_name,
                        op.offset,
                        op.strides,
                        slice_sizes,
                        tuple(skeys),
                    )
            if ckey is not None and ckey in load_cache:  # type: ignore[operator]
                outs = [load_cache[ckey]]  # type: ignore[index]
            else:
                clamped = [
                    _clip_index(
                        _materialize(a, rv).scalar(), op.shape[d] - slice_sizes[d]
                    )
                    for d, a in enumerate(args[1:])
                ]
                op_strides = (
                    op.strides if isinstance(op, _MemView) else _c_strides(op.shape)
                )
                base_addr: int | jax.Array = (
                    op.offset if isinstance(op, _MemView) else 0
                )
                for c_val, st_val in zip(clamped, op_strides):
                    if st_val != 0:
                        base_addr = base_addr + c_val * st_val
                elems = []
                for loc in itertools.product(*map(range, slice_sizes)):
                    addr = base_addr + _ravel(loc, op_strides)
                    if isinstance(op, _MemView):
                        if isinstance(addr, int):
                            v = rv.scalar(op.ref_name, addr)
                            elems.append(v != 0 if op.is_bool else v)
                        else:
                            elems.append(_load_mem_elem(op, addr, rv, load_mask))
                    else:
                        elems.append(
                            op.flat()[addr]
                            if isinstance(addr, int)
                            else _lookup_reg_elem(op, addr)
                        )
                res_reg = _reg(slice_sizes, op.dtype, elems)
                if ckey is not None and load_cache is not None:
                    load_cache[ckey] = res_reg
                outs = [res_reg]

        elif name == "dynamic_update_slice":
            op_r = _materialize(args[0], rv)
            upd_r = _materialize(args[1], rv)
            clamped = [
                _clip_index(
                    _materialize(a, rv).scalar(), op_r.shape[d] - upd_r.shape[d]
                )
                for d, a in enumerate(args[2:])
            ]
            op_strides = _c_strides(op_r.shape)
            elems = op_r.flat()
            for u_coord in itertools.product(*map(range, upd_r.shape)):
                u_val = upd_r.lanes[u_coord]
                dst_flat: int | jax.Array = 0
                for d, (uc, st_val) in enumerate(zip(u_coord, op_strides)):
                    dst_flat = dst_flat + (clamped[d] + uc) * st_val
                for k in range(len(elems)):
                    elems[k] = jnp.where(dst_flat == k, u_val, elems[k])
            outs = [_reg(op_r.shape, op_r.dtype, elems)]

        elif name == "gather":
            outs = [
                _eval_gather(
                    args[0],
                    _materialize(args[1], rv),
                    params,
                    tuple(out_avals[0].shape),
                    rv,
                    load_mask,
                )
            ]

        elif name in _SCATTER_OPS:
            outs = [
                _eval_scatter(
                    name,
                    _materialize(args[0], rv),
                    _materialize(args[1], rv),
                    _materialize(args[2], rv),
                    params,
                )
            ]

        elif name == "dot_general":
            lhs = _materialize(args[0], rv)
            rhs = _materialize(args[1], rv)
            (lhs_c, rhs_c), (lhs_b, rhs_b) = params["dimension_numbers"]
            lhs_free = [
                d for d in range(len(lhs.shape)) if d not in lhs_c and d not in lhs_b
            ]
            rhs_free = [
                d for d in range(len(rhs.shape)) if d not in rhs_c and d not in rhs_b
            ]
            out_shape = tuple(out_avals[0].shape)
            out_dt = out_avals[0].dtype
            _check_reg_size(out_shape)
            nb = math.prod(lhs.shape[d] for d in lhs_b)
            nlf = math.prod(lhs.shape[d] for d in lhs_free)
            nrf = math.prod(rhs.shape[d] for d in rhs_free)
            nk_c = math.prod(lhs.shape[d] for d in lhs_c)
            L = np.transpose(lhs.lanes, (*lhs_b, *lhs_free, *lhs_c)).reshape(
                nb, nlf, nk_c
            )
            R = np.transpose(rhs.lanes, (*rhs_b, *rhs_free, *rhs_c)).reshape(
                nb, nrf, nk_c
            )
            elems = []
            for b_i in range(nb):
                for l_i in range(nlf):
                    for r_i in range(nrf):
                        if nk_c == 0:
                            elems.append(jnp.asarray(0, dtype=out_dt))
                        else:
                            acc = L[b_i, l_i, 0].astype(out_dt) * R[b_i, r_i, 0].astype(
                                out_dt
                            )
                            for k_i in range(1, nk_c):
                                acc = acc + L[b_i, l_i, k_i].astype(out_dt) * R[
                                    b_i, r_i, k_i
                                ].astype(out_dt)
                            elems.append(acc)
            outs = [_reg(out_shape, out_dt, elems)]

        elif name in _REDUCE_OPS:
            r0 = _materialize(args[0], rv)
            red_axes = tuple(sorted(params["axes"]))
            out_shape = tuple(out_avals[0].shape)
            op_fn = _REDUCE_OPS[name]
            moved = np.moveaxis(r0.lanes, red_axes, range(-len(red_axes), 0))
            flat_2d = moved.reshape(math.prod(out_shape), -1)
            outs = [
                _reg(
                    out_shape,
                    out_avals[0].dtype,
                    [functools.reduce(op_fn, row) for row in flat_2d],
                )
            ]

        elif name in ("argmax", "argmin"):
            r0 = _materialize(args[0], rv)
            ax, idx_dt = int(params["axes"][0]), params["index_dtype"]
            out_shape = tuple(out_avals[0].shape)
            flat_2d = np.moveaxis(r0.lanes, ax, -1).reshape(math.prod(out_shape), -1)
            elems = []
            for row in flat_2d:
                best_val, best_idx = row[0], jnp.asarray(0, dtype=idx_dt)
                for k in range(1, len(row)):
                    better = (
                        (row[k] > best_val) if name == "argmax" else (row[k] < best_val)
                    )
                    best_val = jnp.where(better, row[k], best_val)
                    best_idx = jnp.where(better, jnp.asarray(k, dtype=idx_dt), best_idx)
                elems.append(best_idx)
            outs = [_reg(out_shape, idx_dt, elems)]

        elif name == "round":
            r0 = _materialize(args[0], rv)
            rm = params.get("rounding_method")
            if rm is not None and "AWAY" in str(rm).upper():
                elems = [
                    jnp.where(e >= 0.0, jnp.floor(e + 0.5), jnp.ceil(e - 0.5))
                    for e in r0.flat()
                ]
            else:
                elems = [triton_rint(e) for e in r0.flat()]
            outs = [_reg(r0.shape, r0.dtype, elems)]

        elif name == "integer_pow":
            r0 = _materialize(args[0], rv)
            y = int(params["y"])
            outs = [
                _reg(r0.shape, r0.dtype, [_eval_integer_pow(e, y) for e in r0.flat()])
            ]

        elif name in ("convert_element_type", "bitcast_convert_type"):
            r0 = _materialize(args[0], rv)
            dt = params["new_dtype"]
            fn_cv = (
                (lambda e: e.astype(dt))
                if name == "convert_element_type"
                else (lambda e: jax.lax.bitcast_convert_type(e, dt))
            )
            outs = [_reg(r0.shape, dt, [fn_cv(e) for e in r0.flat()])]

        elif name == "select_n":
            if (
                len(args) == 3
                and not isinstance(eqn.invars[0], Literal)
                and eqn.invars[0] in lt_zero
            ):
                # Non-negative particle index: `where(idx < 0, idx + N, idx)` is `idx`.
                outs = [args[1]]
                if (
                    nonneg_tags is not None
                    and not isinstance(eqn.invars[1], Literal)
                    and eqn.invars[1] in nonneg_tags
                ):
                    nonneg_tags[eqn.outvars[0]] = nonneg_tags[eqn.invars[1]]
            else:
                out_shape = tuple(out_avals[0].shape)
                _check_reg_size(out_shape)
                regs = [_materialize(a, rv) for a in args]
                flats = [
                    b.ravel() for b in np.broadcast_arrays(*(r.lanes for r in regs))
                ]
                n_cases = len(regs) - 1
                elems = []
                for idx_e in range(len(flats[0])):
                    w_k = flats[0][idx_e]
                    if n_cases == 2:
                        elems.append(
                            jnp.where(
                                w_k.astype(jnp.bool_), flats[2][idx_e], flats[1][idx_e]
                            )
                        )
                    else:
                        acc_s = flats[1][idx_e]
                        for c_i in range(1, n_cases):
                            acc_s = jnp.where(w_k == c_i, flats[1 + c_i][idx_e], acc_s)
                        elems.append(acc_s)
                outs = [_reg(out_shape, out_avals[0].dtype, elems)]

        elif name in _ELEMENTWISE_PRIMS:
            if (
                name == "lt"
                and nonneg_tags
                and not isinstance(eqn.invars[0], Literal)
                and eqn.invars[0] in nonneg_tags
                and isinstance(eqn.invars[1], Literal)
                and int(eqn.invars[1].val) == 0
            ):
                lt_zero.add(eqn.outvars[0])
            out_shape = tuple(out_avals[0].shape)
            _check_reg_size(out_shape)
            regs = [_materialize(a, rv) for a in args]
            flats = [b.ravel() for b in np.broadcast_arrays(*(r.lanes for r in regs))]
            elems = []
            for idx_e in range(len(flats[0])):
                lane_args = [f[idx_e] for f in flats]
                if any(a.ndim > 0 for a in lane_args):
                    lane_args = [_as_block(a, block) for a in lane_args]
                elems.append(prim.bind(*lane_args, **params))
            outs = [_reg(out_shape, out_avals[0].dtype, elems)]

        else:
            raise _UnsupportedTritonOp(f"unsupported primitive: {name}")

        for ov, oval in zip(eqn.outvars, outs):
            env[ov] = oval


# ---------------------------------------------------------------------------
# Public pre-loop / per-pair evaluators
# ---------------------------------------------------------------------------


def _eval_pre_env(
    compiled: CompiledJaxpr,
    rv: RefView,
    arg_values: dict[int, _RegTensor],
    load_mask: jax.Array,
    block: int,
    nonneg_slots: dict[int, str] | None = None,
) -> dict[Var, _MemView | _RegTensor]:
    jaxpr = compiled.closed_jaxpr.jaxpr
    env: dict[Var, _MemView | _RegTensor] = {}
    nonneg_tags: dict[Var, str] = {}

    for cv, b in zip(jaxpr.constvars, compiled.const_bindings):
        if b.kind == "const_inline":
            env[cv] = _const_to_reg(b.const_val)
        else:
            env[cv] = _MemView(
                b.ref_name, b.shape, b.strides, b.offset, b.dtype, b.is_bool
            )

    for v, b in zip(jaxpr.invars, compiled.in_bindings):
        if b.kind == "arg":
            if b.arg_slot in arg_values:
                env[v] = arg_values[b.arg_slot]
                if nonneg_slots and b.arg_slot in nonneg_slots:
                    nonneg_tags[v] = nonneg_slots[b.arg_slot]
        elif b.kind == "empty":
            if math.prod(b.shape) == 0:
                env[v] = _reg(b.shape, b.dtype, [])
        else:
            view = _MemView(
                b.ref_name, b.shape, b.strides, b.offset, b.dtype, b.is_bool
            )
            env[v] = _materialize(view, rv) if view.size <= _MAX_REG_ELEMS else view

    _eval_eqns(compiled.pre_eqns, env, rv, load_mask, block, {}, nonneg_tags)
    return env


def eval_force_pre(
    spec: ForceSpec,
    rv: RefView,
    i_reg: jax.Array,
    m_i: jax.Array,
) -> dict[Var, _MemView | _RegTensor]:
    """Hoist ``i``-only loads and small scalar/table inputs once before the neighbor loop."""
    return _eval_pre_env(
        spec.compiled,
        rv,
        {0: _reg_scalar(i_reg.dtype, i_reg)},
        m_i,
        int(i_reg.shape[0]),
        nonneg_slots={0: "i"},
    )


def eval_force_pair(
    spec: ForceSpec,
    pre_env: dict[Var, _MemView | _RegTensor],
    rv: RefView,
    j_reg: jax.Array,
    valid: jax.Array,
    *,
    slot: jax.Array | None = None,
    hist_out_ref: Any = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Evaluate one neighbor pair using hoisted ``pre_env``; update history lazily."""
    block = int(j_reg.shape[0])
    jaxpr = spec.compiled.closed_jaxpr.jaxpr
    env = dict(pre_env)

    env[jaxpr.invars[1]] = _reg_scalar(j_reg.dtype, j_reg)
    h_bind = spec.compiled.in_bindings[2]
    h_in_elems: list[jax.Array] = []
    if spec.has_history:
        if slot is not None and rv.has("hist"):
            h_base = slot * spec.hist_width
            h_in_elems = [
                rv.load1(
                    "hist",
                    h_base + c,
                    valid,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                for c in range(spec.hist_width)
            ]
        else:
            h_in_elems = [
                jnp.zeros((block,), dtype=h_bind.dtype) for _ in range(spec.hist_width)
            ]
    env[jaxpr.invars[2]] = _reg(h_bind.shape, h_bind.dtype, h_in_elems)

    nonneg_tags: dict[Var, str] = {jaxpr.invars[0]: "i", jaxpr.invars[1]: "j"}
    _eval_eqns(spec.compiled.loop_eqns, env, rv, valid, block, {}, nonneg_tags)

    def _out_reg(atom: Any) -> _RegTensor:
        return (
            _const_to_reg(atom.val)
            if isinstance(atom, Literal)
            else _materialize(env[atom], rv)
        )

    f_reg = _out_reg(jaxpr.outvars[0])
    t_reg = _out_reg(jaxpr.outvars[1])
    h_reg = _out_reg(jaxpr.outvars[2])

    z_f = jnp.asarray(0.0, dtype=spec.out_dtype)
    df = tuple(
        jnp.where(valid, _as_block(e, block), z_f).astype(spec.out_dtype)
        for e in f_reg.flat()
    )
    if spec.has_torque:
        dt = tuple(
            jnp.where(valid, _as_block(e, block), z_f).astype(spec.out_dtype)
            for e in t_reg.flat()
        )
    else:
        dt = tuple(
            jnp.zeros((block,), dtype=spec.out_dtype) for _ in range(spec.ang_dim)
        )

    if (
        spec.has_history
        and spec.advance_history
        and hist_out_ref is not None
        and slot is not None
    ):
        h_new = [_as_block(e, block) for e in h_reg.flat()]
        changed = jnp.zeros((block,), dtype=jnp.bool_)
        for old_c, new_c in zip(h_in_elems, h_new):
            changed = changed | (new_c != old_c)
        write_mask = valid & changed
        h_base = slot * spec.hist_width

        def _write_hist() -> None:
            for c, val in enumerate(h_new):
                store1(hist_out_ref, rv.b, rv.has_batch, h_base + c, val, write_mask)

        jax.lax.cond(
            jnp.sum(write_mask.astype(jnp.int32)) != 0, _write_hist, lambda: None
        )

    return df, dt


def eval_displacement_pre(
    spec: DisplacementSpec, rv: RefView, block: int
) -> dict[Var, _MemView | _RegTensor]:
    """Hoist domain scalars once per Triton block before cell/pair loops."""
    return _eval_pre_env(
        spec.compiled, rv, {}, jnp.ones((block,), dtype=jnp.bool_), block
    )


def eval_displacement(
    spec: DisplacementSpec,
    pre_env: dict[Var, _MemView | _RegTensor],
    rv: RefView,
    pi: tuple[jax.Array, ...],
    pj: tuple[jax.Array, ...],
    mask: jax.Array,
) -> tuple[jax.Array, ...]:
    """Evaluate ``Domain.displacement(pi, pj, system)`` in Triton registers."""
    block = int(pi[0].shape[0])
    jaxpr = spec.compiled.closed_jaxpr.jaxpr
    env = dict(pre_env)
    env[jaxpr.invars[0]] = _reg((spec.dim,), spec.dtype, list(pi))
    env[jaxpr.invars[1]] = _reg((spec.dim,), spec.dtype, list(pj))
    _eval_eqns(spec.compiled.loop_eqns, env, rv, mask, block)
    out = jaxpr.outvars[0]
    res = (
        _const_to_reg(out.val)
        if isinstance(out, Literal)
        else _materialize(env[out], rv)
    )
    return tuple(_as_block(e, block) for e in res.flat())
