# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Backend detection, runtime configuration, and ``platform_dependent`` dispatch."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

import jax

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    _HAS_PALLAS_TRITON = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore[assignment]
    pltriton = None  # type: ignore[assignment]
    _HAS_PALLAS_TRITON = False

__all__ = [
    "has_manual_axes",
    "is_pallas_enabled",
    "pallas_enabled",
    "pl",
    "platform_dispatch",
    "pltriton",
    "set_pallas_enabled",
    "shape_dtype",
    "triton_compiler_params",
]

_GPU_REGISTERED = False
_PALLAS_OVERRIDE: bool | None = None


def _env_disables_pallas() -> bool:
    return os.environ.get("JAXDEM_DISABLE_PALLAS", "") == "1"


def _want_pallas() -> bool:
    if _PALLAS_OVERRIDE is not None:
        return _PALLAS_OVERRIDE
    return not _env_disables_pallas()


def _gpu_platform_name(dev: Any) -> str | None:
    """Return ``'cuda'`` or ``'rocm'`` for a GPU device, else ``None``."""
    plat = getattr(dev, "platform", "")
    if plat in ("cuda", "rocm"):
        return plat
    if plat != "gpu":
        return None
    name = type(dev).__name__.lower()
    if "rocm" in name or "amd" in name:
        return "rocm"
    return "cuda"


def _register_unlisted_cuda_gpus(devices: list[Any]) -> None:
    """Register CUDA GPUs absent from JAX's static Triton ``gpu_info`` table."""
    global _GPU_REGISTERED
    if _GPU_REGISTERED:
        return
    _GPU_REGISTERED = True
    try:
        from jax._src.pallas.triton import gpu_info  # type: ignore[attr-defined]

        for dev in devices:
            if (
                _gpu_platform_name(dev) == "cuda"
                and dev.device_kind not in gpu_info.registry
            ):
                cc_str = getattr(dev, "compute_capability", None)
                if cc_str is not None:
                    cc = int(round(float(cc_str) * 10))
                    info = gpu_info.GpuInfo(
                        gpu_version=None,
                        arch_name=str(cc_str),
                        compute_capability=cc,
                    )

                    def _make_info(_i: Any = info) -> Any:
                        return _i

                    gpu_info.registry[dev.device_kind] = _make_info
    except Exception:
        pass


def _active_triton_platforms() -> tuple[str, ...]:
    """Return available GPU platforms (``'cuda'``, ``'rocm'``) enabled for Pallas Triton."""
    if not _HAS_PALLAS_TRITON or not _want_pallas():
        return ()
    try:
        devices = jax.devices()
    except Exception:
        return ()
    plats: list[str] = []
    has_cuda = False
    for dev in devices:
        gplat = _gpu_platform_name(dev)
        if gplat == "cuda":
            has_cuda = True
            if "cuda" not in plats:
                plats.append("cuda")
        elif gplat == "rocm":
            if "rocm" not in plats:
                plats.append("rocm")
    if has_cuda:
        _register_unlisted_cuda_gpus(devices)
    return tuple(plats)


def is_pallas_enabled() -> bool:
    """Return ``True`` when Pallas Triton GPU kernels are available and enabled."""
    return len(_active_triton_platforms()) > 0


def set_pallas_enabled(enabled: bool | None, *, clear_caches: bool = True) -> None:
    """Enable, disable, or reset Pallas Triton GPU kernels at runtime.

    Parameters
    ----------
    enabled : bool or None
        ``True`` enables Pallas Triton kernels on supported CUDA/ROCm backends,
        ``False`` forces the Pure-JAX implementation everywhere, and ``None``
        restores the default behavior controlled by the
        ``JAXDEM_DISABLE_PALLAS`` environment variable.
    clear_caches : bool, optional
        When ``True`` (default) and the effective state changes, call
        ``jax.clear_caches()`` so previously compiled ``@jax.jit`` functions
        (such as :meth:`System.step <jaxdem.system.System.step>`) recompile with
        the new backend choice.
    """
    global _PALLAS_OVERRIDE
    prev = is_pallas_enabled()
    _PALLAS_OVERRIDE = None if enabled is None else bool(enabled)
    if clear_caches and is_pallas_enabled() != prev:
        jax.clear_caches()  # type: ignore[no-untyped-call]


@contextmanager
def pallas_enabled(enabled: bool, *, clear_caches: bool = True) -> Iterator[None]:
    """Temporarily enable or disable Pallas Triton GPU kernels."""
    prev_override = _PALLAS_OVERRIDE
    set_pallas_enabled(enabled, clear_caches=clear_caches)
    try:
        yield
    finally:
        set_pallas_enabled(prev_override, clear_caches=clear_caches)


def has_manual_axes(*objs: Any) -> bool:
    """Return ``True`` if any array leaf carries manual ``shard_map`` axes."""
    for obj in objs:
        for leaf in jax.tree_util.tree_leaves(obj):
            if hasattr(leaf, "dtype"):
                try:
                    t = jax.typeof(leaf)
                    mat = getattr(t, "manual_axis_type", None)
                    if mat is not None:
                        if (
                            getattr(mat, "varying", None)
                            or getattr(mat, "unreduced", None)
                            or getattr(mat, "reduced", None)
                        ):
                            return True
                    elif getattr(t, "vma", None):
                        return True
                except Exception:
                    pass
    return False


def platform_dispatch(
    pallas_fn: Callable[..., Any], pure_fn: Callable[..., Any], *args: Any
) -> Any:
    """Dispatch to ``pallas_fn`` on CUDA/ROCm and ``pure_fn`` on all other backends."""
    plats = _active_triton_platforms()
    if not plats:
        return pure_fn(*args)
    kw: dict[str, Any] = {"default": pure_fn}
    for p in plats:
        kw[p] = pallas_fn
    return jax.lax.platform_dependent(*args, **kw)


def shape_dtype(shape: tuple[int, ...], dtype: Any) -> Any:
    return jax.ShapeDtypeStruct(shape, dtype)  # type: ignore[no-untyped-call]


def triton_compiler_params(num_warps: int = 4, num_stages: int = 2) -> Any:
    return pltriton.CompilerParams(num_warps=num_warps, num_stages=num_stages)
