# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Utilities to generate step indices for trajectory logging.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def make_save_steps_linear(
    *,
    num_steps: int,
    save_freq: int,
    include_step0: bool = True,
) -> np.ndarray:
    num_steps = int(num_steps)
    save_freq = int(save_freq)
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    if save_freq < 1:
        raise ValueError("save_freq must be >= 1")

    start = 0 if include_step0 else save_freq
    if start > num_steps:
        return np.zeros((0,), dtype=np.int32)
    return np.arange(start, num_steps + 1, save_freq, dtype=np.int32)


def make_save_steps_pseudolog(
    *,
    num_steps: int,
    reset_save_decade: int,
    min_save_decade: int,
    decade: int = 10,
    include_step0: bool = True,
    cap: Optional[int] = None,
) -> np.ndarray:
    """
    Pseudo-log schedule compatible with the BaseLogGroup logic.

    Parameters are interpreted on the integer timestep grid 0..num_steps (inclusive).
    """
    num_steps = int(num_steps)
    reset = int(reset_save_decade)
    f0 = int(min_save_decade)
    decade = int(decade)
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    if reset < 1:
        raise ValueError("reset_save_decade must be >= 1")
    if f0 < 1:
        raise ValueError("min_save_decade must be >= 1")
    if decade < 2:
        raise ValueError("decade must be >= 2")

    out: list[int] = []
    max_block = num_steps // reset
    for b in range(max_block + 1):
        base = b * reset
        off_min = 0 if (b == 0 and include_step0) else 1
        off_max = min(reset, num_steps - base)
        if off_min > off_max:
            continue

        k = 0
        while True:
            freq = f0 * (decade**k)
            region_end = min(off_max, f0 * (decade ** (k + 1)))
            region_start = off_min if k == 0 else max(off_min, f0 * (decade**k) + 1)
            if region_start <= region_end:
                first = ((region_start + freq - 1) // freq) * freq
                for off in range(first, region_end + 1, freq):
                    out.append(base + off)
                    if cap is not None and len(out) >= cap:
                        return np.asarray(sorted(set(out)), dtype=np.int32)

            if region_end >= off_max:
                break
            k += 1

    return np.asarray(sorted(set(out)), dtype=np.int32)
