# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-curve throughput-vs-N log-log scaling sweep and plotting utilities."""

from __future__ import annotations

import gc
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from .cases import (
    COLLIDERS,
    DIMS,
    DOMAINS,
    FORCES,
    HISTORY_COLLIDERS,
    MODES,
    NAIVE_SIZED_COLLIDERS,
    NL_MAX_NEIGHBORS,
    NL_SKIN,
    PROFILES,
    build_spheres_in_box,
)
from .runner import (
    _git_info,
    _run_and_time,
    _warn_if_throttled,
    current_pallas_enabled,
    default_profile,
    get_hardware_label,
    pallas_context,
)

COLLIDER_COLORS: dict[str, str] = {
    "cell_list": "#2563eb",
    "cell_list_static": "#16a34a",
    "multi_cell_list": "#9333ea",
    "multi_cell_list_static": "#0891b2",
    "naive": "#dc2626",
    "nl_cell_list": "#ea580c",
    "nl_cell_list_static": "#ca8a04",
    "nl_multi_cell_list": "#db2777",
    "nl_multi_cell_list_static": "#0d9488",
    "nl_naive": "#6366f1",
}

VARIANT_PALETTE: tuple[str, ...] = (
    "#2563eb",
    "#ea580c",
    "#16a34a",
    "#9333ea",
    "#dc2626",
    "#0891b2",
    "#ca8a04",
    "#db2777",
    "#0d9488",
    "#6366f1",
)

# Axes that can be fixed (shown in title) or varied across curves (shown in legend).
PARAM_ORDER: tuple[str, ...] = (
    "dim",
    "domain",
    "force",
    "mode",
    "pallas",
    "phi",
    "vel",
    "alpha",
    "only_spheres",
)

_MPL_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")
_MPL_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")
_SVG_DASHES: tuple[str, ...] = ("", "6,3", "8,3,2,3", "2,3")


def parse_bool_token(raw: str) -> bool:
    """Parse ``on/off``, ``true/false``, ``1/0``, ``yes/no`` into a bool."""
    tok = raw.strip().lower()
    if tok in ("on", "true", "1", "yes"):
        return True
    if tok in ("off", "false", "0", "no"):
        return False
    raise ValueError(f"Expected boolean token (on/off/true/false/1/0), got {raw!r}.")


def default_ns(
    min_exp: int = 1,
    max_exp: int = 21,
    *,
    step_exp: int = 1,
    min_n: int | None = None,
    max_n: int | None = None,
) -> list[int]:
    """Return powers of two in ``[2**min_exp, 2**max_exp]`` (or ``[min_n, max_n]``)."""
    if step_exp < 1:
        raise ValueError(f"step_exp must be >= 1, got {step_exp}.")
    if min_n is not None:
        if min_n < 2:
            raise ValueError(f"min_n must be >= 2, got {min_n}.")
        min_exp = max(1, math.ceil(math.log2(min_n)))
    if max_n is not None:
        if max_n < 2:
            raise ValueError(f"max_n must be >= 2, got {max_n}.")
        max_exp = max(min_exp, math.floor(math.log2(max_n)))
    if min_exp < 1 or max_exp < min_exp:
        raise ValueError(f"Expected 1 <= min_exp <= max_exp, got {min_exp}..{max_exp}.")
    ns = [1 << e for e in range(min_exp, max_exp + 1, step_exp)]
    if max_n is not None and ns[-1] < max_n and (max_n & (max_n - 1)) == 0:
        ns.append(max_n)
    return ns


def _as_list(val: Any) -> list[Any]:
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


def _fmt_param(key: str, val: Any, batch: int = 6) -> str:
    if key == "dim":
        return str(val).upper()
    if key == "mode":
        return f"batch(B={batch})" if val == "batch" else "single"
    if key == "pallas":
        return "pallas=on" if val else "pallas=off"
    if key == "phi":
        return f"φ={val:g}"
    if key == "vel":
        return f"v={val:g}"
    if key == "alpha":
        return f"α={val:g}"
    if key == "only_spheres":
        return "only_spheres=True" if val else "only_spheres=False"
    return str(val)


def _split_fixed_and_varying(
    curves: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Split curve parameters into fixed (1 unique value -> title) vs varying (>1 -> legend)."""
    fixed: dict[str, Any] = {}
    varying: list[str] = []
    for key in PARAM_ORDER:
        uniq: list[Any] = []
        for c in curves:
            v = c["params"][key]
            if v not in uniq:
                uniq.append(v)
        if len(uniq) == 1:
            fixed[key] = uniq[0]
        elif len(uniq) > 1:
            varying.append(key)
    return fixed, varying


def _curve_label(
    collider: str, params: dict[str, Any], varying: Sequence[str], batch: int
) -> str:
    if not varying:
        return collider
    parts = [_fmt_param(k, params[k], batch=batch) for k in varying]
    return f"{collider} ({', '.join(parts)})"


def _title_from_fixed(record: dict[str, Any]) -> tuple[str, str]:
    fixed = record.get("fixed", {})
    batch = int(record.get("batch", 6))
    parts: list[str] = []
    for key in PARAM_ORDER:
        if key not in fixed:
            continue
        val = fixed[key]
        # Keep default alpha=1 / only_spheres=False compact unless explicitly non-default
        if key == "only_spheres" and not val:
            continue
        parts.append(_fmt_param(key, val, batch=batch))
    main_title = "JaxDEM Throughput Scaling" + (f" — {' · '.join(parts)}" if parts else "")
    sub_title = (
        f"{record['hw']} · steps={record['steps']}/{record.get('steps_naive', 100)} · "
        f"{record['dtype']} · {record['commit']} ({record['date']})"
    )
    return main_title, sub_title


def _is_oom(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return (
        isinstance(exc, MemoryError)
        or "resource_exhausted" in msg
        or "out of memory" in msg
    )


def run_scaling(
    *,
    colliders: Sequence[str] | None = None,
    ns: Sequence[int] | None = None,
    min_exp: int = 1,
    max_exp: int = 21,
    step_exp: int = 1,
    min_n: int | None = None,
    max_n: int | None = None,
    dim: str | Sequence[str] = "3d",
    domain: str | Sequence[str] = "periodic",
    force: str | Sequence[str] = "spring",
    mode: str | Sequence[str] = "single",
    phi: float | Sequence[float] = 0.55,
    vel: float | Sequence[float] = 0.15,
    alpha: float | Sequence[float] = 1.0,
    only_spheres: bool | Sequence[bool] = False,
    pallas: bool | None | Sequence[bool] = None,
    batch_size: int | None = None,
    steps: int | None = None,
    steps_naive: int | None = None,
    profile: str | None = None,
    nl_skin: float = NL_SKIN,
    nl_max_neighbors: int = NL_MAX_NEIGHBORS,
    max_n_naive: int | None = 8_192,
    ignore_naive_limits: bool = False,
    max_seconds: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Measure throughput (M particle-steps/s) vs ``N`` across colliders and varying parameters."""
    prof_name = profile or default_profile()
    preset = PROFILES[prof_name]
    run_batch = batch_size if batch_size is not None else preset["batch"]
    run_steps = steps if steps is not None else preset["steps"]
    run_steps_naive = (
        run_steps
        if ignore_naive_limits
        else (steps_naive if steps_naive is not None else preset["steps_naive"])
    )

    dims = [str(d).strip().lower() for d in _as_list(dim)]
    domains = [str(d).strip().lower() for d in _as_list(domain)]
    forces = [str(f).strip().lower() for f in _as_list(force)]
    modes = [str(m).strip().lower() for m in _as_list(mode)]
    phis = [float(x) for x in _as_list(phi)]
    vels = [float(x) for x in _as_list(vel)]
    alphas = [float(x) for x in _as_list(alpha)]
    spheres_opts = [bool(x) for x in _as_list(only_spheres)]

    if pallas is None:
        pallas_opts = [current_pallas_enabled()]
    else:
        pallas_opts = [bool(x) for x in _as_list(pallas)]

    for d in dims:
        if d not in DIMS:
            raise ValueError(f"Unknown dim {d!r}; expected one of {DIMS}.")
    for d in domains:
        if d not in DOMAINS:
            raise ValueError(f"Unknown domain {d!r}; expected one of {DOMAINS}.")
    for f in forces:
        if f not in FORCES:
            raise ValueError(f"Unknown force {f!r}; expected one of {FORCES}.")
    for m in modes:
        if m not in MODES:
            raise ValueError(f"Unknown mode {m!r}; expected one of {MODES}.")

    if colliders is None:
        chosen_colliders = (
            list(COLLIDERS)
            if "spring" in forces
            else [c for c in COLLIDERS if c in HISTORY_COLLIDERS]
        )
    else:
        chosen_colliders = []
        for c in colliders:
            c_norm = c.strip().lower()
            if c_norm not in COLLIDERS:
                raise ValueError(f"Unknown collider {c!r}; expected one of {COLLIDERS}.")
            chosen_colliders.append(c_norm)

    ns_list = sorted(
        {
            int(x)
            for x in (
                ns
                if ns is not None
                else default_ns(
                    min_exp, max_exp, step_exp=step_exp, min_n=min_n, max_n=max_n
                )
            )
        }
    )
    if not ns_list or ns_list[0] < 2:
        raise ValueError("All N values in scaling sweep must be >= 2.")

    # Build curve list in clean order (collider outer, varying parameters inner).
    curve_specs: list[tuple[str, dict[str, Any]]] = []
    for col in chosen_colliders:
        for d, dom, frc, mod, pal, ph, vl, al, osp in itertools.product(
            dims, domains, forces, modes, pallas_opts, phis, vels, alphas, spheres_opts
        ):
            if frc == "cundallstrack" and col not in HISTORY_COLLIDERS:
                continue
            curve_specs.append(
                (
                    col,
                    {
                        "dim": d,
                        "domain": dom,
                        "force": frc,
                        "mode": mod,
                        "pallas": pal,
                        "phi": ph,
                        "vel": vl,
                        "alpha": al,
                        "only_spheres": osp,
                    },
                )
            )

    if not curve_specs:
        raise ValueError("No valid (collider, force, ...) curves matched the requested parameters.")

    placeholder_curves = [{"collider": c, "params": p} for c, p in curve_specs]
    fixed_params, varying_params = _split_fixed_and_varying(placeholder_curves)

    throttle_warnings = _warn_if_throttled()
    commit, date, dirty = _git_info()
    hw = get_hardware_label()
    dtype = "f64" if jax.config.x64_enabled else "f32"

    # Assign visual styles: same collider ALWAYS keeps the same color; distinguish
    # varying parameter combinations using distinct linestyles and marker shapes.
    uniq_colliders = list(dict.fromkeys(c for c, _ in curve_specs))
    variant_keys = list(
        dict.fromkeys(tuple(p[k] for k in varying_params) for _, p in curve_specs)
    )

    curves: list[dict[str, Any]] = []
    if verbose:
        fixed_str = " · ".join(
            _fmt_param(k, v, batch=run_batch) for k, v in fixed_params.items()
        )
        print(
            f"JaxDEM Scaling Sweep | {hw} | {fixed_str} | "
            f"{len(curve_specs)} curve(s) × N={ns_list[0]:,}..{ns_list[-1]:,} ({len(ns_list)} pts) "
            f"| steps={run_steps}/{run_steps_naive} | {dtype} | commit={commit}{'*' if dirty else ''}"
        )

    t_all = time.perf_counter()
    for curve_idx, (col, params) in enumerate(curve_specs):
        label = _curve_label(col, params, varying_params, run_batch)
        v_idx = variant_keys.index(tuple(params[k] for k in varying_params))
        color = COLLIDER_COLORS.get(
            col, VARIANT_PALETTE[uniq_colliders.index(col) % len(VARIANT_PALETTE)]
        )
        style_idx = v_idx

        if ignore_naive_limits or max_n_naive is None or max_n_naive <= 0:
            naive_cap: int | None = None
        else:
            naive_cap = int(max_n_naive)

        mps_row: list[float | None] = [None] * len(ns_list)
        rb_row: list[int | float | None] = [None] * len(ns_list)
        if verbose:
            print(f"\n  [{curve_idx + 1}/{len(curve_specs)}] {label}", flush=True)

        with pallas_context(params["pallas"]):
            for idx, n_val in enumerate(ns_list):
                if col in NAIVE_SIZED_COLLIDERS and naive_cap is not None and n_val > naive_cap:
                    if verbose:
                        print(f"    N={n_val:>9,d}  SKIP (> max_n_naive={naive_cap:,d})", flush=True)
                    break

                t_case = time.perf_counter()
                jax.clear_caches()
                try:
                    n_arg = n_val * 2 if params["mode"] == "batch" else n_val
                    built = build_spheres_in_box(
                        collider=col,
                        force=params["force"],
                        domain=params["domain"],
                        dim=params["dim"],
                        mode=params["mode"],
                        phi=params["phi"],
                        vel=params["vel"],
                        n=n_arg,
                        n_naive=n_val,
                        n_naive_batch=n_val,
                        batch_size=run_batch,
                        steps=run_steps,
                        steps_naive=run_steps_naive,
                        nl_skin=nl_skin,
                        nl_max_neighbors=nl_max_neighbors,
                        alpha=params["alpha"],
                        only_spheres=params["only_spheres"],
                    )
                    mps, tot_p, act_st, nl_rb = _run_and_time(built)
                    del built
                    mps_row[idx] = mps
                    rb_row[idx] = nl_rb
                    dt_case = time.perf_counter() - t_case
                    if verbose:
                        rb_s = f" ({nl_rb} rb)" if nl_rb is not None else ""
                        print(
                            f"    N={n_val:>9,d} (tot={tot_p:>9,d}, st={act_st:>4d})  "
                            f"{mps:8.2f} M pstep/s{rb_s}  [{dt_case:.1f}s]",
                            flush=True,
                        )
                    if max_seconds is not None and dt_case > max_seconds:
                        if verbose:
                            print(f"    Stopping {label}: exceeded --max-seconds={max_seconds}s.", flush=True)
                        break
                except Exception as exc:
                    gc.collect()
                    short_err = str(exc).splitlines()[0][:80]
                    if _is_oom(exc):
                        if verbose:
                            print(f"    N={n_val:>9,d}  STOP (OOM: {short_err})", flush=True)
                        break
                    if verbose:
                        print(f"    N={n_val:>9,d}  SKIP ({type(exc).__name__}: {short_err})", flush=True)

        curve_entry: dict[str, Any] = {
            "label": label,
            "collider": col,
            "params": params,
            "color": color,
            "style_idx": style_idx,
            "mps": mps_row,
        }
        if any(v is not None for v in rb_row):
            curve_entry["nl_rebuilds"] = rb_row
        curves.append(curve_entry)

    if verbose:
        print(f"\nScaling sweep finished in {time.perf_counter() - t_all:.1f}s.")

    return {
        "v": 2,
        "kind": "scaling",
        "commit": commit,
        "date": date,
        "dirty": dirty,
        "hw": hw,
        "profile": prof_name,
        "dtype": dtype,
        "batch": run_batch,
        "steps": run_steps,
        "steps_naive": run_steps_naive,
        "jax": jax.__version__,
        "throttled": bool(throttle_warnings),
        "ns": ns_list,
        "fixed": fixed_params,
        "varying": varying_params,
        "curves": curves,
    }


def _normalize_curves(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Support both v2 (``curves``) and v1 (``series``) scaling records."""
    if "curves" in record:
        return list(record["curves"])
    # Backwards-compat with v1 format
    out: list[dict[str, Any]] = []
    for col, entry in record.get("series", {}).items():
        out.append(
            {
                "label": col,
                "collider": col,
                "params": {
                    "dim": record.get("dim", "3d"),
                    "domain": record.get("domain", "periodic"),
                    "force": record.get("force", "spring"),
                    "mode": record.get("mode", "single"),
                    "pallas": record.get("pallas", True),
                    "phi": record.get("phi", 0.55),
                    "vel": record.get("vel", 0.15),
                    "alpha": record.get("alpha", 1.0),
                    "only_spheres": record.get("only_spheres", False),
                },
                "color": COLLIDER_COLORS.get(col, "#2563eb"),
                "style_idx": 0,
                "mps": entry["mps"],
                **({"nl_rebuilds": entry["nl_rebuilds"]} if "nl_rebuilds" in entry else {}),
            }
        )
    return out


def format_scaling_table(record: dict[str, Any]) -> str:
    """Format a scaling record as an aligned stdout table (N x Curve in M pstep/s)."""
    ns = record.get("ns", [])
    curves = _normalize_curves(record)
    if not ns or not curves:
        return "No scaling data."

    main_title, sub_title = _title_from_fixed(record)
    labels = [c["label"] for c in curves]
    col_w = max(12, max(len(lbl) for lbl in labels) + 2)
    header = f"{'N':>10}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    lines = [f"\n{main_title}\n{sub_title}", header, "-" * len(header)]

    for idx, n_val in enumerate(ns):
        row_vals = [c["mps"][idx] for c in curves]
        if all(v is None for v in row_vals):
            continue
        cells = "".join(
            f"{v:>{col_w}.2f}" if v is not None else f"{'—':>{col_w}}"
            for v in row_vals
        )
        lines.append(f"{n_val:>10,d}{cells}")
    return "\n".join(lines)


def _render_scaling_html(record: dict[str, Any]) -> str:
    """Render a zero-CDN interactive log-log SVG HTML page for a scaling record."""
    curves = _normalize_curves(record)
    main_title, sub_title = _title_from_fixed(record)
    payload = {
        "ns": record.get("ns", []),
        "batch": record.get("batch", 6),
        "fixed": record.get("fixed", {}),
        "title": main_title,
        "subtitle": sub_title,
        "curves": curves,
        "dashes": list(_SVG_DASHES),
    }
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{main_title}</title>
<style>
:root {{
  --bg: #ffffff; --surface: #f8fafc; --border: #e2e8f0;
  --text: #0f172a; --muted: #64748b; --accent: #2563eb;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0b1120; --surface: #111827; --border: #1f2937;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #3b82f6;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 1.25rem; max-width: 980px; margin-inline: auto;
  background: var(--bg); color: var(--text);
  font-family: system-ui, -apple-system, sans-serif;
}}
h1 {{ font-size: 1.2rem; margin: 0 0 0.25rem; }}
.meta {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 0.85rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.75rem; }}
.leg-item {{
  display: inline-flex; align-items: center; gap: 0.35rem;
  font-size: 0.75rem; padding: 0.18rem 0.52rem; border-radius: 5px;
  border: 1px solid var(--border); background: var(--surface);
  cursor: pointer; user-select: none; font-family: ui-monospace, monospace;
}}
.leg-item:hover {{ border-color: var(--accent); }}
.leg-item.off {{ opacity: 0.35; text-decoration: line-through; }}
.swatch {{ width: 14px; height: 3px; border-radius: 2px; flex-shrink: 0; }}
.card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.85rem; margin-bottom: 1rem;
}}
.wrap {{ position: relative; }}
svg {{ width: 100%; height: auto; display: block; }}
.tip {{
  position: absolute; pointer-events: none; display: none; z-index: 10;
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 7px; padding: 0.45rem 0.65rem; font-size: 0.74rem;
  box-shadow: 0 6px 18px rgba(0,0,0,0.22); min-width: 210px;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 0.77rem; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 0.35rem 0.45rem; border-bottom: 1px solid var(--border); text-align: right; }}
th:first-child, td:first-child {{ text-align: left; font-family: ui-monospace, monospace; }}
th {{ color: var(--muted); font-size: 0.7rem; }}
</style>
</head>
<body>
<div id="root"></div>
<script id="scaling-data" type="application/json">{data_json}</script>
<script>
(function() {{
  const rec = JSON.parse(document.getElementById("scaling-data").textContent);
  const root = document.getElementById("root");
  const ns = rec.ns || [];
  const curves = rec.curves || [];
  const dashes = rec.dashes || ["", "6,3", "8,3,2,3", "2,3"];
  const hidden = new Set();

  function fmtN(n) {{
    if (n >= 1048576) return (n / 1048576).toFixed(n % 1048576 ? 1 : 0) + "M";
    if (n >= 1024) return (n / 1024).toFixed(n % 1024 ? 1 : 0) + "k";
    return String(n);
  }}

  function render() {{
    const visCurves = curves.filter((_, i) => !hidden.has(i));
    const xs = [], ys = [];
    visCurves.forEach(c => {{
      (c.mps || []).forEach((v, i) => {{ if (v != null && v > 0) {{ xs.push(ns[i]); ys.push(v); }} }});
    }});

    const W = 880, H = 420;
    const m = {{ top: 20, right: 24, bottom: 48, left: 64 }};
    const pw = W - m.left - m.right, ph = H - m.top - m.bottom;

    let svgHtml = '<div style="padding:2rem;text-align:center;color:var(--muted)">No visible data</div>';
    let xOf = () => 0;

    if (xs.length && ys.length) {{
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const lx0 = Math.log2(xMin), lx1 = Math.max(lx0 + 1, Math.log2(xMax));
      const ly0 = Math.floor(Math.log10(Math.max(yMin * 0.7, 1e-3)));
      const ly1 = Math.ceil(Math.log10(Math.max(yMax * 1.3, 1e-2)));
      xOf = n => m.left + ((Math.log2(n) - lx0) / (lx1 - lx0)) * pw;
      const yOf = v => m.top + ph * (1 - (Math.log10(v) - ly0) / Math.max(ly1 - ly0, 1e-6));

      let svg = `<svg viewBox="0 0 ${{W}} ${{H}}">`;
      for (let p = ly0; p <= ly1; p++) {{
        const v = Math.pow(10, p);
        const y = yOf(v);
        svg += `<line x1="${{m.left}}" y1="${{y}}" x2="${{W - m.right}}" y2="${{y}}" stroke="var(--border)" stroke-dasharray="3,3"/>`;
        svg += `<text x="${{m.left - 8}}" y="${{y + 4}}" text-anchor="end" font-size="10.5" fill="var(--muted)">${{v >= 1 ? v.toLocaleString() : v}}</text>`;
      }}
      const eStart = Math.ceil(lx0), eEnd = Math.floor(lx1);
      const eStep = (eEnd - eStart) > 12 ? 2 : 1;
      for (let e = eStart; e <= eEnd; e += eStep) {{
        const n = Math.pow(2, e);
        const x = xOf(n);
        svg += `<line x1="${{x}}" y1="${{m.top}}" x2="${{x}}" y2="${{m.top + ph}}" stroke="var(--border)" stroke-dasharray="2,3" opacity="0.55"/>`;
        svg += `<text x="${{x}}" y="${{H - 26}}" text-anchor="middle" font-size="10" fill="var(--muted)">${{fmtN(n)}}</text>`;
      }}
      const xLabel = rec.fixed?.mode === "batch"
        ? `Particles N per sim (B=${{rec.batch}}, log₂ scale)`
        : "Particles N (log₂ scale)";
      svg += `<text x="${{m.left + pw / 2}}" y="${{H - 6}}" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--text)">${{xLabel}}</text>`;
      svg += `<text transform="translate(16,${{m.top + ph / 2}}) rotate(-90)" text-anchor="middle" font-size="11.5" font-weight="600" fill="var(--text)">Throughput (M particle-steps/s, log₁₀)</text>`;

      function markerSvg(x, y, styleIdx, color) {{
        const kind = styleIdx % 4;
        const fx = x.toFixed(1), fy = y.toFixed(1);
        if (kind === 1) {{
          return `<rect x="${{(x - 3.2).toFixed(1)}}" y="${{(y - 3.2).toFixed(1)}}" width="6.4" height="6.4" fill="${{color}}" stroke="var(--bg)" stroke-width="1"/>`;
        }}
        if (kind === 2) {{
          return `<polygon points="${{fx}},${{(y - 4.2).toFixed(1)}} ${{(x - 3.8).toFixed(1)}},${{(y + 3.2).toFixed(1)}} ${{(x + 3.8).toFixed(1)}},${{(y + 3.2).toFixed(1)}}" fill="${{color}}" stroke="var(--bg)" stroke-width="1"/>`;
        }}
        if (kind === 3) {{
          return `<polygon points="${{fx}},${{(y - 4.2).toFixed(1)}} ${{(x + 4.2).toFixed(1)}},${{fy}} ${{fx}},${{(y + 4.2).toFixed(1)}} ${{(x - 4.2).toFixed(1)}},${{fy}}" fill="${{color}}" stroke="var(--bg)" stroke-width="1"/>`;
        }}
        return `<circle cx="${{fx}}" cy="${{fy}}" r="3.4" fill="${{color}}" stroke="var(--bg)" stroke-width="1.1"/>`;
      }}

      visCurves.forEach(c => {{
        const color = c.color || "#2563eb";
        const si = c.style_idx || 0;
        const dash = dashes[si % dashes.length];
        const dashAttr = dash ? ` stroke-dasharray="${{dash}}"` : "";
        const pts = [];
        (c.mps || []).forEach((v, i) => {{
          if (v != null && v > 0) pts.push([xOf(ns[i]), yOf(v)]);
        }});
        if (pts.length >= 2) {{
          svg += `<polyline fill="none" stroke="${{color}}" stroke-width="2.2"${{dashAttr}} stroke-linejoin="round" points="${{pts.map(p => p[0].toFixed(1)+","+p[1].toFixed(1)).join(" ")}}"/>`;
        }}
        pts.forEach(p => {{
          svg += markerSvg(p[0], p[1], si, color);
        }});
      }});

      const colW = pw / Math.max(ns.length, 1);
      ns.forEach((n, idx) => {{
        const x = xOf(n);
        svg += `<rect class="hcol" data-idx="${{idx}}" x="${{Math.max(m.left, x - colW / 2)}}" y="${{m.top}}" width="${{colW}}" height="${{ph}}" fill="transparent"/>`;
      }});
      svg += `<line id="ch" x1="0" y1="${{m.top}}" x2="0" y2="${{m.top + ph}}" stroke="var(--muted)" stroke-dasharray="2,2" opacity="0"/>`;
      svg += `</svg>`;
      svgHtml = svg;
    }}

    const rowsHtml = ns.map((n, i) => {{
      const vals = curves.map(c => c.mps[i]);
      if (vals.every(v => v == null)) return "";
      return `<tr><td>${{n.toLocaleString()}}</td>${{vals.map(v => `<td>${{v != null ? v.toFixed(2) : "—"}}</td>`).join("")}}</tr>`;
    }}).join("");

    root.innerHTML = `
      <h1>${{rec.title}}</h1>
      <div class="meta">${{rec.subtitle}}</div>
      <div class="legend">
        ${{curves.map((c, i) => {{
          const si = c.style_idx || 0;
          const dash = dashes[si % dashes.length];
          const dashAttr = dash ? ` stroke-dasharray="${{dash}}"` : "";
          const mk = si % 4 === 1
            ? `<rect x="7.5" y="3.5" width="5" height="5" fill="${{c.color}}"/>`
            : si % 4 === 2
              ? `<polygon points="10,2.5 6.8,9 13.2,9" fill="${{c.color}}"/>`
              : si % 4 === 3
                ? `<polygon points="10,2.5 13.5,6 10,9.5 6.5,6" fill="${{c.color}}"/>`
                : `<circle cx="10" cy="6" r="2.8" fill="${{c.color}}"/>`;
          const icon = `<svg width="20" height="12" viewBox="0 0 20 12" style="width:20px;height:12px;flex-shrink:0"><line x1="0" y1="6" x2="20" y2="6" stroke="${{c.color}}" stroke-width="2"${{dashAttr}}/>${{mk}}</svg>`;
          return `<span class="leg-item${{hidden.has(i) ? " off" : ""}}" data-ci="${{i}}">${{icon}}${{c.label}}</span>`;
        }}).join("")}}
      </div>
      <div class="card"><div class="wrap" id="wrap">${{svgHtml}}<div class="tip" id="tip"></div></div></div>
      <div class="card" style="overflow-x:auto">
        <table>
          <thead><tr><th>N</th>${{curves.map(c => `<th>${{c.label}}</th>`).join("")}}</tr></thead>
          <tbody>${{rowsHtml}}</tbody>
        </table>
      </div>`;

    root.querySelectorAll("[data-ci]").forEach(el => el.addEventListener("click", () => {{
      const idx = Number(el.dataset.ci);
      if (hidden.has(idx)) hidden.delete(idx); else hidden.add(idx);
      render();
    }}));

    const wrap = document.getElementById("wrap");
    const tip = document.getElementById("tip");
    const ch = document.getElementById("ch");
    root.querySelectorAll(".hcol").forEach(el => {{
      el.addEventListener("mouseenter", e => {{
        const idx = Number(el.dataset.idx);
        const n = ns[idx];
        if (ch) {{ const x = xOf(n); ch.setAttribute("x1", x); ch.setAttribute("x2", x); ch.setAttribute("opacity", "0.75"); }}
        const items = visCurves
          .filter(c => c.mps[idx] != null)
          .map(c => {{
            const v = c.mps[idx];
            const rb = c.nl_rebuilds?.[idx];
            const rbTag = rb != null ? ` <span style="color:var(--muted)">(${{rb}} rb)</span>` : "";
            return `<div style="display:flex;justify-content:space-between;gap:10px"><span><span class="swatch" style="display:inline-block;background:${{c.color}};margin-right:5px;vertical-align:middle"></span>${{c.label}}</span><strong>${{v.toFixed(2)}} M/s${{rbTag}}</strong></div>`;
          }}).join("");
        if (!items) return;
        tip.innerHTML = `<div style="font-weight:700;margin-bottom:3px">N = ${{n.toLocaleString()}}</div>` + items;
        tip.style.display = "block";
        const rect = wrap.getBoundingClientRect();
        tip.style.left = `${{Math.min(Math.max(8, e.clientX - rect.left + 12), Math.max(8, rect.width - 240))}}px`;
        tip.style.top = "10px";
      }});
      el.addEventListener("mouseleave", () => {{
        tip.style.display = "none";
        if (ch) ch.setAttribute("opacity", "0");
      }});
    }});
  }}
  render();
}})();
</script>
</body>
</html>
"""


def plot_scaling(record: dict[str, Any], out_path: Path | str) -> Path:
    """Save a log-log throughput-vs-N plot as interactive HTML (``.html``) or matplotlib image."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".html", ".htm"):
        out.write_text(_render_scaling_html(record), encoding="utf-8")
        return out

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ns = np.asarray(record["ns"], dtype=np.float64)
    curves = _normalize_curves(record)
    main_title, sub_title = _title_from_fixed(record)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
    for curve in curves:
        vals = curve["mps"]
        xs = [n for n, v in zip(ns, vals) if v is not None and v > 0]
        ys = [v for v in vals if v is not None and v > 0]
        if not xs:
            continue
        si = int(curve.get("style_idx", 0))
        ax.loglog(
            xs,
            ys,
            linestyle=_MPL_LINESTYLES[si % len(_MPL_LINESTYLES)],
            marker=_MPL_MARKERS[si % len(_MPL_MARKERS)],
            markersize=4.8,
            linewidth=2.0,
            label=curve["label"],
            color=curve.get("color"),
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)

    def _fmt_n(x: float, _pos: int) -> str:
        if x <= 0:
            return ""
        if x >= 1048576:
            v = x / 1048576.0
            return f"{int(v)}M" if v.is_integer() else f"{v:.1f}M"
        if x >= 1024:
            v = x / 1024.0
            return f"{int(v)}k" if v.is_integer() else f"{v:.1f}k"
        return f"{int(round(x))}"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_n))
    ax.grid(True, which="major", linestyle="--", alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", alpha=0.22)

    fixed = record.get("fixed", {})
    xlabel = (
        f"Particles N per simulation (B={record.get('batch', 6)})"
        if fixed.get("mode") == "batch"
        else "Particles N"
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Throughput (M particle-steps / s)", fontsize=11)
    ax.set_title(f"{main_title}\n{sub_title}", fontsize=10.2)
    ncol = 2 if len(curves) > 5 else 1
    ax.legend(frameon=True, fontsize=8.5, ncol=ncol)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
