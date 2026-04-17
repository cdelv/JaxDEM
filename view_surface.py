"""Interactive 3D viewer for a surface-sweep result.

Loads a ``.npz`` produced by ``compute_surface_properties`` (augmented with
``central_pos_p`` / ``central_rad`` so the central clump can be drawn) and
renders, in the central's body frame:

- Every sphere of the central clump.
- One marker per approach direction at ``approach_direction * separation``,
  i.e. the tracer COM at the converged contact for that probe, colored by
  center-to-center separation.

Only a single ``(orientation, roll)`` slice is drawn. Mixing orientations in
the same scatter would conflate geometrically distinct contacts, since the
tracer body frame (and therefore what "separation" means as a function of
approach direction) is different per slice.

Usage
-----
    python view_surface.py
    python view_surface.py -i delete-this-data/arad-0.1-trad-0.1.npz
    python view_surface.py --orient 3 --roll 2

Requires plotly. Opens the figure in the default browser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _sphere_surface(
    center: np.ndarray,
    radius: float,
    resolution: int = 18,
    color: str = "lightsteelblue",
    opacity: float = 0.85,
) -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        showscale=False,
        colorscale=[[0.0, color], [1.0, color]],
        opacity=opacity,
        hoverinfo="skip",
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.15, roughness=0.6),
        lightposition=dict(x=100, y=100, z=100),
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        default="delete-this-data/arad-0.1-trad-0.1.npz",
        help="Path to the .npz produced by compute_surface_properties.",
    )
    ap.add_argument(
        "--orient",
        type=int,
        default=0,
        help="Tracer-orientation index to slice (0 .. n_orientations - 1).",
    )
    ap.add_argument(
        "--roll",
        type=int,
        default=0,
        help="Tracer-roll index to slice (0 .. n_rolls - 1).",
    )
    ap.add_argument(
        "--marker-size",
        type=float,
        default=3.0,
        help="Scatter marker size in pixels.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"input file not found: {path}")

    d = np.load(path)
    if int(d["dim"]) != 3:
        raise SystemExit(
            f"view_surface only supports dim=3 results; got dim={int(d['dim'])}."
        )

    required = ("central_pos_p", "central_rad", "approach_directions", "separation")
    missing = [k for k in required if k not in d.files]
    if missing:
        raise SystemExit(
            f"{path} is missing required keys: {missing}. "
            "Re-run the sweep after adding central geometry to np.savez."
        )

    approach = np.asarray(d["approach_directions"])
    separation = np.asarray(d["separation"])
    central_pos_p = np.asarray(d["central_pos_p"])
    central_rad = np.asarray(d["central_rad"])

    n_orient = separation.shape[1]
    n_rolls = separation.shape[2]
    if not (0 <= args.orient < n_orient):
        raise SystemExit(
            f"--orient={args.orient} out of range [0, {n_orient})."
        )
    if not (0 <= args.roll < n_rolls):
        raise SystemExit(f"--roll={args.roll} out of range [0, {n_rolls}).")

    sep_slice = separation[:, args.orient, args.roll]
    tracer_centers = approach * sep_slice[:, None]

    fig = go.Figure()
    for pos, rad in zip(central_pos_p, central_rad):
        fig.add_trace(_sphere_surface(pos, float(rad)))

    fig.add_trace(
        go.Scatter3d(
            x=tracer_centers[:, 0],
            y=tracer_centers[:, 1],
            z=tracer_centers[:, 2],
            mode="markers",
            marker=dict(
                size=args.marker_size,
                color=sep_slice,
                colorscale="Viridis",
                colorbar=dict(title="separation"),
                showscale=True,
            ),
            text=[
                f"idx={i}<br>sep={s:.6g}<br>dir=({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})"
                for i, (s, a) in enumerate(zip(sep_slice, approach))
            ],
            hoverinfo="text",
            name="tracer COM @ contact",
        )
    )

    title = (
        f"{path.name} -- orient={args.orient}/{n_orient}, "
        f"roll={args.roll}/{n_rolls}, n_points={len(approach)}"
    )
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()


if __name__ == "__main__":
    main()
