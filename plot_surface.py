"""Non-interactive 3D plot of a surface-sweep result.

Same content as ``view_surface.py`` but renders with matplotlib and writes a
PNG. See that file's docstring for semantics; the key point is that only a
single ``(orientation, roll)`` slice is drawn per image.

Usage
-----
    python plot_surface.py
    python plot_surface.py -i delete-this-data/arad-0.1-trad-0.1.npz
    python plot_surface.py --orient 3 --roll 2 --out surface.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


def _add_sphere(
    ax: Axes3D,
    center: np.ndarray,
    radius: float,
    *,
    color: str = "lightsteelblue",
    alpha: float = 0.35,
    resolution: int = 16,
) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        x, y, z,
        color=color, alpha=alpha,
        linewidth=0, antialiased=True, shade=True,
    )


def _set_equal_aspect(ax: Axes3D, pts: np.ndarray) -> None:
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    c = 0.5 * (lo + hi)
    r = 0.5 * float((hi - lo).max())
    r = max(r, 1e-12)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))  # matplotlib >= 3.3
    except AttributeError:
        pass


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i", "--input",
        default="delete-this-data/arad-0.1-trad-0.1.npz",
        help="Path to the .npz produced by compute_surface_properties.",
    )
    ap.add_argument("--orient", type=int, default=0)
    ap.add_argument("--roll", type=int, default=0)
    ap.add_argument("--out", default="surface.png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--marker-size",
        type=float,
        default=6.0,
        help="Scatter marker area (matplotlib s=).",
    )
    ap.add_argument(
        "--elev", type=float, default=25.0, help="Camera elevation (deg)."
    )
    ap.add_argument(
        "--azim", type=float, default=35.0, help="Camera azimuth (deg)."
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
            f"plot_surface only supports dim=3 results; got dim={int(d['dim'])}."
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

    fig = plt.figure(figsize=(8, 7))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    for pos, rad in zip(central_pos_p, central_rad):
        _add_sphere(ax, pos, float(rad))

    sc = ax.scatter(
        tracer_centers[:, 0],
        tracer_centers[:, 1],
        tracer_centers[:, 2],
        c=sep_slice,
        cmap="viridis",
        s=args.marker_size,
        depthshade=False,
    )
    cbar = fig.colorbar(sc, ax=ax, label="separation", shrink=0.7, pad=0.08)
    cbar.ax.tick_params(labelsize=8)

    corners = np.concatenate(
        [
            central_pos_p + central_rad[:, None],
            central_pos_p - central_rad[:, None],
            tracer_centers,
        ],
        axis=0,
    )
    _set_equal_aspect(ax, corners)

    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        f"{path.name} -- orient={args.orient}/{n_orient}, "
        f"roll={args.roll}/{n_rolls}, n_points={len(approach)}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
