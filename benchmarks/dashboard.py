# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Zero-CDN, light/dark-aware interactive per-hardware throughput dashboard & Sphinx directive."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .cases import COLLIDERS, DIMS, DOMAINS, FORCES, MODES, REGIME_ORDER, REGIMES
from .store import DEFAULT_HISTORY_PATH, load_history

_DASHBOARD_CSS = """
.jdem-bench {
  --jb-bg: var(--pst-color-background, var(--background, #ffffff));
  --jb-surface: var(--pst-color-surface, var(--card, #f8fafc));
  --jb-border: var(--pst-color-border, var(--border, #e2e8f0));
  --jb-text: var(--pst-color-text-base, var(--foreground, #0f172a));
  --jb-muted: var(--pst-color-text-muted, var(--muted-foreground, #64748b));
  --jb-accent: var(--pst-color-primary, var(--primary, #2563eb));
  --jb-good: #16a34a;
  --jb-bad: #dc2626;
  font-family: var(--pst-font-family-base, var(--font-sans, system-ui, -apple-system, sans-serif));
  color: var(--jb-text);
  margin: 0.4rem 0 1.4rem;
}
.jdem-bench * { box-sizing: border-box; }
.jb-hw-section {
  border: 1px solid var(--jb-border);
  border-radius: 12px;
  padding: 1rem 1.05rem;
  margin-bottom: 1.35rem;
  background: var(--jb-bg);
}
.jb-hw-head {
  display: flex; flex-wrap: wrap; justify-content: space-between;
  align-items: baseline; gap: 0.5rem;
  padding-bottom: 0.65rem; margin-bottom: 0.75rem;
  border-bottom: 1px solid var(--jb-border);
}
.jb-hw-title { font-size: 1.08rem; font-weight: 700; letter-spacing: -0.01em; }
.jb-hw-meta { font-size: 0.76rem; color: var(--jb-muted); }
.jb-controls {
  display: flex; flex-wrap: wrap; gap: 0.45rem 0.75rem;
  align-items: center; margin-bottom: 0.85rem;
}
.jb-ctl-block {
  display: inline-flex; align-items: center; gap: 0.25rem; flex-wrap: wrap;
}
.jb-ctl-label {
  font-size: 0.69rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--jb-muted); margin-right: 0.15rem; font-weight: 600;
}
.jb-pill {
  background: var(--jb-surface); color: var(--jb-muted);
  border: 1px solid var(--jb-border); border-radius: 999px;
  padding: 0.22rem 0.62rem; font-size: 0.76rem; cursor: pointer;
  transition: all 0.12s ease; line-height: 1.3;
}
.jb-pill:hover { border-color: var(--jb-accent); color: var(--jb-text); }
.jb-pill.active {
  background: var(--jb-accent); color: #fff; border-color: var(--jb-accent);
  font-weight: 600;
}
.jb-kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 0.55rem; margin-bottom: 0.85rem;
}
.jb-kpi {
  background: var(--jb-surface); border: 1px solid var(--jb-border);
  border-radius: 8px; padding: 0.58rem 0.75rem;
}
.jb-kpi-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--jb-muted); }
.jb-kpi-value { font-size: 1.12rem; font-weight: 700; margin-top: 0.12rem; font-variant-numeric: tabular-nums; }
.jb-kpi-sub { font-size: 0.72rem; color: var(--jb-muted); margin-top: 0.08rem; }
.jb-legend {
  display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.75rem;
}
.jb-leg-item {
  display: inline-flex; align-items: center; gap: 0.32rem;
  font-size: 0.73rem; padding: 0.16rem 0.46rem; border-radius: 5px;
  border: 1px solid var(--jb-border); background: var(--jb-surface);
  color: var(--jb-text); cursor: pointer; user-select: none;
  font-family: var(--pst-font-family-monospace, var(--font-mono, monospace));
}
.jb-leg-item.off { opacity: 0.35; text-decoration: line-through; }
.jb-swatch { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
.jb-grid-2x2 {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.7rem; margin-bottom: 0.9rem;
}
@media (max-width: 620px) { .jb-grid-2x2 { grid-template-columns: 1fr; } }
.jb-card {
  background: var(--jb-surface); border: 1px solid var(--jb-border);
  border-radius: 9px; padding: 0.7rem 0.8rem;
}
.jb-card-wide { margin-bottom: 0.85rem; }
.jb-card-head {
  display: flex; flex-wrap: wrap; justify-content: space-between;
  align-items: center; gap: 0.4rem; margin-bottom: 0.45rem;
}
.jb-card-title { font-size: 0.86rem; font-weight: 650; }
.jb-card-sub { font-size: 0.71rem; color: var(--jb-muted); }
.jb-chart-wrap { position: relative; width: 100%; }
.jb-chart-svg { width: 100%; height: auto; display: block; }
.jb-tooltip {
  position: absolute; pointer-events: none; display: none; z-index: 25;
  background: var(--jb-bg); color: var(--jb-text);
  border: 1px solid var(--jb-border); border-radius: 7px;
  padding: 0.42rem 0.6rem; font-size: 0.72rem;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22); min-width: 175px;
}
.jb-table-wrap { overflow-x: auto; }
.jb-table {
  width: 100%; border-collapse: collapse; font-size: 0.79rem;
  font-variant-numeric: tabular-nums;
}
.jb-table th {
  text-align: left; font-size: 0.68rem; text-transform: uppercase;
  letter-spacing: 0.02em; color: var(--jb-muted);
  border-bottom: 1px solid var(--jb-border); padding: 0.4rem 0.34rem;
  white-space: nowrap;
}
.jb-table td {
  border-bottom: 1px solid var(--jb-border); padding: 0.4rem 0.34rem;
  vertical-align: middle;
}
.jb-table tr:last-child td { border-bottom: none; }
.jb-num { text-align: right; white-space: nowrap; }
.jb-col-name {
  display: flex; align-items: center; gap: 6px; white-space: nowrap;
  font-weight: 600; font-family: var(--pst-font-family-monospace, var(--font-mono, monospace));
  font-size: 0.76rem;
}
.jb-badge {
  display: inline-block; font-size: 0.7rem; font-weight: 600;
  padding: 0.1rem 0.42rem; border-radius: 999px; white-space: nowrap;
}
.jb-badge-good { background: rgba(22, 163, 74, 0.14); color: var(--jb-good); }
.jb-badge-bad { background: rgba(220, 38, 38, 0.14); color: var(--jb-bad); }
.jb-badge-flat { background: rgba(100, 116, 139, 0.14); color: var(--jb-muted); }
.jb-warn {
  font-size: 0.74rem; color: #b45309; background: rgba(245, 158, 11, 0.14);
  border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 6px;
  padding: 0.32rem 0.6rem; margin-bottom: 0.7rem;
}
.jb-empty {
  padding: 1.4rem; text-align: center; color: var(--jb-muted);
  background: var(--jb-surface); border: 1px dashed var(--jb-border); border-radius: 8px;
  font-size: 0.84rem;
}
"""

_DASHBOARD_JS = r"""
(function() {
  const root = document.getElementById("jdem-bench-root");
  if (!root) return;
  const payload = JSON.parse(document.getElementById("jdem-bench-data").textContent);
  const runs = payload.runs || [];
  const DIMS = payload.dims || ["3d", "2d"];
  const DOMAINS = payload.domains || ["periodic", "leesedwards", "reflect", "reflectsphere"];
  const COLLIDERS = payload.colliders || [];
  const FORCES = payload.forces || ["spring", "cundallstrack"];
  const MODES = payload.modes || ["single", "batch"];
  const REGIMES = payload.regimes || {};
  const REGIME_ORDER = payload.regimeOrder || Object.keys(REGIMES);

  if (!runs.length) {
    root.innerHTML = '<div class="jb-empty">No benchmark runs recorded yet. Run <code>python -m benchmarks --record</code> to populate <code>benchmarks/history.jsonl</code>.</div>';
    return;
  }

  const COLORS = {
    cell_list: "#2563eb",
    cell_list_static: "#16a34a",
    multi_cell_list: "#9333ea",
    multi_cell_list_static: "#0891b2",
    naive: "#dc2626",
    nl_cell_list: "#ea580c",
    nl_cell_list_static: "#ca8a04",
    nl_multi_cell_list: "#db2777",
    nl_multi_cell_list_static: "#0d9488",
    nl_naive: "#6366f1"
  };

  // Group history into separate sections per hardware (+ profile/dtype/only_spheres).
  const hwMap = new Map();
  runs.forEach(r => {
    const tag = r.only_spheres ? "only_spheres" : "clump-equiv";
    const key = `${r.hw}||${r.profile}||${r.dtype}||${tag}`;
    if (!hwMap.has(key)) hwMap.set(key, { hw: r.hw, profile: r.profile, dtype: r.dtype, tag, runs: [] });
    hwMap.get(key).runs.push(r);
  });

  function getSlice(run, dim, dom, frc, mod) {
    return (((run.data || {})[dim] || {})[dom] || {})[frc]?.[mod] || {};
  }

  function niceTicks(minV, maxV, isLog) {
    if (isLog) {
      const lo = Math.floor(Math.log10(Math.max(minV, 1e-2)));
      const hi = Math.ceil(Math.log10(Math.max(maxV, 1e-1)));
      const out = [];
      for (let p = lo; p <= hi; p++) out.push(Math.pow(10, p));
      if (out.length < 3) {
        const dense = [];
        for (let p = lo; p < hi; p++) { dense.push(Math.pow(10, p)); dense.push(3 * Math.pow(10, p)); }
        dense.push(Math.pow(10, hi));
        return dense;
      }
      return out;
    }
    const ticks = [];
    const span = Math.max(maxV - minV, 1e-6);
    for (let i = 0; i <= 4; i++) ticks.push(minV + (span * i) / 4);
    return ticks;
  }

  function fmtDelta(curr, prev) {
    if (prev == null || prev <= 0 || curr == null) return { text: "baseline", cls: "jb-badge-flat" };
    const pct = ((curr - prev) / prev) * 100;
    const sign = pct > 0 ? "+" : "";
    const cls = pct > 1.0 ? "jb-badge-good" : (pct < -5.0 ? "jb-badge-bad" : "jb-badge-flat");
    return { text: `${sign}${pct.toFixed(1)}%`, cls };
  }

  function renderRegimeProgressionSvg(hwRuns, state, cols, regIdx) {
    const W = 340, H = 185;
    const m = { top: 14, right: 26, bottom: 34, left: 42 };
    const pw = W - m.left - m.right;
    const ph = H - m.top - m.bottom;
    const visCols = cols.filter(c => !state.hidden.has(c));

    const vals = [];
    hwRuns.forEach(r => {
      const sl = getSlice(r, state.dim, state.domain, state.force, state.mode);
      visCols.forEach(c => {
        const v = sl[c]?.[regIdx];
        if (v != null && v > 0) vals.push(v);
      });
    });

    if (!vals.length) {
      return { svg: `<div class="jb-empty" style="padding:1.1rem">No data for this selection</div>`, xOf: () => 0 };
    }

    const minV = Math.min(...vals), maxV = Math.max(...vals);
    const useLog = state.scale === "log" && minV > 0;
    const yLo = useLog ? Math.pow(10, Math.floor(Math.log10(minV * 0.8))) : 0;
    const yHi = useLog ? Math.pow(10, Math.ceil(Math.log10(maxV * 1.2))) : maxV * 1.15;

    const xOf = idx => hwRuns.length === 1 ? m.left + pw / 2 : m.left + (idx / (hwRuns.length - 1)) * pw;
    const yOf = v => {
      if (useLog) {
        const t = (Math.log10(Math.max(v, yLo)) - Math.log10(yLo)) / Math.max(Math.log10(yHi) - Math.log10(yLo), 1e-6);
        return m.top + ph * (1 - t);
      }
      return m.top + ph * (1 - (v - yLo) / Math.max(yHi - yLo, 1e-6));
    };

    let svg = `<svg class="jb-chart-svg" viewBox="0 0 ${W} ${H}">`;
    niceTicks(yLo, yHi, useLog).forEach(t => {
      const y = yOf(t);
      if (y >= m.top - 2 && y <= m.top + ph + 2) {
        svg += `<line x1="${m.left}" y1="${y}" x2="${W - m.right}" y2="${y}" stroke="var(--jb-border)" stroke-dasharray="2.5,2.5"/>`;
        svg += `<text x="${m.left - 6}" y="${y + 3}" text-anchor="end" font-size="9" fill="var(--jb-muted)">${t >= 10 ? t.toFixed(0) : t.toFixed(1)}</text>`;
      }
    });

    const stepX = Math.max(1, Math.ceil(hwRuns.length / 4));
    hwRuns.forEach((r, idx) => {
      if (idx % stepX === 0 || idx === hwRuns.length - 1) {
        const x = xOf(idx);
        svg += `<text x="${x}" y="${H - 18}" text-anchor="middle" font-size="8.5" font-family="var(--pst-font-family-monospace,monospace)" fill="var(--jb-text)">${r.commit}</text>`;
        svg += `<text x="${x}" y="${H - 7}" text-anchor="middle" font-size="8" fill="var(--jb-muted)">${r.date.slice(5)}</text>`;
      }
    });

    visCols.forEach(c => {
      const color = COLORS[c] || "#2563eb";
      const pts = [];
      hwRuns.forEach((r, idx) => {
        const v = getSlice(r, state.dim, state.domain, state.force, state.mode)[c]?.[regIdx];
        if (v != null) pts.push([xOf(idx), yOf(v)]);
      });
      if (pts.length >= 2) {
        svg += `<polyline fill="none" stroke="${color}" stroke-width="1.9" stroke-linejoin="round" points="${pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ")}"/>`;
      }
      pts.forEach(p => {
        svg += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.8" fill="${color}" stroke="var(--jb-bg)" stroke-width="1"/>`;
      });
    });

    const colW = pw / Math.max(hwRuns.length, 1);
    hwRuns.forEach((_, idx) => {
      svg += `<rect class="jb-hover-col" data-reg="${regIdx}" data-idx="${idx}" x="${Math.max(m.left, xOf(idx) - colW / 2)}" y="${m.top}" width="${colW}" height="${ph}" fill="transparent"/>`;
    });
    svg += `<line class="jb-ch" data-reg="${regIdx}" x1="0" y1="${m.top}" x2="0" y2="${m.top + ph}" stroke="var(--jb-muted)" stroke-dasharray="2,2" opacity="0"/>`;
    svg += `</svg>`;
    return { svg, xOf };
  }

  function renderLatestBarSvg(latest, state, cols) {
    const W = 720, H = 225;
    const m = { top: 18, right: 14, bottom: 32, left: 46 };
    const pw = W - m.left - m.right;
    const ph = H - m.top - m.bottom;
    const visCols = cols.filter(c => !state.hidden.has(c));
    const sl = getSlice(latest, state.dim, state.domain, state.force, state.mode);

    const vals = [];
    visCols.forEach(c => (sl[c] || []).forEach(v => { if (v != null && v > 0) vals.push(v); }));
    if (!vals.length) {
      return '<div class="jb-empty" style="padding:1.1rem">No latest-commit data for this selection</div>';
    }

    const minV = Math.min(...vals), maxV = Math.max(...vals);
    const useLog = state.scale === "log" && minV > 0;
    const yLo = useLog ? Math.pow(10, Math.floor(Math.log10(minV * 0.7))) : 0;
    const yHi = useLog ? Math.pow(10, Math.ceil(Math.log10(maxV * 1.22))) : maxV * 1.15;

    const yOf = v => {
      if (useLog) {
        const t = (Math.log10(Math.max(v, yLo)) - Math.log10(yLo)) / Math.max(Math.log10(yHi) - Math.log10(yLo), 1e-6);
        return m.top + ph * (1 - t);
      }
      return m.top + ph * (1 - (v - yLo) / Math.max(yHi - yLo, 1e-6));
    };

    let svg = `<svg class="jb-chart-svg" viewBox="0 0 ${W} ${H}">`;
    niceTicks(yLo, yHi, useLog).forEach(t => {
      const y = yOf(t);
      if (y >= m.top - 2 && y <= m.top + ph + 2) {
        svg += `<line x1="${m.left}" y1="${y}" x2="${W - m.right}" y2="${y}" stroke="var(--jb-border)" stroke-dasharray="3,3"/>`;
        svg += `<text x="${m.left - 7}" y="${y + 3}" text-anchor="end" font-size="9.5" fill="var(--jb-muted)">${t >= 10 ? t.toFixed(0) : t.toFixed(1)}</text>`;
      }
    });

    const grpW = pw / REGIME_ORDER.length;
    const nBars = Math.max(visCols.length, 1);
    const gap = 2;
    const barW = Math.min(20, (grpW - 20) / nBars - gap);

    REGIME_ORDER.forEach((reg, ri) => {
      const gx = m.left + ri * grpW + (grpW - nBars * (barW + gap)) / 2;
      visCols.forEach((col, ci) => {
        const v = sl[col]?.[ri];
        if (v == null) return;
        const x = gx + ci * (barW + gap);
        const y = yOf(v);
        const h = Math.max(2, m.top + ph - y);
        svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="2.5" fill="${COLORS[col] || '#2563eb'}" opacity="0.9"><title>${col} (${REGIMES[reg].label}): ${v.toFixed(2)} M pstep/s</title></rect>`;
        if (barW >= 15) {
          svg += `<text x="${(x + barW / 2).toFixed(1)}" y="${(y - 3).toFixed(1)}" text-anchor="middle" font-size="7.5" font-weight="600" fill="var(--jb-text)">${v >= 10 ? v.toFixed(0) : v.toFixed(1)}</text>`;
        }
      });
      svg += `<text x="${(m.left + (ri + 0.5) * grpW).toFixed(1)}" y="${H - 10}" text-anchor="middle" font-size="10.5" font-weight="600" fill="var(--jb-text)">${REGIMES[reg].label}</text>`;
    });
    svg += `</svg>`;
    return svg;
  }

  function mountHardwareSection(container, group) {
    const hwRuns = group.runs;
    const latestInit = hwRuns[hwRuns.length - 1];

    // Pick first available dim/domain/force/mode present in the latest snapshot.
    const availDims = DIMS.filter(d => latestInit.data && latestInit.data[d]);
    const initDim = availDims[0] || DIMS[0];
    const availDoms = DOMAINS.filter(d => latestInit.data?.[initDim]?.[d]);
    const initDom = availDoms[0] || DOMAINS[0];

    const state = {
      dim: initDim,
      domain: initDom,
      force: FORCES[0],
      mode: MODES[0],
      scale: "linear",
      hidden: new Set()
    };

    function render() {
      const latest = hwRuns[hwRuns.length - 1];
      const prev = hwRuns.length >= 2 ? hwRuns[hwRuns.length - 2] : null;
      const slLatest = getSlice(latest, state.dim, state.domain, state.force, state.mode);
      const slPrev = prev ? getSlice(prev, state.dim, state.domain, state.force, state.mode) : {};
      const cols = COLLIDERS.filter(c => slLatest[c] && slLatest[c].some(v => v != null));

      const allVals = [];
      cols.forEach(c => (slLatest[c] || []).forEach(v => { if (v != null) allVals.push(v); }));
      const peakMps = allVals.length ? Math.max(...allVals) : 0;

      let gmText = "Baseline";
      let gmSub = "Need ≥2 commits";
      if (prev && cols.length) {
        const ratios = [];
        cols.forEach(c => REGIME_ORDER.forEach((_, ri) => {
          const a = slPrev[c]?.[ri], b = slLatest[c]?.[ri];
          if (a > 0 && b > 0) ratios.push(b / a);
        }));
        if (ratios.length) {
          const gm = Math.exp(ratios.reduce((acc, x) => acc + Math.log(x), 0) / ratios.length);
          const pct = (gm - 1) * 100;
          gmText = `${gm.toFixed(2)}× (${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%)`;
          gmSub = `vs ${prev.commit}`;
        }
      }

      const xOfByReg = {};
      const gridCards = REGIME_ORDER.map((reg, ri) => {
        const res = renderRegimeProgressionSvg(hwRuns, state, cols, ri);
        xOfByReg[ri] = res.xOf;
        const phiVal = REGIMES[reg].phi[state.dim];
        const velVal = REGIMES[reg].vel;
        return `
          <div class="jb-card">
            <div class="jb-card-head">
              <div class="jb-card-title">${REGIMES[reg].label}</div>
              <div class="jb-card-sub">v=${velVal}, φ=${phiVal} · M pstep/s</div>
            </div>
            <div class="jb-chart-wrap" data-wrap-reg="${ri}">
              ${res.svg}
              <div class="jb-tooltip" data-tip-reg="${ri}"></div>
            </div>
          </div>`;
      }).join("");

      const latestBar = renderLatestBarSvg(latest, state, cols);

      const nSpatial = state.mode === "batch" ? `${latest.batch || 8}×${((latest.n || 65536) / 2).toLocaleString()}` : (latest.n || 65536).toLocaleString();
      const nNaive = state.mode === "batch" ? `${latest.batch || 8}×${((latest.n_naive || 5000) / 2).toLocaleString()}` : (latest.n_naive || 5000).toLocaleString();

      const rowsHtml = cols.map(c => {
        const isNaive = c === "naive" || c === "nl_naive";
        const vals = slLatest[c] || [];
        const cells = REGIME_ORDER.map((_, ri) => {
          const v = vals[ri];
          return `<td class="jb-num">${v != null ? `<strong>${v.toFixed(1)}</strong>` : "—"}</td>`;
        }).join("");

        let delta = { text: "baseline", cls: "jb-badge-flat" };
        if (prev) {
          const ratios = REGIME_ORDER.map((_, ri) => {
            const a = slPrev[c]?.[ri], b = vals[ri];
            return (a > 0 && b > 0) ? b / a : null;
          }).filter(x => x != null);
          if (ratios.length) {
            const gm = Math.exp(ratios.reduce((a, x) => a + Math.log(x), 0) / ratios.length);
            delta = fmtDelta(gm, 1.0);
          }
        }
        return `<tr>
          <td><div class="jb-col-name"><span class="jb-swatch" style="background:${COLORS[c] || '#2563eb'}"></span><span>${c}</span></div></td>
          <td class="jb-num">${isNaive ? nNaive : nSpatial}</td>
          ${cells}
          <td class="jb-num"><span class="jb-badge ${delta.cls}">${delta.text}</span></td>
        </tr>`;
      }).join("");

      const warnHtml = latest.throttled
        ? `<div class="jb-warn">⚠ Latest snapshot was recorded while system throttle policy was not high performance.</div>`
        : "";

      container.innerHTML = `
        <div class="jb-hw-head">
          <div class="jb-hw-title">${group.hw}</div>
          <div class="jb-hw-meta">${group.profile} · ${group.dtype} · steps=${latest.steps || 2000} · only_spheres=${latest.only_spheres ? "True" : "False"} · ${hwRuns.length} commit(s)</div>
        </div>
        ${warnHtml}

        <div class="jb-controls">
          <div class="jb-ctl-block">
            <span class="jb-ctl-label">Dim</span>
            ${DIMS.map(d => `<button class="jb-pill ${state.dim === d ? "active" : ""}" data-dim="${d}">${d.toUpperCase()}</button>`).join("")}
          </div>
          <div class="jb-ctl-block">
            <span class="jb-ctl-label">Boundary</span>
            ${DOMAINS.map(d => `<button class="jb-pill ${state.domain === d ? "active" : ""}" data-domain="${d}">${d}</button>`).join("")}
          </div>
          <div class="jb-ctl-block">
            <span class="jb-ctl-label">Force</span>
            ${FORCES.map(f => `<button class="jb-pill ${state.force === f ? "active" : ""}" data-force="${f}">${f}</button>`).join("")}
          </div>
          <div class="jb-ctl-block">
            <span class="jb-ctl-label">Mode</span>
            ${MODES.map(m => `<button class="jb-pill ${state.mode === m ? "active" : ""}" data-mode="${m}">${m === "batch" ? `batch (B=${latest.batch || 8})` : "single"}</button>`).join("")}
          </div>
          <div class="jb-ctl-block">
            <button class="jb-pill ${state.scale === "log" ? "active" : ""}" data-scale="1">${state.scale === "log" ? "Log Y" : "Linear Y"}</button>
          </div>
        </div>

        <div class="jb-kpis">
          <div class="jb-kpi">
            <div class="jb-kpi-label">Latest Commit</div>
            <div class="jb-kpi-value"><a href="https://github.com/cdelv/JaxDEM/commit/${latest.commit}" target="_blank" rel="noopener">${latest.commit}</a></div>
            <div class="jb-kpi-sub">${latest.date} · JAX ${latest.jax || ""}</div>
          </div>
          <div class="jb-kpi">
            <div class="jb-kpi-label">Peak Throughput</div>
            <div class="jb-kpi-value">${peakMps.toFixed(1)} M/s</div>
            <div class="jb-kpi-sub">${state.dim.toUpperCase()} · ${state.domain} · ${state.force}</div>
          </div>
          <div class="jb-kpi">
            <div class="jb-kpi-label">Geomean vs Prev</div>
            <div class="jb-kpi-value">${gmText}</div>
            <div class="jb-kpi-sub">${gmSub}</div>
          </div>
          <div class="jb-kpi">
            <div class="jb-kpi-label">Particles (B×N)</div>
            <div class="jb-kpi-value">${nSpatial}</div>
            <div class="jb-kpi-sub">naive / nl_naive: ${nNaive}</div>
          </div>
        </div>

        <div class="jb-legend">
          ${cols.map(c => `<span class="jb-leg-item${state.hidden.has(c) ? " off" : ""}" data-toggle="${c}"><span class="jb-swatch" style="background:${COLORS[c] || '#2563eb'}"></span>${c}</span>`).join("")}
        </div>

        <div class="jb-grid-2x2">${gridCards}</div>

        <div class="jb-card jb-card-wide">
          <div class="jb-card-head">
            <div class="jb-card-title">Latest Commit (${latest.commit}) — Throughput (M particle-steps/s)</div>
            <div class="jb-card-sub">${state.dim.toUpperCase()} · ${state.domain} · ${state.force} · ${state.mode} · Higher is better</div>
          </div>
          ${latestBar}
        </div>

        <div class="jb-card">
          <div class="jb-card-head">
            <div class="jb-card-title">Latest Commit Matrix (${latest.commit})</div>
            <div class="jb-card-sub">Throughput in M particle-steps/s</div>
          </div>
          <div class="jb-table-wrap">
            <table class="jb-table">
              <thead>
                <tr>
                  <th>Collider</th>
                  <th class="jb-num">B×N</th>
                  ${REGIME_ORDER.map(r => `<th class="jb-num">${REGIMES[r].short}</th>`).join("")}
                  <th class="jb-num">Δ</th>
                </tr>
              </thead>
              <tbody>${rowsHtml || '<tr><td colspan="7" style="text-align:center;color:var(--jb-muted);padding:0.9rem">No measurements recorded for this combination yet.</td></tr>'}</tbody>
            </table>
          </div>
        </div>
      `;

      container.querySelectorAll("[data-dim]").forEach(b => b.addEventListener("click", () => { state.dim = b.dataset.dim; render(); }));
      container.querySelectorAll("[data-domain]").forEach(b => b.addEventListener("click", () => { state.domain = b.dataset.domain; render(); }));
      container.querySelectorAll("[data-force]").forEach(b => b.addEventListener("click", () => { state.force = b.dataset.force; render(); }));
      container.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", () => { state.mode = b.dataset.mode; render(); }));
      container.querySelector("[data-scale]").addEventListener("click", () => { state.scale = state.scale === "log" ? "linear" : "log"; render(); });
      container.querySelectorAll("[data-toggle]").forEach(el => el.addEventListener("click", () => {
        const c = el.dataset.toggle;
        if (state.hidden.has(c)) state.hidden.delete(c); else state.hidden.add(c);
        render();
      }));

      container.querySelectorAll(".jb-hover-col").forEach(colEl => {
        colEl.addEventListener("mouseenter", e => {
          const ri = Number(colEl.dataset.reg);
          const idx = Number(colEl.dataset.idx);
          const run = hwRuns[idx];
          const wrap = container.querySelector(`[data-wrap-reg="${ri}"]`);
          const tip = container.querySelector(`[data-tip-reg="${ri}"]`);
          const ch = container.querySelector(`.jb-ch[data-reg="${ri}"]`);
          const x = xOfByReg[ri](idx);
          if (ch) { ch.setAttribute("x1", x); ch.setAttribute("x2", x); ch.setAttribute("opacity", "0.75"); }
          const sl = getSlice(run, state.dim, state.domain, state.force, state.mode);
          const items = cols
            .filter(c => !state.hidden.has(c) && sl[c]?.[ri] != null)
            .map(c => `<div style="display:flex;justify-content:space-between;gap:8px"><span><span class="jb-swatch" style="display:inline-block;background:${COLORS[c]};margin-right:4px"></span>${c}</span><strong>${sl[c][ri].toFixed(1)} M/s</strong></div>`)
            .join("");
          tip.innerHTML = `<div style="font-weight:700;margin-bottom:3px">${run.commit} · ${run.date}</div>${items}`;
          tip.style.display = "block";
          const rect = wrap.getBoundingClientRect();
          tip.style.left = `${Math.min(Math.max(6, e.clientX - rect.left + 10), Math.max(6, rect.width - 190))}px`;
          tip.style.top = "8px";
        });
        colEl.addEventListener("mouseleave", () => {
          const ri = Number(colEl.dataset.reg);
          const tip = container.querySelector(`[data-tip-reg="${ri}"]`);
          const ch = container.querySelector(`.jb-ch[data-reg="${ri}"]`);
          if (tip) tip.style.display = "none";
          if (ch) ch.setAttribute("opacity", "0");
        });
      });
    }

    render();
  }

  root.innerHTML = "";
  Array.from(hwMap.values()).forEach(group => {
    const sec = document.createElement("section");
    sec.className = "jb-hw-section";
    root.appendChild(sec);
    mountHardwareSection(sec, group);
  });
})();
"""


def render_dashboard_html(
    history: list[dict[str, Any]] | None = None,
    *,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
) -> str:
    """Return an embeddable HTML fragment (for Sphinx or inline preview)."""
    records = load_history(history_path) if history is None else history
    data_json = json.dumps(
        {
            "dims": list(DIMS),
            "domains": list(DOMAINS),
            "colliders": list(COLLIDERS),
            "forces": list(FORCES),
            "modes": list(MODES),
            "regimes": REGIMES,
            "regimeOrder": list(REGIME_ORDER),
            "runs": records,
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return (
        f"<style>{_DASHBOARD_CSS}</style>\n"
        f'<div class="jdem-bench" id="jdem-bench-root"></div>\n'
        f'<script type="application/json" id="jdem-bench-data">{data_json}</script>\n'
        f"<script>{_DASHBOARD_JS}</script>\n"
    )


def render_standalone_html(
    history: list[dict[str, Any]] | None = None,
    *,
    history_path: Path | str = DEFAULT_HISTORY_PATH,
) -> str:
    """Return a complete standalone HTML page for local viewing."""
    body = render_dashboard_html(history, history_path=history_path)
    return (
        "<!doctype html>\n<html lang='en'>\n<head>\n"
        "<meta charset='utf-8'>\n<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        "<title>JaxDEM Benchmarks</title>\n"
        "<style>"
        "body{margin:0;padding:1.25rem;max-width:1020px;margin-inline:auto;"
        "background:#fff;color:#0f172a;font-family:system-ui,-apple-system,sans-serif;}"
        "@media(prefers-color-scheme:dark){body{background:#0b1120;color:#e2e8f0;}"
        ".jdem-bench{--jb-bg:#0b1120;--jb-surface:#111827;--jb-border:#1f2937;--jb-text:#e2e8f0;--jb-muted:#94a3b8;}}"
        "h1{font-size:1.45rem;margin:0 0 0.25rem;}"
        "</style>\n</head>\n<body>\n"
        "<h1>JaxDEM Benchmarks</h1>\n"
        f"{body}\n</body>\n</html>\n"
    )


def make_sphinx_directive() -> Any:
    """Create the ``.. jaxdem-benchmarks::`` Sphinx directive class lazily."""
    from docutils import nodes
    from docutils.parsers.rst import Directive

    class JaxDEMBenchmarksDirective(Directive):
        has_content = False

        def run(self) -> list[Any]:
            env = self.state.document.settings.env
            hist_path = Path(
                os.environ.get("JAXDEM_BENCHMARKS_HISTORY", str(DEFAULT_HISTORY_PATH))
            )
            if hist_path.exists():
                env.note_dependency(str(hist_path))
            html = render_dashboard_html(history_path=hist_path)
            return [nodes.raw("", html, format="html")]

    return JaxDEMBenchmarksDirective
