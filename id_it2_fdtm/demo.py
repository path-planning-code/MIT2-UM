"""
Demo: ID-IT2-FDTM — Visualization and Walkthrough
===================================================

Demonstrates the core capabilities of the Iteration-Driven Interval Type-2
Fuzzy Dynamic Terrain Model:

  1. Nominal terrain generation (Gaussian peaks)
  2. Slope vulnerability quantification  (Eq. 3-5)
  3. Interval type-2 fuzzy elevation bounds h_L, h_U  (Eqs. 3-3, 3-4)
  4. Three-phase terrain evolution driven by zero-order T-S fuzzy rules
  5. Robust safety constraint enforcement  (Eq. 3-8)

Output
------
- ``figure_model_overview.png`` — 2x3 panel characterizing the model
- ``figure_evolution.png`` — cross-section evolution across all three phases
- Console output with phase-by-phase statistics

Run
---
    python demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; change to "TkAgg" for interactive
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from id_it2_fdtm import (
    SimpleTerrain, FuzzyDynamicTerrain,
    MAP_RANGE, SAFE_ALTITUDE, ALPHA, T1, T2, T_UPDATE,
)

# ============================================================
# Helpers
# ============================================================

def _cross_section(fdtm, y_fixed=50.0, n_pts=200):
    """Extract cross-section at y = y_fixed."""
    xs = np.linspace(0, MAP_RANGE[0], n_pts)
    h0 = np.array([fdtm.get_h0(x, y_fixed) for x in xs])
    hL = np.array([fdtm.get_hL(x, y_fixed) for x in xs])
    hU = np.array([fdtm.get_hU(x, y_fixed) for x in xs])
    return xs, h0, hL, hU

def _phase_label(iter_count):
    if iter_count < T1:
        return "Early"
    elif iter_count < T2:
        return "Middle"
    else:
        return "Late"

# ============================================================
# Main demo
# ============================================================

def main():
    print("=" * 64)
    print("  ID-IT2-FDTM — Full Demo & Visualization")
    print("=" * 64)
    print()

    # ---- 1. Create terrain ----
    print("[1/5] Generating synthetic terrain (8 peaks)...")
    terrain = SimpleTerrain(num_peaks=8, seed=2024)
    print(terrain.get_peaks_summary())

    # ---- 2. Initialize ID-IT2-FDTM ----
    print("\n[2/5] Initializing ID-IT2-FDTM...")
    fdtm = FuzzyDynamicTerrain(
        terrain, alpha=ALPHA, T1=T1, T2=T2,
        T_update=T_UPDATE, grid_size=100,
    )
    s = fdtm.get_summary()
    print(f"      alpha = {s['alpha']:.2f}")
    print(f"      h0    = [{s['h0_range'][0]:.1f}, {s['h0_range'][1]:.1f}] m")
    print(f"      h_U   = [{s['hU_range'][0]:.1f}, {s['hU_range'][1]:.1f}] m")
    print(f"      vuln  = [{s['vuln_range'][0]:.3f}, {s['vuln_range'][1]:.3f}]")

    # ---- 3. Figure 1: Model Characterization (2x3) ----
    print("\n[3/5] Rendering model characterization figure...")

    X, Y, h0 = fdtm.get_base_surface()
    _, _, vuln = fdtm.get_vulnerability_map()
    _, _, grad = fdtm.get_gradient_map()
    xs_prof, h0_prof, hL_prof, hU_prof = _cross_section(fdtm, y_fixed=50.0)
    _, _, hL_full, hU_full = fdtm.get_fuzzy_bounds()

    fig1, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    # --- (a) Nominal DEM 3D surface ---
    ax = axes[0, 0]
    ax = fig1.add_subplot(2, 3, 1, projection="3d")
    surf = ax.plot_surface(
        X, Y, h0, cmap=cm.terrain, linewidth=0, antialiased=True, alpha=0.92
    )
    ax.set_title("(a) Nominal DEM  $h_0(x,y)$  [Eq. 3-1]", fontsize=11, pad=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=30, azim=-60)
    fig1.colorbar(surf, ax=ax, shrink=0.5, label="Elevation (m)")

    # --- (b) Slope magnitude ---
    ax = axes[0, 1]
    im = ax.pcolormesh(X, Y, grad, cmap="YlOrRd", shading="auto")
    ax.set_title("(b) Slope Magnitude  $\\|\\nabla h_0\\|$", fontsize=11)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig1.colorbar(im, ax=ax, shrink=0.82, label="Slope (m/m)")

    # --- (c) Vulnerability map ---
    ax = axes[0, 2]
    im = ax.pcolormesh(X, Y, vuln, cmap="RdYlGn_r", shading="auto", vmin=0, vmax=1)
    ax.contour(X, Y, vuln, levels=[0.3, 0.7], colors="k", linewidths=1.0,
               linestyles="dashed")
    ax.set_title(
        "(c) Vulnerability  $v(x,y) = \\tanh(\\|\\nabla h_0\\| / s_{\\max})$  [Eq. 3-5]",
        fontsize=11,
    )
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    cbar = fig1.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("v  (0 = stable,  1 = fragile)")

    # --- (d) IT2 fuzzy bounds (cross-section) ---
    ax = axes[1, 0]
    ax.fill_between(xs_prof, hL_prof, hU_prof, alpha=0.25, color="C0",
                    label="IT2 fuzzy envelope  $[h_L, h_U]$")
    ax.plot(xs_prof, h0_prof, "k-", linewidth=1.5, label="Nominal $h_0$")
    ax.plot(xs_prof, hL_prof, "C0--", linewidth=1.0, label="$h_L$ (lower bound)")
    ax.plot(xs_prof, hU_prof, "C0--", linewidth=1.0, label="$h_U$ (upper bound)")
    ax.set_title(
        "(d) IT2 Fuzzy Bounds  [Eqs. 3-3, 3-4]  —  Cross-section at $y=50$ m",
        fontsize=11,
    )
    ax.set_xlabel("x (m)"); ax.set_ylabel("Elevation (m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, MAP_RANGE[0])

    # --- (e) Fuzzy rule table visualization ---
    ax = axes[1, 1]
    rule_table = fdtm._RULE_TABLE * 100  # convert to percentage
    phases = ["Early\n(iter < 30)", "Middle\n(30–60)", "Late\n(iter ≥ 60)"]
    vuln_labels = ["Low v\n(v < 0.3)", "Mid v\n(0.3–0.7)", "High v\n(v > 0.7)"]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    x_pos = np.arange(len(phases))
    width = 0.25
    for j, (label, color) in enumerate(zip(vuln_labels, colors)):
        bars = ax.bar(
            x_pos + (j - 1) * width, rule_table[:, j], width,
            label=label, color=color, edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, rule_table[:, j]):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                )

    ax.set_title("(e) T–S Fuzzy Rule Table  [Table 3-1]", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(phases, fontsize=9)
    ax.set_ylabel("Reduction Rate  $r(iter, v)$  (%)", fontsize=10)
    ax.legend(fontsize=8, title="Vulnerability Class", title_fontsize=9)
    ax.set_ylim(0, 8)
    ax.grid(axis="y", alpha=0.3)

    # --- (f) 3D IT2 bounds surface ---
    ax = fig1.add_subplot(2, 3, 6, projection="3d")
    stride = 3
    ax.plot_surface(
        X[::stride, ::stride], Y[::stride, ::stride], hL_full[::stride, ::stride],
        color="C0", alpha=0.35, linewidth=0, antialiased=True,
    )
    ax.plot_surface(
        X[::stride, ::stride], Y[::stride, ::stride], hU_full[::stride, ::stride],
        color="C3", alpha=0.35, linewidth=0, antialiased=True,
    )
    ax.plot_surface(
        X[::stride, ::stride], Y[::stride, ::stride], h0[::stride, ::stride],
        cmap=cm.terrain, alpha=0.8, linewidth=0, antialiased=True,
    )
    ax.set_title("(f) IT2 Fuzzy Elevation Bounds  [Eqs. 3-2–3-4]", fontsize=11, pad=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=30, azim=-60)

    plt.tight_layout(pad=2.0)
    fig1.savefig("figure_model_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print("      Saved: figure_model_overview.png")

    # ---- 4. Evolve and capture snapshots ----
    print("\n[4/5] Evolving terrain through three phases...")
    snapshots = {}  # iter -> cross-section data

    # Record initial state
    snapshots[0] = _cross_section(fdtm, y_fixed=50.0)
    print(f"      iter={fdtm.iter_count:3d}  phase={_phase_label(fdtm.iter_count):6s}")

    # Evolve to late phase
    total_iters = T2 + 40  # go well into late phase
    for it in range(1, total_iters + 1):
        fdtm.evolve(it)
        if it in [T1, T2, total_iters]:
            snapshots[it] = _cross_section(fdtm, y_fixed=50.0)
            s = fdtm.get_summary()
            print(
                f"      iter={it:3d}  phase={_phase_label(it):6s}  "
                f"hU=[{s['hU_range'][0]:.1f}, {s['hU_range'][1]:.1f}] m"
            )

    # ---- 5. Figure 2: Evolution ----
    print("\n[5/5] Rendering evolution comparison figure...")

    fig2, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    labels = [
        (0, "Initial  (iter = 0)", "Early Phase"),
        (T1, f"After Early Phase  (iter = {T1})", "Middle Phase"),
        (total_iters, f"After Full Evolution  (iter = {total_iters})", "Late Phase"),
    ]

    for ax, (it, title, phase_name) in zip(axes, labels):
        xs_p, h0_p, hL_p, hU_p = snapshots[it]
        ax.fill_between(xs_p, hL_p, hU_p, alpha=0.25, color="C0")
        ax.plot(xs_p, h0_p, "k-", linewidth=1.8, label="$h_0$ (nominal)")
        ax.plot(xs_p, hL_p, "C0--", linewidth=1.0, label="$h_L$")
        ax.plot(xs_p, hU_p, "C0--", linewidth=1.0, label="$h_U$")

        # Mark reduction
        if it > 0:
            # Annotate max reduction at a high-vulnerability point
            idx_max = np.argmax(np.abs(hU_p - snapshots[0][3]))
            delta = snapshots[0][3][idx_max] - hU_p[idx_max]
            if delta > 0.5:
                ax.annotate(
                    f"$\\Delta h_U$ = {delta:.1f} m",
                    xy=(xs_p[idx_max], hU_p[idx_max]),
                    xytext=(xs_p[idx_max] + 8, hU_p[idx_max] + 5),
                    arrowprops=dict(arrowstyle="->", color="C3", lw=1.2),
                    fontsize=9, color="C3",
                )

        ax.set_title(f"{title}\n({phase_name})", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_xlim(0, MAP_RANGE[0])
        if ax == axes[0]:
            ax.set_ylabel("Elevation (m)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    fig2.suptitle(
        "Terrain Evolution Under Post-Disaster Conditions  —  Cross-section at $y=50$ m",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    fig2.savefig("figure_evolution.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print("      Saved: figure_evolution.png")

    # ---- Summary ----
    s = fdtm.get_summary()
    print("\n" + "=" * 64)
    print("  DEMO COMPLETE")
    print("=" * 64)
    print(f"  Final state:  iter={s['iter_count']},  phase={s['phase']}")
    print(f"  h_U range:    [{s['hU_range'][0]:.1f}, {s['hU_range'][1]:.1f}] m")
    print(f"  Max h_U drop:  {snapshots[0][3].max() - snapshots[total_iters][3].max():.1f} m")
    print()
    print("  Output files:")
    print("    figure_model_overview.png  — model characterization (6 panels)")
    print("    figure_evolution.png       — cross-section evolution (3 phases)")
    print()

if __name__ == "__main__":
    main()
