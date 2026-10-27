"""Figures of results_report.tex. Run from technicalReport/: python -I figures/make_results_figures.py
fig_gain_maps.pdf : passive per-cell gain g(|k|,|D|) of the three regimes (parsed from ../runs/dipolelet2/design.txt)
fig_endi_main.pdf : eNDI NRMSE at 25 fixed iterations, dirty / noise_clean / joint regime net / passive cascade (values of Table 'main')."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"          # categorical slots 1-3 of the reference palette + neutral baseline
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

# ------------------------------------------------------------------ gain maps
text = (HERE.parents[1] / "runs" / "dipolelet2" / "design.txt").read_text(encoding="utf8").splitlines()
KE = np.concatenate([[0.0], np.geomspace(0.012, 0.87, 23), [1.0]])
DE = np.array([0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.27, 0.333, 0.5, 0.667])
maps = {}
for name in ("real", "noise", "sim2"):
    i = next(n for n, l in enumerate(text) if l.startswith(f"[{name}] total E_b"))
    rows = text[i + 3: i + 3 + 24]
    maps[name] = np.array([[np.nan if t == "." else float(t) for t in r.split()[1:]] for r in rows])
titles = {"real": "real-model residual\n(STI/off-diagonal + noise)", "noise": "noise (RC1, SNR 60)", "sim2": r"real Sim2 $\psi$" + "\n(noise-like)"}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#efeeea")
for ax, name in zip(axes, ("real", "noise", "sim2")):
    im = ax.pcolormesh(np.arange(13), np.arange(25), np.ma.masked_invalid(maps[name]), cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=0.3)
    ax.set_title(titles[name], fontsize=8)
    ax.set_xticks(np.arange(0, 13, 2)); ax.set_xticklabels([f"{DE[k]:.2g}" for k in range(0, 13, 2)], rotation=0)
    ax.set_xlabel("$|D|$ (bin edge)")
axes[0].set_yticks(np.arange(0, 25, 4)); axes[0].set_yticklabels([f"{KE[k]:.2g}" for k in range(0, 25, 4)])
axes[0].set_ylabel(r"$|\mathbf{k}|$ (cycles/voxel, bin edge)")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02); cb.set_label("passive gain $g$"); cb.outline.set_visible(False)
fig.savefig(HERE / "fig_gain_maps.pdf"); plt.close(fig)

# ------------------------------------------------------------------ main eNDI bars
rows = ["exp7\n(real tissue)", "exp8\n(STI)", "exp9\n(off-diag.)", "Sim2\nreal", "Sim2 fwd\njumps", "exp1\n(noiseless)", "exp3\n(SNR 40)"]
data = {"dirty": [93.60, 67.69, 83.33, 53.94, 66.81, 35.76, 61.15],
        "noise_clean (B)": [71.69, 61.15, 76.33, 47.48, 48.43, 36.11, 43.02],
        "joint regime net": [56.58, 50.40, 53.60, 48.62, 50.58, 37.54, 43.98],
        "passive cascade (A then B)": [58.63, 52.35, 60.46, 48.72, 48.57, 36.11, 43.05]}
colors = [GRAY, BLUE, ORANGE, AQUA]
fig, ax = plt.subplots(figsize=(7.2, 3.1), constrained_layout=True)
w, x = 0.2, np.arange(len(rows))
for k, ((name, v), c) in enumerate(zip(data.items(), colors)):
    ax.bar(x + (k - 1.5) * w, v, w * 0.92, color=c, label=name, edgecolor="white", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(rows); ax.set_ylabel("eNDI NRMSE (%), 25 iterations")
ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_ylim(0, 100)
ax.legend(ncol=2, frameon=False, loc="upper right", fontsize=7.5)
fig.savefig(HERE / "fig_endi_main.pdf"); plt.close(fig)
print("figures written")
