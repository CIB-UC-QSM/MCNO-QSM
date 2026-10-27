"""Writes notebooks/generalization_paper_experiments.ipynb (run: python -I notebooks/build_generalization_notebook.py)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Does the cleaner work on the experiments of the Dipole-lets paper plots? (generalization check)

Same experiments as `utils/Dipole-lets_Paper_Plots_3.ipynb` (cell 6: `exp1` ... `exp11`, same positions, same seed 47), plus the Sim2 ground truth forward-simulated and corrupted with the notebook's complex-noise recipe. Every volume is shown with `imagesc3d2` and every error with `compute_rmse`, as in the original.

**Read this before the numbers**

* **RC1 experiments are NOT an independent test.** The nets were trained on this very head (patches from x in [0,100) of axis 0, plus rotated/rescaled copies) with random placements of the same artifact families. The only held-out part is the check slab x in [112,160), and even there the net saw the dipole field of the whole head in the original pose. `exp7` (in-vivo tissue phase), `exp8`, `exp9` use real residuals that only the `mix` net was trained on; `noise` never saw them.
* **Forward-simulated Sim2 is a test of the anatomy, not of the artifacts**: the head was never trained on, but its inputs contain only the synthetic noise (the real Sim2 `psi` is not in them).
* Two conventions for the field **outside the brain mask**: `clean` = noise-free `D*chi` there (what the nets saw in training) and `out0` = zeros (what `Frequency.nii` and a BET-masked in-vivo field look like). A large gap between them means the nets use the outside field, which does not exist in real data.
* Targets are always `mask * D*chi` (COSMOS for RC1, ground truth for Sim2). eNDI = `ndiCG_auto`, alpha 1e-5, linear, 150 iterations max, zero-padded to 192^3 (RC1) or 256^3 (Sim2).

Data: made on the server by `scripts/export_generalization.py` (the cleaning and the inversions need the GPU); copy `runs/generalization/` to the PC.
""")

code("""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({         # Global font size, as in the paper notebook
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

here = [Path.cwd(), *Path.cwd().parents]
def find(*rel):
    for base in here:
        for r in rel:
            if (base / r).exists():
                return base / r
    raise FileNotFoundError(f"none of {rel} found from {Path.cwd()} or its parents")
sys.path.insert(0, str(find("external/FANSI_tools.py", "utils/FANSI_tools.py", "FANSI_tools.py").parent))
from FANSI_tools import imagesc3d2, compute_rmse     # the user's own plotting / RMSE functions
GEN = find("runs/generalization")
print("data:", GEN)
""")

code("""
def load_set(tag):
    common = np.load(GEN / f"{tag}_common.npz")
    common = {k: common[k].astype(np.float32) if common[k].ndim == 3 else common[k] for k in common.files}
    meta = json.loads((GEN / f"{tag}_meta.json").read_text())
    return common, meta

def load_exp(tag, name):
    z = np.load(GEN / f"{tag}_{name}.npz")
    return {k: z[k].astype(np.float32) for k in z.files}

def view(vol, title, contrast, rot, pos=None):
    # same call as the paper notebook: three planes through the centre, grey, shared contrast
    pos = tuple(x // 2 for x in vol.shape) if pos is None else pos
    imagesc3d2(vol, pos, cmap='grey', contrast=contrast, fig_num=2, rot_deg=rot, scale_fig=[-0.12, 0.12, -0.12, 0.12],
               avg_size=0, title_fig=title)

ROT_RC1 = [90, 90, -90]      # cells 7 / 17 of the paper notebook (RC1 views)
ROT_SIM2 = [90, 90, 90]      # cells 11 / 13 (Sim2 views)
FIELD = [-0.03, 0.03]        # ppm, field images
ERR = [-0.015, 0.015]
CHI = [-0.1, 0.1]            # ppm, susceptibility maps (as the paper notebook)
CHI_ERR = [-0.05, 0.05]

SETS = {}
for tag in ("rc1", "sim2"):
    try:
        SETS[tag] = load_set(tag)
        print(tag, "experiments:", [k for k in SETS[tag][1] if k != "oracle"])
    except FileNotFoundError as e:
        print("missing", tag, e)
""")

code("""
def table(tag, region="all"):
    common, meta = SETS[tag]
    exps = [k for k in meta if k != "oracle"]
    methods = list(meta[exps[0]]["methods"])
    key = f"phase_nrmse_{region}"
    print(f"phase NRMSE (%) vs mask*D*chi, region = {region}")
    print(f"{'experiment':14s}" + "".join(f"{m:>12s}" for m in methods))
    for e in exps:
        print(f"{e:14s}" + "".join(f"{meta[e]['methods'][m].get(key, float('nan')):12.2f}" for m in methods))
    print()
    print("eNDI NRMSE (%) vs chi ground truth" + (f"   (oracle mask*D*chi: {meta['oracle']['nrmse']:.2f}%)" if "oracle" in meta else ""))
    print(f"{'experiment':14s}" + "".join(f"{m:>12s}" for m in methods))
    for e in exps:
        print(f"{e:14s}" + "".join(f"{meta[e]['methods'][m].get('endi_nrmse', float('nan')):12.2f}" for m in methods))
""")

md("""
## A. Paper experiments on RC1 (exp1 ... exp11)

Phase NRMSE: distance of each field to the target `mask * D*chi_cosmos` (dirty = the experiment itself, masked). Columns ending in `_out0` use zeros outside the mask. The `check` table covers only the held-out slab x in [112,160). Where the cleaner is useful, the number falls; for `exp1` / `exp10` (nothing to remove) any change is a harm.
""")
code("""
if "rc1" in SETS:
    table("rc1", "all"); print("=" * 90); table("rc1", "check")
""")

md("""
### A.1 Fields: dirty, cleaned, and what remains (RC1)

For each experiment: the dirty field (ppm), the cleaned field, and `cleaned - target`. Same grey scale and plane rotation as the paper notebook.
""")
code("""
EXPS_RC1 = ["exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9", "exp11"]   # edit to look at others
MAIN = "noise"                                                                          # which run to display
if "rc1" in SETS:
    common, meta = SETS["rc1"]
    mask, clean, chi_gt = common["mask"], common["clean"], common["chi_gt"]
    target = clean * mask
    for e in EXPS_RC1:
        d = load_exp("rc1", e)
        print(e, meta[e]["description"])
        view(d["field_dirty"], f"{e} dirty (masked)", FIELD, ROT_RC1)
        for m in (MAIN, MAIN + "_out0"):
            if f"field_{m}" in d:
                view(d[f"field_{m}"], f"{e} cleaned {m}: RMSE {compute_rmse(d[f'field_{m}'], target):.2f}%", FIELD, ROT_RC1)
        view((d[f"field_{MAIN}"] - target), f"{e} remaining error ({MAIN})", ERR, ROT_RC1)
""")

md("""
### A.2 Inversion with eNDI (RC1)

Susceptibility maps from the dirty field and from the cleaned fields, titled with the RMSE against the COSMOS susceptibility inside the mask (as `compute_rmse` in the paper notebook). Only the experiments and methods listed in the table above were inverted.
""")
code("""
if "rc1" in SETS:
    for e in EXPS_RC1:
        d = load_exp("rc1", e)
        for m in meta[e]["methods"]:
            if f"chi_{m}" in d:
                view(d[f"chi_{m}"] * mask, f"{e} eNDI {m}: RMSE {compute_rmse(d[f'chi_{m}'] * mask, chi_gt * mask):.2f}%", CHI, ROT_RC1)
""")

md("""
## B. Sim2 ground truth, forward-simulated

`chi` of Sim2 -> `D*chi` -> complex noise (the notebook recipe, the magnitude being the raw eNDI weight, whose scale matches the RC1 magnitude the nets were trained with) at SNR 100, 40, and at the SNR that reproduces the rms of the real Sim2 `psi` (0.0037 ppm); then two variants with phase jumps and with spheres in `chi` (as exp4 / exp11). The head was never trained on.
""")
code("""
if "sim2" in SETS:
    table("sim2", "all")
""")
code("""
EXPS_SIM2 = ["sim2_snr100", "sim2_snr40", "sim2_snrcal", "sim2_jumps", "sim2_spheres"]
if "sim2" in SETS:
    common2, meta2 = SETS["sim2"]
    mask2, clean2, chi2 = common2["mask"], common2["clean"], common2["chi_gt"]
    target2 = clean2 * mask2
    for e in EXPS_SIM2:
        if e not in meta2:
            continue
        d = load_exp("sim2", e)
        print(e, meta2[e]["description"])
        view(d["field_dirty"], f"{e} dirty (masked)", FIELD, ROT_SIM2)
        for m in (MAIN, MAIN + "_out0"):
            if f"field_{m}" in d:
                view(d[f"field_{m}"], f"{e} cleaned {m}: RMSE {compute_rmse(d[f'field_{m}'], target2):.2f}%", FIELD, ROT_SIM2)
        view((d[f"field_{MAIN}"] - target2), f"{e} remaining error ({MAIN})", ERR, ROT_SIM2)
""")
code("""
if "sim2" in SETS:
    for e in EXPS_SIM2:
        if e not in meta2:
            continue
        d = load_exp("sim2", e)
        for m in meta2[e]["methods"]:
            if f"chi_{m}" in d:
                view(d[f"chi_{m}"] * mask2, f"{e} eNDI {m}: RMSE {compute_rmse(d[f'chi_{m}'] * mask2, chi2 * mask2):.2f}%", CHI, ROT_SIM2)
""")

md("""
## B2. Cleaning judged at a FIXED number of eNDI iterations (no automatic stop)

eNDI is a measure of how well the cleaning worked, so every field gets the same `ndiCG` (alpha 1e-5, no auto-stop, 40 iterations; `scripts/rescore_fixed.py`) and the NRMSE against the susceptibility ground truth is read after 10, 25 and 40 iterations. A dirty field does not converge (its error rises after a few iterations); a good cleaned field keeps improving. `min` is the best iteration (needs the ground truth, so it is optimistic). The `oracle` is the clean masked field `mask*D*chi`.
""")
code("""
FIXED = json.loads((GEN / "fixed_meta.json").read_text())
def fixed_table(tag):
    rows = {}
    for key, h in FIXED.items():
        t, exp, method = key.split("/")
        if t == tag:
            rows.setdefault(exp, {})[method] = h
    g = lambda h, i: h["nrmse"][min(i, len(h["nrmse"]) - 1)]
    print(f"eNDI NRMSE (%) vs ground truth, fixed iterations, set {tag}")
    print(f"{'experiment':14s}{'method':8s}{'@10':>8s}{'@25':>8s}{'@40':>8s}{'min':>8s}{'@min':>6s}{'XSIM@25':>9s}")
    for exp, methods in rows.items():
        for m, h in methods.items():
            n = h["nrmse"]
            print(f"{exp:14s}{m:8s}{g(h, 10):8.2f}{g(h, 25):8.2f}{g(h, 40):8.2f}{min(n):8.2f}{n.index(min(n)):6d}{h['xsim'][min(25, len(h['xsim']) - 1)]:9.3f}")
        print()
    return rows

ROWS = {tag: fixed_table(tag) for tag in ("rc1", "sim2")}
""")
code("""
# NRMSE against iteration: dirty (grey), noise net, mix net, and the oracle of each set (dashed)
for tag, rows in ROWS.items():
    exps = [e for e in rows if e != "oracle"]
    cols = 4
    fig, axes = plt.subplots((len(exps) + cols - 1) // cols, cols, figsize=(4 * cols, 3 * ((len(exps) + cols - 1) // cols)), squeeze=False)
    for ax, e in zip(axes.ravel(), exps):
        for m, color in (("dirty", "0.5"), ("noise", "C0"), ("mix", "C1")):
            if m in rows[e]:
                ax.plot(rows[e][m]["nrmse"], color=color, label=m)
        ax.plot(rows["oracle"]["oracle"]["nrmse"], "k--", lw=0.8, label="oracle")
        ax.axvline(25, color="r", lw=0.5)
        ax.set_title(f"{tag} {e}"); ax.set_ylim(25, 110); ax.set_xlabel("eNDI iteration"); ax.set_ylabel("NRMSE %")
    axes.ravel()[0].legend(fontsize=7)
    plt.tight_layout(); plt.show()
""")

md("""
## C. Does the net use the field outside the mask?

For each experiment: phase NRMSE of the cleaned field with `clean` outside (training convention) against `out0` (zeros outside). If `out0` is clearly worse, the nets rely on information that real data do not have, and the Sim2 results (whose outside field is zero) are pessimistic for the trained recipe; retraining with zeros outside would be the fix.
""")
code("""
fig, axes = plt.subplots(1, len(SETS), figsize=(6.5 * len(SETS), 3.8), squeeze=False)
for ax, (tag, (cm, mt)) in zip(axes[0], SETS.items()):
    exps = [k for k in mt if k != "oracle"]
    runs = [m for m in mt[exps[0]]["methods"] if m != "dirty" and not m.endswith("_out0")]
    w = 0.8 / (1 + 2 * len(runs))
    x = np.arange(len(exps))
    ax.bar(x, [mt[e]["methods"]["dirty"]["phase_nrmse_all"] for e in exps], w, label="dirty", color="0.6")
    for i, r in enumerate(runs):
        ax.bar(x + (1 + 2 * i) * w, [mt[e]["methods"][r]["phase_nrmse_all"] for e in exps], w, label=f"{r} (outside = clean)")
        ax.bar(x + (2 + 2 * i) * w, [mt[e]["methods"][r + "_out0"]["phase_nrmse_all"] for e in exps], w, label=f"{r} (outside = 0)")
    ax.set_xticks(x + 0.4); ax.set_xticklabels(exps, rotation=45, ha="right")
    ax.set_ylabel("phase NRMSE vs target (%)"); ax.set_title(tag); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

md("""
## D. The streaking source: zoom on the strongest susceptibility source of Sim2

The most extreme susceptibility voxel inside the mask is taken as the centre (check it is the calcification you mean; otherwise set `POS` by hand). The three planes of every inversion are shown through that point.
""")
code("""
if "sim2" in SETS:
    inside = np.where(mask2 > 0, chi2, 0)
    POS = tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(inside)), inside.shape))   # or set by hand
    print("strongest |chi| in the mask at", POS, "value %.2f ppm" % inside[POS])
    e = "sim2_snrcal"
    d = load_exp("sim2", e)
    view(chi2 * mask2, "ground truth chi", CHI, ROT_SIM2, pos=POS)
    for m in meta2[e]["methods"]:
        if f"chi_{m}" in d:
            view(d[f"chi_{m}"] * mask2, f"{e} eNDI {m}: RMSE {compute_rmse(d[f'chi_{m}'] * mask2, chi2 * mask2):.2f}%", CHI, ROT_SIM2, pos=POS)
""")

md("""
## Reading guide

* RC1 numbers are in-sample for the head: they say whether the net does what it was trained for, not that it generalizes.
* The forward-simulated Sim2 says whether the *noise* denoiser transfers to an unseen head with a purely synthetic `psi`; it does not say anything about real artifacts (the real Sim2 result is in `s4b_results_visualization.ipynb`).
* Single seed per run, single volume per set.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "mcno-qsm (3.13.2)", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("generalization_paper_experiments.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf8")
print("wrote", out)
