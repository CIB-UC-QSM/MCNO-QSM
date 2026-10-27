"""Writes notebooks/regime_before_after.ipynb (run: python -I notebooks/build_regime_views_notebook.py)."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Regime run: phases before and after cleaning, and the reconstructions

Rows of the results table (eNDI NRMSE at 25 fixed iterations) shown as images with the plotting function of `FANSI_tools` (`imagesc3d2`) and `compute_rmse` inside the mask.
For each experiment: **(A)** the field (phase in ppm) dirty / cleaned by `noise_clean` / cleaned by `regime`, the clean target `mask*D*chi` and the errors; **(B)** the susceptibility map after 25 eNDI iterations of each, the ground truth and the oracle.

**Read this first**
* `exp7` is the real in-vivo tissue residual (the only held-out real psi); `exp8` / `exp9` are STI / off-diagonal residuals the regime net was TRAINED on (modelled with noise on top); `exp1` is a noiseless field (anything removed is damage); `sim2_real` is the real Sim2 (calcification at the centre of the views); `sim2_snr40` is the forward-simulated Sim2 at SNR 40.
* RC1 experiments are in-sample for the head. Single seed per run. Fixed 25 iterations, no automatic stop.
* Data: `runs/regime_views.npz`, made on the server by `scripts/export_regime_views.py` (cropped to the brain box; contrast scales below are in ppm).
""")

code("""
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display

plt.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8})

here = [Path.cwd(), *Path.cwd().parents]
def find(*rel):
    for base in here:
        for r in rel:
            if (base / r).exists():
                return base / r
    raise FileNotFoundError(f"none of {rel} found from {Path.cwd()} or its parents")

sys.path.insert(0, str(find("external/FANSI_tools.py", "utils/FANSI_tools.py", "FANSI_tools.py").parent))
from FANSI_tools import imagesc3d2, compute_rmse          # the user's own plotting / RMSE functions

def show():
    # display every open figure as a PNG (works with any backend), then close them
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor(), bbox_inches="tight")
        display(Image(data=buf.getvalue()))
    plt.close("all")

Z = np.load(find("runs/regime_views.npz"))
print("keys:", len(Z.files), "| backend:", matplotlib.get_backend())
get = lambda k: Z[k].astype(np.float32)

ROT = {"rc1": [90, 90, -90], "sim2": [90, 90, 90]}      # as in the paper notebook (RC1 / Sim2 views)
FIELD, ERR = [-0.03, 0.03], [-0.015, 0.015]             # ppm
CHI, CHI_ERR = [-0.1, 0.1], [-0.05, 0.05]               # ppm
SHOW = ["dirty", "noise_clean", "regime"]               # add "realmix_plain" to also see the control
EXPS = ["exp7", "exp8", "exp9", "exp1", "exp3", "sim2_real", "sim2_snr40"]
POS = {}                                                # optional centre per experiment, e.g. POS["exp7"] = (80, 80, 80) (indices in the cropped box)
tag_of = lambda e: "sim2" if e.startswith("sim2") else "rc1"

def view(vol, title, contrast, rot, pos=None):
    pos = tuple(n // 2 for n in vol.shape) if pos is None else pos
    imagesc3d2(vol, pos, cmap="grey", contrast=contrast, fig_num=2, rot_deg=rot, scale_fig=[-0.12, 0.12, -0.12, 0.12],
               avg_size=0, title_fig=title)
    show()

# Sim2: centre the views on the strongest susceptibility source (the calcification)
g2 = get("sim2__chi_gt") * get("sim2__mask")
POS["sim2_real"] = POS["sim2_snr40"] = tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(g2)), g2.shape))
print("Sim2 views centred on the strongest |chi| (calcification):", POS["sim2_real"])
""")

md("""
## Summary: RMSE (%) inside the mask

Phase = field vs the clean target `mask*D*chi`; eNDI = susceptibility map (25 iterations) vs the ground truth chi. `compute_rmse` of `FANSI_tools`, so the eNDI column is the one of the headline table (up to the float16 storage).
""")

code("""
METHODS = [m for m in ["dirty", "noise_clean", "realmix_plain", "regime"] if f"exp7__field_{m}" in Z.files]
print(f"{'experiment':12s}{'':8s}" + "".join(f"{m:>15s}" for m in METHODS))
for e in EXPS:
    if f"{e}__field_dirty" not in Z.files:
        continue
    t = tag_of(e)
    mk = get(f"{t}__mask") > 0
    tgt, gt = get(f"{t}__clean"), get(f"{t}__chi_gt")
    ph = [compute_rmse(get(f"{e}__field_{m}")[mk], tgt[mk]) if f"{e}__field_{m}" in Z.files else np.nan for m in METHODS]
    ch = [compute_rmse(get(f"{e}__chi_{m}")[mk], gt[mk]) if f"{e}__chi_{m}" in Z.files else np.nan for m in METHODS]
    print(f"{e:12s}{'phase':8s}" + "".join(f"{v:15.2f}" for v in ph))
    print(f"{'':12s}{'eNDI':8s}" + "".join(f"{v:15.2f}" for v in ch))
print("oracle chi:", {t: round(float(compute_rmse(get(f'{t}__chi_oracle')[get(f'{t}__mask') > 0], get(f'{t}__chi_gt')[get(f'{t}__mask') > 0])), 2) for t in ("rc1", "sim2")})
""")

md("""
## A. Phases (fields in ppm): before and after

For each experiment: the clean target, the dirty field, the cleaned fields, then the remaining error of the dirty field and of each cleaned field (same grey scale for all error images).
""")

code("""
def fields(e):
    t = tag_of(e)
    mk = get(f"{t}__mask") > 0
    tgt, rot, pos = get(f"{t}__clean"), ROT[t], POS.get(e)
    print("=" * 100, f"\\n{e}\\n", "=" * 100)
    view(tgt, f"{e}: clean target mask*D*chi", FIELD, rot, pos)
    for m in SHOW:
        f = get(f"{e}__field_{m}")
        view(f, f"{e} {m}: field, RMSE vs target {compute_rmse(f[mk], tgt[mk]):.1f}%", FIELD, rot, pos)
    for m in SHOW:
        f = get(f"{e}__field_{m}")
        view((f - tgt) * mk, f"{e} {m}: error (field - target)", ERR, rot, pos)

for e in EXPS[:3]:
    fields(e)
""")

md("""
## B. Reconstructions (eNDI, 25 fixed iterations)

Susceptibility maps in ppm, ground truth and oracle first (`oracle` = eNDI of the clean masked field: what the inversion gives with a perfect cleaner), then each method and its error against the ground truth.
""")

code("""
def recon(e):
    t = tag_of(e)
    mk = get(f"{t}__mask") > 0
    gt, rot, pos = get(f"{t}__chi_gt") * mk, ROT[t], POS.get(e)
    print("=" * 100, f"\\n{e}\\n", "=" * 100)
    view(gt, f"{e}: ground truth chi", CHI, rot, pos)
    o = get(f"{t}__chi_oracle")
    view(o, f"{e}: oracle (clean masked field), RMSE {compute_rmse(o[mk], gt[mk]):.1f}%", CHI, rot, pos)
    for m in SHOW:
        x = get(f"{e}__chi_{m}")
        view(x, f"{e} {m}: eNDI 25 it, RMSE {compute_rmse(x[mk], gt[mk]):.1f}%", CHI, rot, pos)
    for m in SHOW:
        x = get(f"{e}__chi_{m}")
        view((x - gt) * mk, f"{e} {m}: chi error", CHI_ERR, rot, pos)

for e in EXPS[:3]:
    recon(e)
""")

md("""
## C. The other rows of the table

`exp1` (noiseless: any change is damage), `exp3` (SNR 40), the real Sim2 with the calcification at the centre, and the forward-simulated Sim2 at SNR 40, where the regime net is 3 points worse than `noise_clean`. Fields first, then reconstructions.
""")

code("""
for e in EXPS[3:]:
    if f"{e}__field_dirty" in Z.files:
        fields(e)
        recon(e)
""")

md("""
## Reading guide
* A cleaning is good when the error image is smaller and the eNDI map is closer to the ground truth AFTER 25 fixed iterations; the dirty reconstruction of exp7-9 gets worse with more iterations, which is why the iteration count is fixed.
* Look at `exp1` for damage (removing signal from a clean field) and at `sim2_real` / `sim2_snr40` for the cost of the regime run on noise-like data.
* Edit `SHOW`, `EXPS`, `POS` in the second cell to change the methods, experiments or the centre of the planes.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name("regime_before_after.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf8")
print("wrote", out)
