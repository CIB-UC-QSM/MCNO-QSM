"""What did the regime filter learn? (CLAUDE.md 9ar) Prints g0, g1, w, c of runs/s4b_<run>/best.pt and the distribution of the regime map r on
RC1 noise / tissue (exp7) / STI (exp8) / off-diag (exp9) and on the real Sim2 field. Server or laptop (CPU is enough, ~5 min)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.cleaner import PhaseCleaner  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "regime"
state = torch.load(Path("runs") / f"s4b_{run}" / "best.pt", map_location="cpu", weights_only=False)
config = state["config"]
model = PhaseCleaner(state["voxel_size"], input_scale=state["input_scale"], dipolelet=config["dipolelet"], **config["model"]).eval()
model.load_state_dict(state["model"])
rf = model.regime
np.set_printoptions(precision=3, suppress=True, linewidth=160)
print(f"run {run}, best step {state['step']}")
print("segments", model.blocks[0].real.shape[0])
print("g0", rf.g0.detach().numpy()); print("g1", rf.g1.detach().numpy())
print("w ", rf.w.detach().numpy(), " c", float(rf.c))
init = config["model"]["regime"]
print("g1 - init", rf.g1.detach().numpy() - np.array(init["g1"])); print("g0 - init", rf.g0.detach().numpy() - np.array(init["g0"]))

d = load_rc1("dataRC1")
mask, vox = d["mask"].astype(np.float32), d["voxel_size"]
b = clean_field(d["chi"], vox, mask)
psti, pq = sti_fields(d["chi_13"], d["chi_23"], d["chi_33"], vox)
noisy, _ = make_dirty("noise", Volume.from_rc1(d, "cpu"), np.random.default_rng(3), torch.Generator().manual_seed(3), ArtifactConfig(snr_low=100.0, snr_high=100.0))
r2 = load_rc2("dataRC2", 7.0)
m2 = r2["mask"].astype(np.float32)
sets = {"noise SNR100": (noisy.numpy() * mask, mask, vox), "tissue exp7": (mask * d["phs_tissue"], mask, vox), "STI exp8": (mask * psti, mask, vox),
        "off-diag exp9": (mask * (b - pq), mask, vox), "Sim2 real": (r2["field"] * m2, m2, r2["voxel_size"])}
for k, (f, m, v) in sets.items():
    x = torch.from_numpy(np.ascontiguousarray(f * m, np.float32))[None, None] / float(model.input_scale)
    transform, _, mass = model._geometry(x.shape[-3:], x.device, v)
    with torch.no_grad():
        comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0] for s in transform.analysis(x)], 1)
        r = rf(comps, mass)[1][0, 0].numpy()
    q = np.percentile(r[m > 0], [5, 50, 95])
    print(f"{k:16s} r p5 / p50 / p95 = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}  mean {r[m > 0].mean():.2f}", flush=True)
