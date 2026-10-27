"""Initial values of the regime filter (CLAUDE.md 9ar) and a check that the UNTRAINED regime map already separates the regimes.

g0 = 1 (noise regime: leave the field to the network, as in noise_clean). g1 = least-squares gain per segment of the 12-segment decomposition (Sim2
Dipole-let parameters, coarse split in 3 |D| bands), fitted ONLY on RC1's modelled residuals (STI, off-diagonal) with noise at SNR 100 added;
the measured tissue residual (exp7) is never used, it stays the held-out test. w, c: r = 0.5 where the cone excess of the coarsest detail scale is
0.22 (noise: 0.06-0.14, Sim2 real 0.11-0.18, real residuals 0.28-0.47 in the 9ar table), slope 6 on log R.
Prints a [model.regime] TOML block and the distribution of r (5 / 50 / 95 percentile inside the mask, 96^3 patches and the whole volume) for each set."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.cleaner import PhaseCleaner  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, image_components  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

DIP = {**SIM2_DIPOLELET, "coarse_bands": 3}
d = load_rc1("dataRC1")
mask, chi, vox = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
b = clean_field(chi, vox, mask)
vol = Volume.from_rc1(d, "cpu")
cfg = ArtifactConfig(snr_low=100.0, snr_high=100.0)
fields = {}
for fam in ("noise", "sti_noise", "offdiag_noise"):
    dirty, _ = make_dirty(fam, vol, np.random.default_rng(3), torch.Generator().manual_seed(3), cfg)
    fields[fam] = dirty.numpy() * mask
fields["tissue (held out, exp7)"] = mask * d["phs_tissue"]
psti, pq = sti_fields(d["chi_13"], d["chi_23"], d["chi_33"], vox)
fields["STI noise-free (exp8)"] = mask * psti
fields["off-diag noise-free (exp9)"] = mask * (b - pq)
tr = DipoleletTransform(b.shape, vox, **DIP)
names = tr.segment_names
comps = lambda f: [p.numpy() * mask for p in image_components(tr, torch.from_numpy(np.ascontiguousarray(f, np.float32)))]


def fit(f):
    cs = comps(f)
    A = np.stack([c[mask > 0] for c in cs], 1).astype(np.float64)
    return np.linalg.lstsq(A, b[mask > 0].astype(np.float64), rcond=None)[0]


g1 = np.mean([fit(fields["sti_noise"]), fit(fields["offdiag_noise"])], axis=0)
print("segments:", names)
print("g1 (mean of STI+noise and off-diag+noise fits):", np.round(g1, 3).tolist())
w = [0.0, 0.0, 6.0, 0.0]
c = float(-6.0 * np.log(0.22))
print("\n[model.regime]\nrho = 6.0\nfeed = true")
print(f"g0 = {[1.0] * len(names)}")
print(f"g1 = {np.round(g1, 4).tolist()}")
print(f"w = {w}\nc = {c:.4f}")

model = PhaseCleaner(vox, dipolelet=DIP, channels=2, blocks=1, dilations=(1,), checkpoint=False,
                     regime=dict(rho=6.0, g0=[1.0] * len(names), g1=g1.tolist(), w=w, c=c))
r2 = load_rc2("dataRC2", 7.0)
m2 = r2["mask"].astype(np.float32)
sets = {**{k: (v, mask, vox) for k, v in fields.items()}, "Sim2 real": (r2["field"] * m2, m2, r2["voxel_size"]),
        "Sim2 forward + noise": (forward_field(r2["chi"], m2, r2["voxel_size"]) + 0.0037 * np.random.default_rng(0).standard_normal(m2.shape).astype(np.float32) * m2, m2, r2["voxel_size"])}


def regime_map(f, m, v):
    x = torch.from_numpy(np.ascontiguousarray(f * m, np.float32))[None, None] / float(model.input_scale)
    transform, _, mass = model._geometry(x.shape[-3:], x.device, v)
    cs = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0] for s in transform.analysis(x)], 1)
    with torch.no_grad():
        return model.regime(cs, mass)[1][0, 0].numpy()


print(f"\n{'set':28s} r in the mask: p5 / p50 / p95 (whole volume)  |  96^3 patches: median of the patch mean r [min, max]")
rng = np.random.default_rng(0)
for k, (f, m, v) in sets.items():
    r = regime_map(f, m, v)
    q = np.percentile(r[m > 0], [5, 50, 95])
    means = []
    for _ in range(10):
        o = [int(rng.integers(0, s - 96 + 1)) for s in f.shape]
        sl = tuple(slice(i, i + 96) for i in o)
        if m[sl].mean() < 0.4:
            continue
        means.append(float(regime_map(f[sl], m[sl], v)[m[sl] > 0].mean()))
    print(f"{k:28s} {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}   |   {np.median(means):.2f} [{min(means):.2f}, {max(means):.2f}]", flush=True)
