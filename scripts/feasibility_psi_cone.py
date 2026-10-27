"""Upper-bound feasibility (CLAUDE.md 9bc): is the NON-cone part of the residual psi predictable from its cone part? (RC1 only, no eNDI)

The cone part of the measured field is observable (the clean field has almost no energy there, 0.2-0.9 %), and a notch deletes it. What a notch leaves is the
rest of psi. A learned extractor (cone-masked Noise2Self, a gain net, ...) can only beat the notch if psi outside the cone is predictable from psi inside it. This
script measures that with the TRUE residual (an upper bound for anything that sees the field instead): a small 3D U-Net gets cone(psi) * mask (+ mask, magnitude) and
predicts (psi - cone(psi)) * mask. Train on patches with x < 100, test on x >= 112 (the RC1 hold-out slab of every earlier run). Cone = smooth window |D| < tau
(erf width 0.03), tau 0.1 and 0.2. Residuals: the MEASURED tissue residual (psi = phs_tissue - D*chi_cosmos), the modelled STI and off-diagonal residuals, and a white-noise
control with the rms of the measured one (stationary, so nothing should be predictable: R^2 ~ 0).
Reported on the test slab: R^2 of the non-cone part, and the fraction of the psi energy that REMAINS after (a) the notch alone and (b) the notch plus the prediction.
Run (server or laptop): MCNO_VRAM_GB=10 .venv/bin/python scripts/feasibility_psi_cone.py > runs/feasibility_psi_cone.log 2>&1      Smoke test: python -I scripts/feasibility_psi_cone.py --dry
Writes runs/feasibility_psi_cone/table.txt (table_dry.txt)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import Volume  # noqa: E402
from mcno_qsm.gainnet import UNet3d  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import load_rc1  # noqa: E402
from mcno_qsm.scoring import limit_vram  # noqa: E402

TEST_START, TRAIN_END = 112, 100        # RC1 hold-out of the earlier runs (axis 0): train x < 100, gap, test x >= 112


def cone_window(shape, voxel, tau, device):
    d = dipole_kernel(shape, voxel, device=device).abs()
    return 1.0 - 0.5 * (1.0 + torch.erf((d - tau) / (np.sqrt(2) * 0.03)))


def cone_part(x, window):
    return torch.fft.ifftn(torch.fft.fftn(x) * window).real


def train_and_test(psi, mask, magnitude, window, args, device, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cone = cone_part(psi, window) * mask
    rest = (psi - cone_part(psi, window)) * mask
    scale = float(cone[mask > 0].std())
    x = torch.stack([cone / scale, mask, magnitude])[None]                  # [1,3,X,Y,Z]
    y = (rest / scale)[None, None]
    net = UNet3d(3, 1, args.base).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    size, n = args.patch, psi.shape
    m_cpu = mask.cpu().numpy()
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 * 0.5 * (1 + np.cos(np.pi * step / args.steps))
        xs, ys, ms = [], [], []
        while len(xs) < args.batch:
            o = [int(rng.integers(0, max(1, TRAIN_END - size + 1))), int(rng.integers(0, n[1] - size + 1)), int(rng.integers(0, n[2] - size + 1))]
            sl = tuple(slice(a, a + size) for a in o)
            if m_cpu[sl].mean() < 0.3:
                continue
            xs.append(x[0][(slice(None),) + sl]), ys.append(y[0][(slice(None),) + sl]), ms.append(mask[sl][None])
        xb, yb, mb = torch.stack(xs).to(device), torch.stack(ys).to(device), torch.stack(ms).to(device)
        loss = ((net(xb) - yb) ** 2 * mb).sum() / mb.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    lo = TEST_START - 16                                                        # context before the test slab; only x >= TEST_START is scored
    with torch.no_grad():
        pred = net(x[:, :, lo:].to(device))[0, 0].cpu() * scale
    region = mask[TEST_START:] > 0
    t = rest[TEST_START:][region]
    p = pred[TEST_START - lo:][region]
    psi_energy = float((psi[TEST_START:][region] ** 2).sum())
    r2 = 1 - float(((t - p) ** 2).sum() / (t ** 2).sum())
    corr = float(np.corrcoef(t.numpy(), p.numpy())[0, 1])
    return {"r2": r2, "corr": corr, "cone_share": float((cone[TEST_START:][region] ** 2).sum()) / psi_energy,
            "remain_notch": float((t ** 2).sum()) / psi_energy, "remain_pred": float(((t - p) ** 2).sum()) / psi_energy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patch", type=int, default=64)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--out", type=Path, default=Path("runs/feasibility_psi_cone"))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        args.steps, args.batch, args.patch, args.base, args.taus = 3, 1, 32, 4, [0.1]
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    d = load_rc1(args.rc1)
    v = Volume.from_rc1(d, "cpu")
    mask, magnitude = v.mask, v.magnitude
    clean = v.clean * mask
    residuals = {"tissue (MEASURED)": (v.tissue - v.clean) * mask, "sti (modelled)": (v.sti - v.clean) * mask, "offdiag (modelled)": v.offdiag * mask}
    g = torch.Generator().manual_seed(1)
    rms = float(residuals["tissue (MEASURED)"][mask > 0].square().mean().sqrt())
    residuals["white noise (control)"] = torch.randn(mask.shape, generator=g) * rms * mask
    lines = [f"is the non-cone part of psi predictable from its cone part? RC1, train x<{TRAIN_END}, test x>={TEST_START}, {args.steps} steps, single seed",
             f"{'residual':24s} {'tau':>5s} {'cone share':>10s} {'R2 rest':>8s} {'corr':>6s} {'psi left: notch':>16s} {'notch+pred':>11s}"]
    for name, psi in residuals.items():
        for tau in args.taus:
            window = cone_window(psi.shape, d["voxel_size"], tau, "cpu")
            r = train_and_test(psi, mask, magnitude, window, args, device)
            lines.append(f"{name:24s} {tau:5.2f} {100 * r['cone_share']:9.1f}% {r['r2']:8.3f} {r['corr']:6.3f} {100 * r['remain_notch']:15.1f}% {100 * r['remain_pred']:10.1f}%")
            print(lines[-1], flush=True)
    text = "\n".join(lines)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
