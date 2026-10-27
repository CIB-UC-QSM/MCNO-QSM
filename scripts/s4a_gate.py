"""S4a gate: sanity report on the architecture and the RC1 training set, before any training.

Reads RC1 for the training-set statistics and RC2 (Sim2) only as the reference to compare against; the
training source itself never reads RC2 (checked below). Writes a JSON report and PNGs.
Run: uv run python -I scripts/s4a_gate.py --out runs/s4a_gate
"""

import argparse
import builtins
import json
import time
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from mcno_qsm.artifacts import FAMILIES, ArtifactConfig, Volume, make_dirty
from mcno_qsm.cleaner import PhaseCleaner
from mcno_qsm.noise import calibrate_snr
from mcno_qsm.physics import dipole_kernel
from mcno_qsm.rc1 import load_rc1
from mcno_qsm.rc1_data import RC1Source
from mcno_qsm.rc2 import load_rc2
from mcno_qsm.scoring import forward_field


def psi_stats(psi, mask, voxel):
    """Same definitions as scripts/s0_sim2_diagnostics.py, with k in cycles/mm."""
    inside = psi[mask]
    power = np.abs(np.fft.fftn(psi)) ** 2
    freqs = [np.fft.fftfreq(n, d=v) for n, v in zip(psi.shape, voxel)]
    radius = np.sqrt(sum(f**2 for f in np.meshgrid(*freqs, indexing="ij")))
    cone = np.abs(dipole_kernel(psi.shape, voxel, dtype=torch.float64).numpy()) < 0.03
    z = (inside - inside.mean()) / inside.std()
    total = power.sum()
    return {
        "rms_ppm": float(inside.std()), "max_abs_ppm": float(np.abs(inside).max()),
        "p99.9_abs_ppm": float(np.percentile(np.abs(inside), 99.9)),
        "excess_kurtosis": float((z**4).mean() - 3),
        "energy_k>0.2": float(power[radius > 0.2].sum() / total),
        "energy_k<0.02": float(power[radius < 0.02].sum() / total),
        "energy_|D|<0.03": float(power[cone].sum() / total),
    }


def radial_spectrum(psi, voxel, bins=40):
    freqs = [np.fft.fftfreq(n, d=v) for n, v in zip(psi.shape, voxel)]
    radius = np.sqrt(sum(f**2 for f in np.meshgrid(*freqs, indexing="ij")))
    power = np.abs(np.fft.fftn(psi)) ** 2
    edges = np.linspace(0, 0.5, bins + 1)
    index = np.digitize(radius.ravel(), edges) - 1
    keep = index < bins
    s = np.bincount(index[keep], power.ravel()[keep], minlength=bins) / np.maximum(
        np.bincount(index[keep], minlength=bins), 1)
    return 0.5 * (edges[1:] + edges[:-1]), s / s.sum()


def architecture(report, config, rc1_voxel, sim2_voxel, fast):
    section = {}
    model = PhaseCleaner(rc1_voxel, dipolelet=config["dipolelet"], **config["model"])
    plain = PhaseCleaner(rc1_voxel, dipolelet=config["dipolelet"], plain=True, **config["model"])
    count = lambda m: sum(p.numel() for p in m.parameters())
    section["parameters"] = {
        "total": count(model),
        "lift": count(model.lift), "spatial_dilated": count(model.spatial),
        "blocks": count(model.blocks), "head": count(model.head),
        "per_block_segment_mix_complex_CxC": int(model.blocks[0].real.numel() + model.blocks[0].imag.numel()),
        "per_block_k_gain_mlp": count(model.blocks[0].gain), "per_block_skip_1x1": count(model.blocks[0].skip),
        "plain_fno_ablation_total": count(plain), "segments_dipolelet": len(model.blocks[0].real),
        "segments_plain": len(plain.blocks[0].real),
    }
    local = 3 + sum(model.spatial[i].dilation[0] for i in range(len(model.spatial)))
    section["local_receptive_radius_voxels"] = {
        "lift_7x7x7": 3, "dilated_3x3x3_dilations": [c.dilation[0] for c in model.spatial], "total": local,
        "note": "the FFT branch of every block is global (every voxel sees the whole patch/volume)"}

    torch.manual_seed(0)
    shape = (48, 48, 48)
    g = torch.Generator().manual_seed(0)
    field = torch.randn(1, 1, *shape, generator=g) * 0.01
    mask = (torch.rand(1, 1, *shape, generator=g) > 0.3).float()
    weight = torch.rand(1, 1, *shape, generator=g)
    with torch.no_grad():
        out = model(field, mask, weight, voxel_size=rc1_voxel)
        plain_out = plain(field, mask, weight, voxel_size=rc1_voxel)
    section["identity_at_init_max_abs_error"] = {
        "dipolelet": float((out - field * mask).abs().max()), "plain": float((plain_out - field * mask).abs().max())}

    # Influence of the input on one output voxel (random head so the output depends on the input).
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)
        for block in model.blocks:
            if block.gain is not None:
                block.gain[-1].weight.normal_(0, 0.05)
    x = field.clone().requires_grad_(True)
    model(x, torch.ones_like(mask), weight, voxel_size=rc1_voxel)[0, 0, 24, 24, 24].backward()
    influence = x.grad[0, 0].abs()
    coords = torch.stack(torch.meshgrid(*[torch.arange(48) - 24] * 3, indexing="ij")).abs().amax(0)
    total = influence.sum()
    section["influence_share_within_radius_of_one_output_voxel"] = {
        f"r<={r}": float(influence[coords <= r].sum() / total) for r in (3, 7, 10, 15, 20, 24)}

    # Gradient flow with a live head/gain: every parameter must receive a finite gradient.
    model.zero_grad()
    model(field, mask, weight, voxel_size=rc1_voxel).square().sum().backward()
    bad = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    zero = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() == 0]
    section["gradient_flow"] = {"missing_or_nonfinite": bad, "all_zero_gradient": zero}

    if not fast:
        volumes = {"RC1_160^3": (rc1_voxel, (160, 160, 160)), "Sim2_164x205x205": (sim2_voxel, (164, 205, 205))}
        full = {}
        for name, (voxel, volume_shape) in volumes.items():
            f = torch.randn(1, 1, *volume_shape) * 0.01
            m = torch.ones_like(f)
            started = time.perf_counter()
            with torch.no_grad():
                y = model(f, m, m, voxel_size=voxel)
            full[name] = {"voxel_mm": [round(v, 4) for v in voxel], "finite": bool(torch.isfinite(y).all()),
                          "output_shape": list(y.shape), "cpu_seconds": round(time.perf_counter() - started, 1)}
        section["full_volume_forward_cpu"] = full
    section["gpu_memory_server_measured_2026-10-20"] = {
        "96^3 batch1": "2.68 GB", "96^3 batch2": "5.10 GB", "128^3 batch1": "6.29 GB", "128^3 batch2": "12.0 GB"}
    report["architecture"] = section


def to_np(t):
    return t.detach().cpu().numpy()


def training_set(report, args, out):
    section = {}
    data = load_rc1(args.rc1)
    volume = Volume.from_rc1(data, "cpu")
    voxel, mask = data["voxel_size"], data["mask"]
    target = to_np(volume.target)
    rc2 = load_rc2(args.rc2)
    rc2_mask = rc2["mask"]
    rc2_clean = forward_field(rc2["chi"], rc2_mask, rc2["voxel_size"])
    rc2_psi = rc2["field"] * rc2_mask - rc2_clean
    reference = psi_stats(rc2_psi, rc2_mask, rc2["voxel_size"])
    section["sim2_reference_psi"] = reference
    section["clean_target_in_mask"] = {
        "rc1_rms_ppm": float(target[mask].std()), "sim2_rms_ppm": float(rc2_clean[rc2_mask].std()),
        "rc1_mask_fraction": float(mask.mean()), "sim2_mask_fraction": float(rc2_mask.mean()),
        "rc1_voxel_mm": voxel, "sim2_voxel_mm": rc2["voxel_size"]}

    # Calibration: SNR that gives Sim2's psi rms with the RC1 magnitude.
    cfg = ArtifactConfig()
    snr_match = calibrate_snr(volume.clean, volume.magnitude * volume.mask, volume.mask, volume.phs_scale,
                              reference["rms_ppm"])
    section["snr_matching_sim2_psi_rms"] = {"snr": snr_match, "default_range": [cfg.snr_low, cfg.snr_high]}

    rows, psis = {}, {}
    for family in FAMILIES:
        draws = 1 if family in ("tissue", "sti", "offdiag") else args.draws
        stats, snrs = [], []
        for seed in range(draws):
            rng = np.random.default_rng(seed)
            generator = torch.Generator().manual_seed(seed)
            dirty, info = make_dirty(family, volume, rng, generator, cfg)
            psi = to_np(dirty) * mask - target
            if seed == 0:
                psis[family] = (to_np(dirty), psi)
            stats.append(psi_stats(psi, mask, voxel))
            if "snr" in info:
                snrs.append(info["snr"])
        keys = stats[0].keys()
        rows[family] = {k: float(np.mean([s[k] for s in stats])) for k in keys}
        rows[family]["rms_ppm_range"] = [min(s["rms_ppm"] for s in stats), max(s["rms_ppm"] for s in stats)]
        if snrs:
            rows[family]["snr_draws"] = [round(s, 1) for s in snrs]
        rows[family]["draws"] = draws
    section["families_vs_sim2"] = rows

    # Patches: mask coverage and where the brain sits along each axis.
    source = RC1Source(args.rc1, "cpu", cfg, holdout={"axis": 2, "train_fraction": 0.7, "gap": 16})
    rng = np.random.default_rng(0)
    fractions = []
    for _ in range(500):
        high = [n - 96 for n in source.shape]
        origin = [int(rng.integers(0, h + 1)) for h in high]
        fractions.append(float(mask[tuple(slice(o, o + 96) for o in origin)].mean()))
    fractions = np.array(fractions)
    extent = {ax: [int(np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0][i]) for i in (0, -1)]
              for ax in range(3)}
    section["patches_96^3_unrestricted_origin"] = {
        "share_with_mask>5%": float((fractions > 0.05).mean()), "mean_mask_fraction": float(fractions.mean()),
        "p5_p50_p95": [float(np.percentile(fractions, q)) for q in (5, 50, 95)]}
    section["rc1_mask_extent_per_axis"] = extent
    section["holdout_example"] = {"axis": 2, "train_range": source.train_range, "test_range": source.test_range,
                                  "mask_voxels_train": int(source.regions["_train"].sum()),
                                  "mask_voxels_test": int(source.regions["_test"].sum())}

    # Determinism by seed and no RC2 access in the training path.
    def sample(seed):
        return source.patch(96, np.random.default_rng(seed), torch.Generator().manual_seed(seed))

    a, b, c = sample(5), sample(5), sample(6)
    section["determinism"] = {"same_seed_identical": all(torch.equal(x, y) for x, y in zip(a[:4], b[:4])),
                              "different_seed_differs": not torch.equal(a[0], c[0])}
    opened = []
    real_open, real_loadmat = builtins.open, sio.loadmat
    builtins.open = lambda f, *x, **k: (opened.append(str(f)), real_open(f, *x, **k))[1]
    sio.loadmat = lambda f, *x, **k: (opened.append(str(f)), real_loadmat(f, *x, **k))[1]
    try:
        fresh = RC1Source(args.rc1, "cpu", cfg)
        for seed in range(12):
            fresh.patch(64, np.random.default_rng(seed), torch.Generator().manual_seed(seed))
    finally:
        builtins.open, sio.loadmat = real_open, real_loadmat
    section["training_source_files_opened"] = sorted({Path(p).name for p in opened})
    section["training_source_touches_rc2"] = any("RC2" in p.upper() for p in opened)
    report["training_set"] = section

    plots(out, psis, volume, target, mask, voxel, rc2_psi, rc2_mask, rc2["voxel_size"], rc2_clean)


def plots(out, psis, volume, target, mask, voxel, rc2_psi, rc2_mask, rc2_voxel, rc2_clean):
    families = list(psis)
    fig, axes = plt.subplots(len(families), 3, figsize=(10, 3.1 * len(families)))
    for row, family in enumerate(families):
        dirty, psi = psis[family]
        z = int(np.argmax(np.abs(psi).max(axis=(0, 1)))) if family not in ("noise", "spheres_chi") else 80
        for col, (image, title, lim) in enumerate(((dirty * mask, "dirty (masked)", 0.06),
                                                   (target, "target mask*D*chi", 0.06), (psi, "psi = dirty - target", None))):
            slice_ = image[:, :, z].T
            vmax = lim or max(np.abs(slice_).max() * 0.8, 1e-4)
            axes[row, col].imshow(slice_, cmap="gray", vmin=-vmax, vmax=vmax, origin="lower")
            axes[row, col].set_title(f"{family}: {title}  z={z}", fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out / "families_slices.png", dpi=80)
    plt.close(fig)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.5))
    edges = np.logspace(-5, 0.6, 80)
    for family, (_, psi) in psis.items():
        h, _ = np.histogram(np.abs(psi[mask]), bins=edges)
        ax1.step(edges[:-1], h / h.sum(), label=family, where="post")
    h, _ = np.histogram(np.abs(rc2_psi[rc2_mask]), bins=edges)
    ax1.step(edges[:-1], h / h.sum(), "k--", lw=2, label="Sim2 psi")
    ax1.set(xscale="log", yscale="log", xlabel="|psi| in mask (ppm)", ylabel="share of voxels")
    ax1.legend(fontsize=6)
    centers, spectrum = radial_spectrum(rc2_psi, rc2_voxel)
    ax2.semilogy(centers, spectrum, "k--", lw=2, label="Sim2 psi")
    for family, (_, psi) in psis.items():
        c, s = radial_spectrum(psi, voxel)
        ax2.semilogy(c, s, label=family)
    ax2.set(xlabel="|k| (cycles/mm)", ylabel="normalized power per bin", title="psi spectrum")
    ax2.legend(fontsize=6)
    edges = np.linspace(-0.12, 0.12, 120)
    ax3.hist(target[mask], bins=edges, histtype="step", density=True, label="RC1 target")
    ax3.hist(rc2_clean[rc2_mask], bins=edges, histtype="step", density=True, label="Sim2 target")
    ax3.set(xlabel="target field in mask (ppm)", yscale="log", title="clean targets")
    ax3.legend()
    fig.tight_layout()
    fig.savefig(out / "histograms_spectra.png", dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--config", default="configs/sim2_clean.toml")
    parser.add_argument("--out", type=Path, default=Path("runs/s4a_gate"))
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--fast", action="store_true", help="skip the two full-volume CPU forwards")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.config, "rb") as stream:
        config = tomllib.load(stream)
    rc1_voxel = load_rc1(args.rc1)["voxel_size"]
    sim2_voxel = load_rc2(args.rc2)["voxel_size"]
    report = {}
    architecture(report, config, rc1_voxel, sim2_voxel, args.fast)
    training_set(report, args, args.out)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
