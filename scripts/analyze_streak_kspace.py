"""Where does the energy sit in k-space (Dipole-let scale x |D| band) and which part of it makes the streaks? (CLAUDE.md 9ap)

Local, CPU only, no training, real Sim2 (one volume). Uses runs/chi_fixed25.npz (eNDI chi at 25 fixed iterations of dirty / oracle /
noise_clean / noise_clean_calc) and runs/s4b_<method>/best_clean.nii.gz. Notation: b = mask*D*chi (clean target), psi = mask*(dirty - b),
r = mask*(cleaned - b), e_chi = mask*(chi_method - chi_true). Segment energies are Parseval sums of the Dipole-let spectra; segments
overlap, so shares are normalised to sum to 1 (raw sums printed too).

Tables written to runs/streak_kspace/: structure.txt (geometry of the 10 segments), energy.txt (shares), budget.txt (linearised streak
budget), transfer.txt (empirical eNDI gain versus |D|), separability.txt (|b_k|^2/|psi_k|^2 map and the boundary D*(k)), plus PNGs.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

OUT = Path("runs/streak_kspace")
OUT.mkdir(parents=True, exist_ok=True)
NAMES = [f"s{j}/b{k}" for j in (1, 2, 3) for k in (0, 1, 2)] + ["coarse"]
D_EDGES = np.array([0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.27, 0.333, 0.5, 0.67])
K_EDGES = np.geomspace(0.01, 0.9, 13)


def seg_spectra(tr, x):
    return [s[0, 0] for s in tr.analysis(torch.from_numpy(x)[None, None])]


def seg_energy(tr, x):
    return np.array([float((s.abs() ** 2).sum()) for s in seg_spectra(tr, x)])


def write(name, text):
    (OUT / name).write_text(text)
    print(text)


def main():
    data = load_rc2("dataRC2", 7.0)
    mask, chi, voxel = data["mask"].astype(np.float32), data["chi"], data["voxel_size"]
    shape = chi.shape
    b = forward_field(chi, mask, voxel)
    dirty = data["field"] * mask
    clean = {m: nib.load(f"runs/s4b_{m}/best_clean.nii.gz").get_fdata(dtype=np.float32) * mask for m in ("noise", "noise_clean")}
    z = np.load("runs/chi_fixed25.npz")
    chis = {k[4:]: z[k].astype(np.float32) * mask for k in z.files}
    tr = DipoleletTransform(shape, voxel, **SIM2_DIPOLELET)
    kern = dipole_kernel(shape, voxel, dtype=torch.float64).numpy().astype(np.float32)
    freqs = np.meshgrid(*[np.fft.fftfreq(n, d=1.0) for n in shape], indexing="ij")  # cycles per voxel (voxels are ~1 mm isotropic)
    kmag = np.sqrt(sum(f ** 2 for f in freqs)).astype(np.float32)
    adk = np.abs(kern)

    # ---- structure: geometry of the segments from a white-noise realisation (energy share = k-space mass) ----
    rng = np.random.default_rng(0)
    white = rng.standard_normal(shape).astype(np.float32)
    spectra = seg_spectra(tr, white)
    tot = sum(float((s.abs() ** 2).sum()) for s in spectra)
    lines = ["Geometry of the 10 Dipole-let segments (white-noise probe; k in cycles/voxel, Nyquist 0.5)",
             f"{'segment':8s} {'mass%':>7s} {'<|k|>':>7s} {'<|D|>':>7s} {'|D| p10':>8s} {'|D| p90':>8s}"]
    for name, s in zip(NAMES, spectra):
        e = (s.abs() ** 2).numpy()
        w = e / e.sum()
        mk, md = float((w * kmag).sum()), float((w * adk).sum())
        order = np.argsort(adk.ravel())
        cw = np.cumsum(w.ravel()[order])
        p10, p90 = adk.ravel()[order][np.searchsorted(cw, [0.1, 0.9])]
        lines.append(f"{name:8s} {100 * e.sum() / tot:7.2f} {mk:7.3f} {md:7.3f} {p10:8.3f} {p90:8.3f}")
    write("structure.txt", "\n".join(lines))

    # ---- energy shares ----
    sig = {"b (true field)": b, "psi dirty": dirty - b, "r noise (cleaned-b)": clean["noise"] - b, "r noise_clean": clean["noise_clean"] - b,
           "chi true": chi * mask}
    for m, c in chis.items():
        sig[f"e_chi {m}"] = c - chi * mask
    shares = {}
    lines = ["Share of segment energy (%, normalised to the sum of the 10 segments; raw sum/total energy in the last column)",
             f"{'signal':26s} " + " ".join(f"{n:>6s}" for n in NAMES) + "  raw/tot"]
    for name, x in sig.items():
        e = seg_energy(tr, x.astype(np.float32))
        shares[name] = e / e.sum()
        lines.append(f"{name:26s} " + " ".join(f"{100 * v:6.2f}" for v in shares[name]) + f"  {e.sum() / float((x.astype(np.float64) ** 2).sum()):.2f}")
    write("energy.txt", "\n".join(lines))

    # ---- linearised streak budget: chi error produced by the segment of a field error through D_lam^+ ----
    lines = ["Linearised chi-error budget of a field error x: share of ||D_lam^+ x_s||^2 per segment (D_lam^+ = D/(D^2+lam), %)"]
    budget = {}
    for lam in (0.003, 0.01, 0.03):
        inv = (kern / (kern ** 2 + lam)).astype(np.float32)
        inv[0, 0, 0] = 0
        lines.append(f"lam={lam}  (cuts D below ~{np.sqrt(lam):.2f})")
        lines.append(f"{'signal':26s} " + " ".join(f"{n:>6s}" for n in NAMES) + "  tot(chi-units)")
        for name in ("psi dirty", "r noise_clean", "b (true field)"):
            x = sig[name].astype(np.float32)
            segs = seg_spectra(tr, x)
            e = np.array([float(((s.numpy() * inv) ** 2 if False else (torch.from_numpy(inv) * s).abs() ** 2).sum()) for s in segs])
            budget[(lam, name)] = e / e.sum()
            chi_norm = float(((torch.from_numpy(inv) * torch.fft.fftn(torch.from_numpy(x), norm="ortho")).abs() ** 2).sum()) ** 0.5
            lines.append(f"{name:26s} " + " ".join(f"{100 * v:6.2f}" for v in budget[(lam, name)]) + f"  {chi_norm / np.linalg.norm(chi * mask):.3f}")
    write("budget.txt", "\n".join(lines))

    # ---- empirical eNDI transfer: |dchi_k|^2 / |psi_k|^2 versus |D| (dirty - oracle, both eNDI 25 iterations) ----
    psi = (dirty - b).astype(np.float32)
    dchi = (chis["dirty"] - chis["oracle"]).astype(np.float32)
    Pf = np.abs(np.fft.fftn(psi)) ** 2
    Cf = np.abs(np.fft.fftn(dchi)) ** 2
    lines = ["Empirical eNDI gain T^2 = sum|dchi_k|^2 / sum|psi_k|^2 (dchi = eNDI(dirty) - eNDI(oracle)) per |D| bin and |k| range; 1/D^2 for reference",
             f"{'|D| bin':>12s} {'1/D2':>8s} " + " ".join(f"k{lo:.2f}-{hi:.2f}".rjust(12) for lo, hi in [(0.01, 0.06), (0.06, 0.15), (0.15, 0.9)])]
    transfer = np.zeros((len(D_EDGES) - 1, 3))
    for i in range(len(D_EDGES) - 1):
        row = []
        for j, (lo, hi) in enumerate([(0.01, 0.06), (0.06, 0.15), (0.15, 0.9)]):
            sel = (adk >= D_EDGES[i]) & (adk < D_EDGES[i + 1]) & (kmag >= lo) & (kmag < hi)
            transfer[i, j] = Cf[sel].sum() / max(Pf[sel].sum(), 1e-30)
            row.append(f"{transfer[i, j]:12.2f}")
        mid = 0.5 * (D_EDGES[i] + D_EDGES[i + 1])
        lines.append(f"{D_EDGES[i]:5.3f}-{D_EDGES[i + 1]:5.3f} {1 / mid ** 2:8.1f} " + " ".join(row))
    write("transfer.txt", "\n".join(lines))

    # ---- separability map: |b_k|^2 / |psi_k|^2 on (|k|, |D|) ----
    Bf = np.abs(np.fft.fftn(b)) ** 2
    Rf = np.abs(np.fft.fftn(clean["noise_clean"] - b)) ** 2
    Ef = {m: np.abs(np.fft.fftn(c - chi * mask)) ** 2 for m, c in chis.items()}
    Xf = np.abs(np.fft.fftn(chi * mask)) ** 2
    nk, nd = len(K_EDGES) - 1, len(D_EDGES) - 1
    maps = {n: np.zeros((nk, nd)) for n in ("b", "psi", "r", "chi", "e_dirty", "e_oracle", "e_nc")}
    for i in range(nk):
        ks = (kmag >= K_EDGES[i]) & (kmag < K_EDGES[i + 1])
        for j in range(nd):
            sel = ks & (adk >= D_EDGES[j]) & (adk < D_EDGES[j + 1])
            for n, arr in (("b", Bf), ("psi", Pf), ("r", Rf), ("chi", Xf), ("e_dirty", Ef["dirty"]), ("e_oracle", Ef["oracle"]), ("e_nc", Ef["noise_clean"])):
                maps[n][i, j] = arr[sel].sum()
    ratio = maps["b"] / np.maximum(maps["psi"], 1e-30)
    lines = ["Local SNR of the true field over the dirty psi, 10log10(sum|b_k|^2 / sum|psi_k|^2) per (|k| bin, |D| bin); rows |k| (cycles/voxel), columns |D| upper edge",
             " " * 12 + " ".join(f"{e:6.3f}" for e in D_EDGES[1:])]
    for i in range(nk):
        lines.append(f"{K_EDGES[i]:5.3f}-{K_EDGES[i + 1]:5.3f} " + " ".join(f"{10 * np.log10(max(ratio[i, j], 1e-12)):6.1f}" for j in range(nd)))
    # boundary D*(k): smallest |D| where the true field overtakes psi (ratio >= 1), per |k| bin
    star = []
    for i in range(nk):
        above = np.where(ratio[i] >= 1)[0]
        star.append(D_EDGES[above[0]] if len(above) else np.nan)
    lines.append("D*(k) (lower edge of the first |D| bin with |b|^2 >= |psi|^2): " + ", ".join(f"{K_EDGES[i]:.3f}:{star[i]:.3f}" for i in range(nk)))
    # chi error share that lies where the field was NOT separable (|b|<|psi|) versus where it was
    for name, key in (("dirty", "e_dirty"), ("oracle", "e_oracle"), ("noise_clean", "e_nc")):
        unsep = ratio < 1
        lines.append(f"chi error energy of {name} in the cells with |b|^2<|psi|^2 (near-cone side): {100 * maps[key][unsep].sum() / maps[key].sum():.1f}% "
                     f"(those cells hold {100 * maps['chi'][unsep].sum() / maps['chi'].sum():.1f}% of the chi energy, {100 * maps['b'][unsep].sum() / maps['b'].sum():.2f}% of b, {100 * maps['psi'][unsep].sum() / maps['psi'].sum():.1f}% of psi)")
    write("separability.txt", "\n".join(lines))

    # ---- figures ----
    fig, ax = plt.subplots(2, 4, figsize=(21, 9))
    ext = [0, nd, nk, 0]
    for a, (n, t) in zip(ax.ravel(), [("b", "true field |b|^2"), ("psi", "dirty psi"), ("r", "left after noise_clean"), ("chi", "chi true"),
                                      ("e_dirty", "chi error dirty"), ("e_nc", "chi error noise_clean"), ("e_oracle", "chi error oracle")]):
        m = np.log10(np.maximum(maps[n] / maps[n].sum(), 1e-12))
        im = a.imshow(m, aspect="auto", vmin=-6, vmax=m.max(), extent=ext)
        a.set_title("log10 energy share: " + t)
        a.set_xticks(np.arange(nd + 1)), a.set_xticklabels([f"{e:.2f}" for e in D_EDGES], rotation=60, fontsize=7)
        a.set_yticks(np.arange(nk + 1)), a.set_yticklabels([f"{e:.2f}" for e in K_EDGES], fontsize=7)
        a.set_xlabel("|D|"), a.set_ylabel("|k| (cycles/voxel)")
        plt.colorbar(im, ax=a)
    im = ax.ravel()[7].imshow(10 * np.log10(np.maximum(ratio, 1e-12)), aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30, extent=ext)
    ax.ravel()[7].set_title("local SNR dB: true field / psi (blue: psi wins)")
    ax.ravel()[7].set_xticks(np.arange(nd + 1)), ax.ravel()[7].set_xticklabels([f"{e:.2f}" for e in D_EDGES], rotation=60, fontsize=7)
    ax.ravel()[7].set_yticks(np.arange(nk + 1)), ax.ravel()[7].set_yticklabels([f"{e:.2f}" for e in K_EDGES], fontsize=7)
    plt.colorbar(im, ax=ax.ravel()[7])
    plt.tight_layout()
    plt.savefig(OUT / "kspace_maps.png", dpi=110)
    plt.close()

    # streak picture: dirty chi error split into near-cone and far parts
    for thr in (0.1,):
        e = chis["dirty"] - chi * mask
        Ek = np.fft.fftn(e)
        near = np.real(np.fft.ifftn(Ek * (adk < thr))) * mask
        far = e - near
        z0 = shape[2] // 2 + 10
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for a, (im_, t) in zip(ax, [(chi * mask, "chi true"), (e, "eNDI dirty error"), (near, f"error part |D|<{thr} (streaks)"), (far, f"error part |D|>={thr}")]):
            a.imshow(im_[:, :, z0].T, cmap="gray", vmin=-0.15 if "true" not in t else -0.1, vmax=0.15 if "true" not in t else 0.1, origin="lower")
            a.set_title(t), a.axis("off")
        plt.tight_layout()
        plt.savefig(OUT / "streak_split.png", dpi=100)
        plt.close()
        frac = {m: float((np.fft.ifftn(np.fft.fftn(c - chi * mask) * (adk < thr)).real * mask).std() ** 2 / (c - chi * mask).std() ** 2) for m, c in chis.items()}
        write("near_cone_fraction.txt", f"fraction of chi-error variance with |D|<{thr}: " + json.dumps(frac))

    # cumulative energy versus |D|
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = D_EDGES[1:]
    for n, lab in (("chi", "chi true"), ("b", "true field"), ("psi", "psi dirty"), ("e_dirty", "chi err dirty"), ("e_nc", "chi err noise_clean"), ("e_oracle", "chi err oracle")):
        ax[0].plot(xs, np.cumsum(maps[n].sum(0)) / maps[n].sum(), "o-", label=lab)
    ax[0].set_xlabel("|D| upper edge"), ax[0].set_ylabel("cumulative energy fraction"), ax[0].set_xscale("log"), ax[0].legend(), ax[0].grid(alpha=.3)
    for j, lab in enumerate(["|k| .01-.06", "|k| .06-.15", "|k| .15-.9"]):
        ax[1].loglog(0.5 * (D_EDGES[:-1] + D_EDGES[1:]), transfer[:, j], "o-", label=lab)
    d = 0.5 * (D_EDGES[:-1] + D_EDGES[1:])
    ax[1].loglog(d, 1 / d ** 2, "k--", label="1/D^2"), ax[1].set_xlabel("|D|"), ax[1].set_ylabel("T^2 = |dchi|^2/|psi|^2"), ax[1].legend(), ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT / "cumulative_transfer.png", dpi=110)
    np.savez(OUT / "maps.npz", **maps, ratio=ratio, transfer=transfer, d_edges=D_EDGES, k_edges=K_EDGES)


if __name__ == "__main__":
    main()
