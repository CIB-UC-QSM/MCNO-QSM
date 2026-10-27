"""Second-generation Dipole-let design from the data (CLAUDE.md 9q / 9ax). CPU, no network, no eNDI (phase-domain linear-filter proxy; eNDI validation is a later step).

1. Cell statistics on a (|k| [cycles/voxel], |D|) grid for each regime: E_b = |B|^2, E_bd = Re(B conj D), E_d = |D|^2 summed per cell (B = FFT of the clean masked field,
   D = FFT of the masked dirty field). Regimes: 'real' = RC1 STI+noise and off-diagonal+noise (SNR 100; the fit set, never the measured tissue), 'noise' = RC1 noise SNR 60,
   'sim2' = the real Sim2 psi (noise-like, only to check that the noise design agrees; not used to design the real one).
2. Best partition of the plane into S segments: a radial partition into R groups and, inside each radial group, its OWN partition of |D| into bands (a Dipole-let segment is a
   radial window x |D| band). Each segment has one passive gain g = clip(sum E_bd / sum E_d, 0, 1) (the passive filter of 9au); loss = sum over segments of E_b - 2 g E_bd + g^2 E_d.
   Dynamic programming over contiguous radial groups and contiguous |D| bands, S = 4 ... 20.
3. Evaluation in the image domain (post-masking, as the real filter) with smooth partition-of-unity windows (logistic edges): gains fitted (bounded [0, 1]) on the real-regime fit
   set and applied to the HELD-OUT measured tissue residual (exp7); the current 12-segment Dipole-let (SIM2_DIPOLELET + coarse_bands 3) under the SAME protocol is the baseline.
Run: python -I scripts/design_dipolelet2.py     Writes runs/dipolelet2/design.txt and designs.json."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, image_components  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

OUT = Path("runs/dipolelet2")
OUT.mkdir(parents=True, exist_ok=True)
lines = []


def say(text=""):
    print(text, flush=True)
    lines.append(text)


KE = np.concatenate([[0.0], np.geomspace(0.012, 0.87, 23), [1.0]])                 # |k| bin edges, cycles / voxel (24 bins)
DE = np.array([0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.27, 0.333, 0.5, 0.667 + 1e-6])   # |D| bin edges (12 bins)
NK, ND = len(KE) - 1, len(DE) - 1


def axes(shape, voxel):
    kern = np.abs(dipole_kernel(shape, voxel, dtype=torch.float64).numpy())
    fr = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")           # cycles / voxel
    return kern, np.sqrt(sum(f ** 2 for f in fr))


def cell_stats(b, d, kern, kmag):
    ik = np.clip(np.digitize(kmag.ravel(), KE) - 1, 0, NK - 1)
    idd = np.clip(np.digitize(kern.ravel(), DE) - 1, 0, ND - 1)
    flat = ik * ND + idd
    B, Dd = np.fft.fftn(b).ravel(), np.fft.fftn(d).ravel()
    acc = lambda w: np.bincount(flat, weights=w, minlength=NK * ND).reshape(NK, ND)
    return np.stack([acc(np.abs(B) ** 2), acc(np.real(B * np.conj(Dd))), acc(np.abs(Dd) ** 2)])      # [3, NK, ND]


def seg_loss(eb, ebd, ed):
    g = np.clip(ebd / np.maximum(ed, 1e-30), 0.0, 1.0)
    return eb - 2 * g * ebd + g * g * ed, g


def best_partition(stats, S_max=20, R_max=6):
    """DP: radial groups (contiguous bins) each with its own contiguous |D| bands. Returns {S: (loss, [(k_lo, k_hi, [d_edges...])])}."""
    cum = np.concatenate([np.zeros((3, 1, ND)), np.cumsum(stats, axis=1)], axis=1)            # cum over radial bins
    inner = {}                                                                                  # (u, t) -> best[m] = (loss, d_edges)
    for u in range(NK):
        for t in range(u + 1, NK + 1):
            row = cum[:, t] - cum[:, u]                                                         # [3, ND]
            c = np.concatenate([np.zeros((3, 1)), np.cumsum(row, axis=1)], axis=1)
            sl = np.full((ND + 1, ND + 1), np.inf)
            for p in range(ND):
                for q in range(p + 1, ND + 1):
                    sl[p, q] = seg_loss(*(c[:, q] - c[:, p]))[0]
            best = np.full((ND + 1, ND + 1), np.inf)                                            # best[m][l]
            arg = np.zeros((ND + 1, ND + 1), dtype=int)
            best[0, 0] = 0.0
            for m in range(1, ND + 1):
                for l in range(m, ND + 1):
                    cand = best[m - 1, :l] + sl[:l, l]
                    arg[m, l] = int(np.argmin(cand))
                    best[m, l] = cand[arg[m, l]]
            res = {}
            for m in range(1, ND + 1):
                edges, l = [ND], ND
                for mm in range(m, 0, -1):
                    l = arg[mm, l]
                    edges.append(l)
                res[m] = (float(best[m, ND]), edges[::-1])
            inner[(u, t)] = res
    INF = float("inf")
    F = np.full((NK + 1, R_max + 1, S_max + 1), INF)
    back = {}
    F[0, 0, 0] = 0.0
    for t in range(1, NK + 1):
        for r in range(1, R_max + 1):
            for s in range(r, S_max + 1):
                for u in range(t):
                    for m in range(1, min(ND, s) + 1):
                        v = F[u, r - 1, s - m] + inner[(u, t)][m][0]
                        if v < F[t, r, s]:
                            F[t, r, s] = v
                            back[(t, r, s)] = (u, m)
    out = {}
    for S in range(2, S_max + 1):
        r_best = min(range(1, R_max + 1), key=lambda r: F[NK, r, S])
        if not np.isfinite(F[NK, r_best, S]):
            continue
        groups, t, r, s = [], NK, r_best, S
        while r > 0:
            u, m = back[(t, r, s)]
            groups.append((u, t, inner[(u, t)][m][1]))
            t, r, s = u, r - 1, s - m
        out[S] = (float(F[NK, r_best, S]), groups[::-1])
    return out


def describe(groups):
    return " | ".join(f"|k| {KE[u]:.3f}-{KE[t]:.3f}: |D| edges " + ",".join(f"{DE[e]:.3f}" for e in edges[1:-1]) if len(edges) > 2 else f"|k| {KE[u]:.3f}-{KE[t]:.3f}: 1 band"
                      for u, t, edges in groups)


def windows(groups, kern, kmag, w_k=0.10, w_d=0.02):
    """Smooth partition of unity: radial logistic edges (in ln|k|) x per-group |D| logistic edges."""
    lk = np.log(np.maximum(kmag, 1e-3))
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    kedges = [KE[u] for u, _, _ in groups] + [KE[groups[-1][1]]]
    S = [sig((lk - np.log(max(e, 1e-3))) / w_k) for e in kedges[1:-1]]
    radial = [1 - S[0]] + [S[i] - S[i + 1] for i in range(len(S) - 1)] + [S[-1]] if S else [np.ones_like(lk)]
    wins = []
    for g, (_, _, edges) in enumerate(groups):
        de = [DE[e] for e in edges[1:-1]]
        Sd = [sig((kern - e) / w_d) for e in de]
        bands = [1 - Sd[0]] + [Sd[i] - Sd[i + 1] for i in range(len(Sd) - 1)] + [Sd[-1]] if Sd else [np.ones_like(kern)]
        wins += [radial[g] * band for band in bands]
    return wins


def comps_hard(wins, d, mask):
    D = np.fft.fftn(d * mask)
    return [np.real(np.fft.ifftn(D * w)) * mask for w in wins]


def main():
    # ------------------------------------------------------------------ data (RC1)
    d1 = load_rc1("dataRC1")
    mask, chi, vox = d1["mask"].astype(np.float32), d1["chi"], d1["voxel_size"]
    b = clean_field(chi, vox, mask)
    vol = Volume.from_rc1(d1, "cpu")
    mk = mask > 0
    fields = {}
    for fam, snr in (("noise", 60.0), ("sti_noise", 100.0), ("offdiag_noise", 100.0)):
        dirty, _ = make_dirty(fam, vol, np.random.default_rng(3), torch.Generator().manual_seed(3), ArtifactConfig(snr_low=snr, snr_high=snr))
        fields[fam] = dirty.numpy() * mask
    tissue = mask * d1["phs_tissue"]
    kern, kmag = axes(b.shape, vox)
    nrmse = lambda f, t=b: 100 * float(np.linalg.norm((f - t)[mk]) / np.linalg.norm(t[mk]))
    say(f"phase NRMSE % of the dirty fields vs the clean target: " + ", ".join(f"{k} {nrmse(v):.1f}" for k, v in {**fields, 'tissue (held out)': tissue}.items()))

    stats = {"real": cell_stats(b * mask, fields["sti_noise"], kern, kmag) + cell_stats(b * mask, fields["offdiag_noise"], kern, kmag),
             "noise": cell_stats(b * mask, fields["noise"], kern, kmag)}
    r2 = load_rc2("dataRC2", 7.0)
    m2, v2 = r2["mask"].astype(np.float32), r2["voxel_size"]
    k2, km2 = axes(m2.shape, v2)
    stats["sim2"] = cell_stats(forward_field(r2["chi"], m2, v2), r2["field"] * m2, k2, km2)

    for name, st in stats.items():
        eb, ebd, ed = st
        floor = seg_loss(eb, ebd, ed)[0].sum()
        say(f"\n[{name}] total E_b {eb.sum():.4g}; per-cell passive floor residual = {100 * np.sqrt(floor / eb.sum()):.2f}% of ||b||; identity (dirty) = {100 * np.sqrt((eb - 2 * ebd + ed).sum() / eb.sum()):.2f}%")
        g = seg_loss(eb, ebd, ed)[1]
        say("passive gain per cell (rows |k| upper edge, cols |D| upper edge; '.' = no energy):")
        say(" " * 8 + " ".join(f"{e:5.2f}" for e in DE[1:]))
        for i in range(NK):
            say(f"{KE[i + 1]:6.3f}  " + " ".join(f"{g[i, j]:5.2f}" if eb[i, j] > 1e-6 * eb.sum() else "    ." for j in range(ND)))

    # ------------------------------------------------------------------ designs
    designs = {}
    for name in ("real", "noise", "sim2"):
        designs[name] = best_partition(stats[name])
        eb = stats[name][0].sum()
        say(f"\n[{name}] best passive partition per segment budget S (loss as % of ||b|| in the cell-filter proxy, no re-masking):")
        for S, (loss, groups) in designs[name].items():
            if S in (4, 6, 8, 10, 12, 14, 16, 20):
                say(f"  S={S:2d}: {100 * np.sqrt(loss / eb):6.2f}%   {describe(groups)}")
    json.dump({n: {str(S): {"loss": l, "groups": [[int(u), int(t), [int(e) for e in ed]] for u, t, ed in gr]} for S, (l, gr) in dd.items()} for n, dd in designs.items()},
              open(OUT / "designs.json", "w"))

    # ------------------------------------------------------------------ evaluation in the image domain (held-out measured tissue)
    say("\nIMAGE-DOMAIN EVALUATION (post-masking, soft windows, passive gains [0,1] fitted on STI+noise and off-diagonal+noise at SNR 100; tissue = held-out exp7).")
    say(f"phase NRMSE (%) vs the clean target (dirty: tissue {nrmse(tissue):.1f}, sti_noise {nrmse(fields['sti_noise']):.1f}, offdiag_noise {nrmse(fields['offdiag_noise']):.1f}, noise60 {nrmse(fields['noise']):.1f}):")
    say(f"{'design':26s}{'S':>4s}{'tissue(held out)':>18s}{'sti_noise':>11s}{'offdiag_noise':>15s}")
    fit_names = ("sti_noise", "offdiag_noise")


    def evaluate(label, comps_of):
        cf = [comps_of(fields[n]) for n in fit_names]
        A = np.concatenate([np.stack([c[mk] for c in cs], 1) for cs in cf]).astype(np.float64)
        y = np.concatenate([b[mk] for _ in fit_names]).astype(np.float64)
        g = lsq_linear(A, y, bounds=(0.0, 1.0)).x
        apply = lambda f: sum(gi * c for gi, c in zip(g, comps_of(f)))
        say(f"{label:26s}{len(g):4d}{nrmse(apply(tissue)):18.2f}{nrmse(apply(fields['sti_noise'])):11.2f}{nrmse(apply(fields['offdiag_noise'])):15.2f}")
        return g


    tr = DipoleletTransform(b.shape, vox, **{**SIM2_DIPOLELET, "coarse_bands": 3})
    evaluate("current Dipole-let (12)", lambda f: [p.numpy() * mask for p in image_components(tr, torch.from_numpy(np.ascontiguousarray(f, np.float32)))])
    for S in (6, 8, 10, 12, 16):
        if S in designs["real"]:
            ws = windows(designs["real"][S][1], kern, kmag)
            evaluate(f"derived (real) S={S}", lambda f, ws=ws: comps_hard(ws, f, mask))
    for S in (12,):
        if S in designs["noise"]:
            ws = windows(designs["noise"][S][1], kern, kmag)
            evaluate(f"derived (noise) S={S}", lambda f, ws=ws: comps_hard(ws, f, mask))
    (OUT / "design.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("wrote", OUT / "design.txt")


if __name__ == "__main__":
    main()
