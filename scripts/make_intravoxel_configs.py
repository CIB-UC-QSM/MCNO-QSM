"""Write the configs of the intravoxel / acquisition-noise runs (CLAUDE.md 9bp). Run from the project root:
    python scripts/make_intravoxel_configs.py
Reference = `noise_clean` (configs/s4b_noise_clean.toml: seed 42, 6000 steps, batch 2, 96^3 patches, 67k-parameter PhaseCleaner, same RC1 check slab,
Sim2 monitored only). The existing s4b_*.toml files are not touched.

  iv_noisestat  the noise_clean recipe with a wider SNR sweep (15-400, log-uniform) and other noise statistics (colored, heavy-tailed, spatially
                varying) in the single-echo `noise` family; no sources. Separates the effect of the noise sweep from the effect of the sources.
  iv_full       noise_clean + the multi-echo `acq` family (same sweep and statistics) + the `intravoxel` family (strong sources simulated on a fine
                grid and block-averaged, 4 synthetic echoes of Sim2's echo train at 7 T): noise 1.0, clean 0.5, acq 0.5, intravoxel 0.5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_paper_configs import render  # noqa: E402

KINDS = '\nnoise_kinds = [["gaussian", 0.4], ["colored", 0.2], ["heavy", 0.2], ["hetero", 0.2]]   # iid Gaussian, correlated, Student-t, smooth gain map'

RUNS = {
    "iv_noisestat": dict(
        title="Noise sweep alone: noise_clean recipe with SNR 15-400 and four noise statistics (single-echo noise family, no sources)",
        rc1_role="training head and check slab", snr=(15.0, 400.0), art=KINDS, weights={"noise": 1.0, "clean": 0.25}),
    "iv_full": dict(
        title="noise_clean + multi-echo acquisition (acq) + intravoxel dephasing around strong sources, SNR 15-400 and four noise statistics",
        rc1_role="training head and check slab", snr=(15.0, 400.0), art=KINDS, weights={"noise": 1.0, "clean": 0.5, "acq": 0.5, "intravoxel": 0.5}),
}


def main():
    for name, spec in RUNS.items():
        text = render(name, spec)
        text = text.replace("noise = 1.0\nclean = 0.25\n", "".join(f"{k} = {v}\n" for k, v in spec["weights"].items()))
        text = text.replace("# Paper-recipe replication (CLAUDE.md 9bo).", "# Intravoxel / acquisition-noise runs (CLAUDE.md 9bp).")
        Path("configs", f"s4b_{name}.toml").write_text(text, encoding="utf-8")
        print("wrote", f"configs/s4b_{name}.toml")


if __name__ == "__main__":
    main()
