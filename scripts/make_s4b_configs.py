"""Write the S4b configs: one per artifact family alone, the family mix, and exp12 (combo) alone.

Same seed, steps, optimizer and hold-out for every run (controlled comparison). Run from the project root:
    python scripts/make_s4b_configs.py
"""

from pathlib import Path

from mcno_qsm.artifacts import CALC, DEFAULT_WEIGHTS, FAMILIES, IDENTITY

RUNS = {
    "noise": {"noise": 1.0},                           # exp2/3
    "jumps": {"jumps": 1.0},                           # exp4
    "cylinders": {"cylinders": 1.0},                   # exp5
    "spheres_phase": {"spheres_phase": 1.0},           # exp6
    "tissue": {"tissue": 1.0},                         # exp7
    "sti_offdiag": {"sti": 1.0, "offdiag": 1.0},       # exp8 + exp9
    "spheres_chi": {"spheres_chi": 1.0},               # exp11
    "combo": {"combo": 1.0},                           # exp12 alone
    "mix": dict(DEFAULT_WEIGHTS),                      # all families, weighted
}
SEEDS = {"mix_seed43": 43, "mix_seed44": 44}           # repeats of the mix with other seeds (init, patches, noise)
for _name in SEEDS:
    RUNS[_name] = dict(DEFAULT_WEIGHTS)

# Added 2026-10-21 (CLAUDE.md 9m): repeats of the best single family, and two mixture ablations. Everything else equal to S4b.
for _seed in (43, 44):
    SEEDS[f"noise_seed{_seed}"] = _seed
    RUNS[f"noise_seed{_seed}"] = {"noise": 1.0}
RUNS["mix_ns"] = {"noise": 1.0, "spheres_chi": 1.0}    # the two best families only, same 6000 steps
RUNS["mix_12k"] = dict(DEFAULT_WEIGHTS)                # default mix, twice the steps (noise then sees 2x the examples)
STEPS = {"mix_12k": 12000}

# Option 0 (CLAUDE.md 9o-bis): band-normalized spectral loss on top of the usual one; same data, seed and steps as `noise` / `mix`.
RUNS["noise_band"] = {"noise": 1.0}
RUNS["mix_band"] = dict(DEFAULT_WEIGHTS)
BAND_WEIGHT = {"noise_band": 1.0, "mix_band": 1.0}

# Identity example + wider noise range (CLAUDE.md 9x): same recipe as `noise`, plus 20% clean (dirty = clean) examples, SNR 40-200.
RUNS["noise_clean"] = {"noise": 1.0, "clean": 0.25}
SNR_HIGH = {"noise_clean": 200.0}

# Calcification family (CLAUDE.md 9am): noise_clean recipe + 'calc_shell' examples (a strong source with a corrupted shell around it).
RUNS["noise_clean_calc"] = {"noise": 1.0, "clean": 0.25, "calc_shell": 0.5}
SNR_HIGH["noise_clean_calc"] = 200.0

TEMPLATE = """# S4b: train on RC1 + artifacts ({name}); Sim2 (dataRC2) is monitored only, never trained on or used for selection.
# Dipolelet parameters are those of utils/Dipole-lets_Paper_Plots_3.ipynb (cell 19).

[data]
rc1 = "dataRC1"
rc2 = "dataRC2"        # monitoring only
field_strength = 7.0   # T of Sim2 (Hz -> ppm)

[data.holdout]
axis = 0               # check slab for checkpoint selection: train x in [0,100), gap 12, test [112,160) = 15% of the brain
train_fraction = 0.625
gap = 12

[artifacts]
snr_low = 40.0
snr_high = {snr_high}
outside = "clean"

[artifacts.weights]
{weights}

[objects]
probability = 0.7      # rotated/rescaled heads (real-psi families are never used on them)
max_angle_deg = 20.0
scale = [0.9, 1.1]

[dipolelet]
scales = 3
scaling_law = "static"
base_angular_bands = 3
delta = 0.05
window = "gaussian"
gaussian_width_factor = 0.5
starlet = "gaussian"
starlet_sigma = 0.5

[model]
channels = 16
blocks = 4
dilations = [1, 2, 4]
k_conditioning = true
checkpoint = true

[training]
steps = {steps}
learning_rate = 0.001
weight_decay = 0.0
grad_clip = 1.0
seed = {seed}
device = "auto"
patch = 96
batch_size = 2
l1_weight = 1.0
l2_weight = 1.0
cone_weight = 0.0{band_line}
log_every = 50
eval_every = 250
save_every = 250
output = "runs/s4b_{name}"
"""

if __name__ == "__main__":
    for name, weights in RUNS.items():
        assert set(weights) <= set(FAMILIES) | {IDENTITY, CALC}
        band_line = f"\nband_weight = {BAND_WEIGHT[name]}" if name in BAND_WEIGHT else ""
        text = TEMPLATE.format(name=name, seed=SEEDS.get(name, 42), steps=STEPS.get(name, 6000), band_line=band_line, snr_high=SNR_HIGH.get(name, 100.0),
                               weights="\n".join(f"{f} = {w}" for f, w in weights.items()))
        Path("configs", f"s4b_{name}.toml").write_text(text, encoding="utf-8")
        print("wrote", f"configs/s4b_{name}.toml")
