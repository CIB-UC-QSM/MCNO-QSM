"""Write the configs of the paper-recipe replication (CLAUDE.md 9bo): the training RECIPE of "Deep Fourier-Space Inversion" (Lambert, Silva,
Milovic, Tejos; ISMRM 2023, abstract 4171) on our phase cleaner. We do not adopt their inversion; only data and augmentation change.

Reference = `noise_clean` (configs/s4b_noise_clean.toml: seed 42, 6000 steps, batch 2, 96^3 patches, 67k-parameter PhaseCleaner, same RC1 check
slab for the checkpoint selection, Sim2 monitored only). Run from the project root:
    python scripts/make_paper_configs.py
The existing s4b_*.toml files are not touched.

  paper_e1_rc1     E1  one brain (the RC1 head): noise 1.0 + clean 0.25, SNR 30-300, random chi scale, isotropic voxel sizes 1-3 mm,
                       axis permutation (the 24 cube rotations plus tilts of `objects`, already in the code)
  paper_e1_ahead1  E1  the same on ONE AHEAD subject instead of the RC1 head
  paper_e2_geo     E2  no brain: random spheres and cylinders in chi, 64^3, SNR 30-120, 15000 steps x batch 2 = 30,000 volumes
  paper_e3_ft      E3  fine-tune the E2 checkpoint on two AHEAD subjects with chi scale, para/dia part weights, sign inversion, SNR mixture
  paper_e4_l2only  E4  loss ablation: the paper's loss is MSE only; ours is masked L1 + L2 (here L2 only), RC1 data of noise_clean
  paper_ws         W   weight-scale augmentation (weight channel x 0.5..3, noise tied to it) on the noise_clean recipe
  paper_e1_rc1_ws  W   E1 with the weight-scale augmentation
"""

from pathlib import Path

TEMPLATE = """# {title}
# Paper-recipe replication (CLAUDE.md 9bo). Reference recipe: configs/s4b_noise_clean.toml. Sim2 (dataRC2) is monitored only.
# Dipolelet parameters are those of utils/Dipole-lets_Paper_Plots_3.ipynb (cell 19).

[data]
rc1 = "dataRC1"        # {rc1_role}
rc2 = "dataRC2"        # monitoring only
field_strength = 7.0   # T of Sim2 (Hz -> ppm)
{data_extra}
[data.holdout]
axis = 0               # check slab for checkpoint selection: train x in [0,100), gap 12, test [112,160) = 15% of the brain
train_fraction = 0.625
gap = 12

[artifacts]
snr_low = {snr_low}
snr_high = {snr_high}
outside = "clean"{artifact_extra}

[artifacts.weights]
noise = 1.0
clean = 0.25

[objects]
probability = 0.7      # rotated/rescaled heads; the 24 cube rotations move the z axis (B0 stays z) onto x or y
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
learning_rate = {lr}
weight_decay = 0.0
grad_clip = 1.0
seed = 42
device = "auto"
patch = {patch}
batch_size = 2
l1_weight = {l1}
l2_weight = 1.0
cone_weight = 0.0{training_extra}
log_every = 50
eval_every = 250
save_every = 250
output = "runs/s4b_{name}"
"""

VOXELS = "[1.0, 1.25, 1.5, 2.0, 2.5, 3.0]"
AHEAD = """
[data.ahead]
root = "dataAHEAD_1mm"  # output of scripts/preprocess_ahead.py
fraction = 1.0         # every batch from AHEAD heads
n_train = {n}            # number of training subjects (a seeded pick from the 85 of the training split)
val_subjects = 10
spare_subjects = 10
split_seed = 0
eval_subjects = 3      # validation subjects scored at every eval (logged only)
crop = 128
"""
GEOMETRIC = """
[data.geometric]       # brain-free training volumes (patch = the whole 64^3 volume)
mask = "ellipsoid"
magnitude = 0.264      # mean RC1 weight; the nets break when the weight channel leaves 0.13-0.38 (CLAUDE.md 9u)
magnitude_variation = 0.3
"""

RUNS = {
    "paper_e1_rc1": dict(
        title="E1: single-brain training (RC1 head only) with the paper's augmentations: chi scale, voxel size 1-3 mm, axis permutation, SNR 30-300",
        rc1_role="the one training head AND the check slab", snr=(30.0, 300.0),
        art=f"\nchi_scale = [0.5, 2.0]     # log-uniform factor on chi, field recomputed\nvoxel_sizes = {VOXELS}   # mm, one per batch; the head is resampled"),
    "paper_e1_ahead1": dict(
        title="E1 on one AHEAD subject (instead of the RC1 head): same augmentations as paper_e1_rc1",
        rc1_role="check slab, input scale and Sim2-independent selection only", snr=(30.0, 300.0), data=AHEAD.format(n=1),
        art=f"\nchi_scale = [0.5, 2.0]\nvoxel_sizes = {VOXELS}"),
    "paper_e2_geo": dict(
        title="E2: geometric figures only (no brain): random spheres and cylinders in chi, 64^3, SNR 30-120, 15000 steps x batch 2 = 30,000 volumes",
        rc1_role="check slab, input scale, phase scale only (no RC1 voxel reaches a training patch)", snr=(30.0, 120.0),
        data=GEOMETRIC, steps=15000, patch=64),
    "paper_e3_ft": dict(
        title="E3: fine-tune the E2 network on two AHEAD subjects with chi scale, para/dia part weights, sign inversion and the SNR mixture 30-50 (30%) / 90-150 (70%)",
        rc1_role="check slab, input scale only", snr=(30.0, 150.0), data=AHEAD.format(n=2), steps=3000, lr=0.0003,
        art="\nsnr_mix = [[30.0, 50.0, 0.3], [90.0, 150.0, 0.7]]\nchi_scale = [0.5, 2.0]\nchi_part_weight = [0.5, 2.0]   # paramagnetic and diamagnetic parts scaled independently\nchi_flip_prob = 0.5",
        train='\ninit_from = "runs/s4b_paper_e2_geo/best.pt"   # E2 checkpoint chosen on the RC1 check slab'),
    "paper_e4_l2only": dict(
        title="E4: loss ablation, masked L2 only (the paper uses MSE) on the noise_clean recipe and data",
        rc1_role="training head and check slab", snr=(40.0, 200.0), l1=0.0),
    "paper_ws": dict(
        title="W: weight-scale augmentation on the noise_clean recipe: weight channel x 0.5..3 (log-uniform), the noise follows it (about 1/weight)",
        rc1_role="training head and check slab", snr=(40.0, 200.0), art="\nweight_scale = [0.5, 3.0]"),
    "paper_e1_rc1_ws": dict(
        title="E1 (RC1 head) plus the weight-scale augmentation",
        rc1_role="the one training head AND the check slab", snr=(30.0, 300.0),
        art=f"\nchi_scale = [0.5, 2.0]\nvoxel_sizes = {VOXELS}\nweight_scale = [0.5, 3.0]"),
}


def render(name, spec):
    return TEMPLATE.format(
        title=spec["title"], rc1_role=spec["rc1_role"], data_extra=spec.get("data", ""), snr_low=spec["snr"][0], snr_high=spec["snr"][1],
        artifact_extra=spec.get("art", ""), steps=spec.get("steps", 6000), lr=spec.get("lr", 0.001), patch=spec.get("patch", 96),
        l1=spec.get("l1", 1.0), training_extra=spec.get("train", ""), name=name)


if __name__ == "__main__":
    for name, spec in RUNS.items():
        Path("configs", f"s4b_{name}.toml").write_text(render(name, spec), encoding="utf-8")
        print("wrote", f"configs/s4b_{name}.toml")
