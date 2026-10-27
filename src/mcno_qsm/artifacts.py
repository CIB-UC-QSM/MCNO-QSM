"""Seeded artifact generators for RC1 training (notebook experiments 2-12), all in ppm.

Every family returns a dirty field for the whole volume; the target is always `mask * D*chi_cosmos`.
Placement, sign, size and SNR are random per sample (the notebook uses fixed positions). Placement uses a
numpy `rng`, noise a torch `generator`, so a (rng, generator) pair fixes a sample exactly.

Notebook quirks handled deliberately: `create_geometric_phantom` permutes the axes (it unpacks `z, y, x`
but applies `center[0]` to `x`); here centers and shapes are in array order (axis 0, 1, 2) and B0 is axis 2.
`exp9 = -pQ` alone carries no signal, so it is applied as `clean + (-pQ)`.
"""

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .acquisition import complex_noise, draw_noise_spec, estimate_ppm, per_ppm_per_s, weight_prime
from .intravoxel import box_echoes, source_chi_fine
from .noise import simulate_dirty
from .physics import dipole_kernel
from .rc1 import clean_field, sti_fields

FAMILIES = ("noise", "jumps", "cylinders", "spheres_phase", "tissue", "sti", "offdiag", "spheres_chi", "combo")
# Identity example (9x): dirty = clean, so the only way to lose loss is to leave the field alone. Not in FAMILIES (the nine
# artifact families whose per-family monitoring and `sel_all` stay as in the earlier runs); sampled only when its weight > 0.
IDENTITY = "clean"
# Calcification with signal dropout (CLAUDE.md 9al/9am): a strong susceptibility source ADDED to chi (so the target field contains it) whose
# surroundings (core) are corrupted in the dirty field: only a fraction of the true field survives and the rest is noise-like, as measured
# in the ~3 mm shell around the calcifications of Sim2. Not in FAMILIES (those nine and their monitoring stay as in earlier runs).
CALC = "calc_shell"
# The modelled real residuals (STI, off-diagonal) with thermal noise ADDED (CLAUDE.md 9ar): the in-vivo situation, both regimes at once. The measured
# `tissue` residual is deliberately NOT given a noisy twin: it stays the held-out test of the regime layer. Not in FAMILIES.
STI_NOISE = "sti_noise"
OFFDIAG_NOISE = "offdiag_noise"
REAL_NOISE = (STI_NOISE, OFFDIAG_NOISE)
# Multi-echo acquisition (CLAUDE.md 9bp): echoes of the clean field with complex noise (several noise statistics, wide SNR range) and the
# frequency fitted over the echoes; `intravoxel` adds strong sources simulated on a fine grid and block-averaged, so the dephasing, the
# missing field and the weight dip near a source come out of one forward model. Not in FAMILIES.
ACQ = "acq"
INTRAVOXEL = "intravoxel"
EXTRA_FAMILIES = (IDENTITY, CALC, *REAL_NOISE, ACQ, INTRAVOXEL)
# These use fields measured/derived on the original head pose, so they cannot follow an object rotation.
OBJECT_BOUND = ("tissue", "sti", "offdiag", *REAL_NOISE)

# Mixture weights (relative sampling probabilities). combo/spheres_chi/noise are the closest to Sim2's psi
# (S4a gate); cylinders and spheres_phase are 3-9x larger than Sim2's psi and kept rare.
DEFAULT_WEIGHTS = {"combo": 3.0, "spheres_chi": 2.0, "noise": 2.0, "jumps": 1.0, "tissue": 1.0, "sti": 1.0,
                   "offdiag": 1.0, "spheres_phase": 0.5, "cylinders": 0.5}


@dataclass
class ArtifactConfig:
    snr_low: float = 40.0                    # notebook uses 100 and 40; sampled log-uniform in between
    snr_high: float = 100.0
    magnitude_floor: float = 0.0             # 0 = notebook recipe (phase noise explodes where magnitude is ~0)
    outside: str = "clean"                   # field outside the mask: noise-free D*chi ("clean") or zero
    jump_count: tuple = (1, 6)               # single-voxel phase impulses
    jump_ppm: tuple = (1.0, 2.0)             # |value| drawn from these, random sign
    sphere_count: tuple = (1, 4)
    sphere_radius: tuple = (3.0, 5.0)        # voxels; notebook 4
    sphere_ppm: tuple = (0.2, 0.45)          # |amplitude|; notebook 0.33
    cylinder_count: tuple = (1, 3)
    cylinder_radius: tuple = (6.0, 10.0)     # voxels; notebook 8
    cylinder_height: tuple = (6.0, 10.0)     # voxels; notebook 8
    cylinder_ppm: tuple = (0.3, 0.7)         # |amplitude|; notebook 0.5
    calc_count: tuple = (1, 2)               # calcification-like sources per sample (calc_shell)
    calc_radius: tuple = (2.0, 3.5)          # voxels
    calc_ppm: tuple = (0.8, 2.0)             # |chi| of the source (RC1 chi spans -0.26..0.39; Sim2's calcification reaches -2.03)
    calc_positive_prob: float = 0.2          # share of sources with positive chi (hemorrhage-like)
    calc_extra: tuple = (2.0, 5.0)           # width of the corrupted shell beyond the source radius, voxels
    calc_keep: tuple = (0.1, 0.7)            # fraction of the true field that survives in the shell
    calc_noise: tuple = (0.3, 1.2)           # shell noise std relative to the rms of the true field there
    calc_dip: float = 0.5                    # relative drop of the weight (magnitude) inside the source + 1 voxel
    # Training recipe of the Deep Fourier-Space Inversion paper (CLAUDE.md 9bo); every key is off by default and draws nothing from the rng then.
    snr_mix: tuple | None = None             # ((low, high, probability), ...): SNR mixture instead of one log-uniform range, e.g. 30-50 (30%), 90-150 (70%)
    chi_scale: tuple | None = None           # (low, high): log-uniform factor on the whole chi, the field is recomputed
    chi_part_weight: tuple | None = None     # (low, high): independent log-uniform factors on the paramagnetic and the diamagnetic part of chi
    chi_flip_prob: float = 0.0               # probability of inverting the sign of chi
    weight_scale: tuple | None = None        # (low, high): log-uniform factor on the magnitude (weight channel); the noise follows it (about 1/magnitude)
    voxel_sizes: tuple | None = None         # isotropic voxel sizes in mm, one drawn per batch (the head is resampled; agnostic families only)
    # Noise statistics (9bp; default off, no rng draw then): ((kind, probability), ...) over gaussian / colored / heavy / hetero, used by the
    # families noise, acq and intravoxel.
    noise_kinds: tuple | None = None
    noise_colored: tuple = (0.5, 1.5)        # blur sigma (voxels) of the spatially correlated noise
    noise_heavy_df: tuple = (3, 6)           # Student-t degrees of freedom (integers)
    noise_hetero_depth: tuple = (0.3, 0.7)   # log-std of the smooth noise gain map
    # Multi-echo acquisition and intravoxel sources (9bp). Defaults: the echo train of Sim2's Magnitude.nii at its field strength.
    iv_echoes: tuple = (0.004, 0.012, 0.020, 0.028)   # s
    iv_b0: float = 7.0                       # T; sets the phase per ppm of the synthetic echoes
    iv_r2star: tuple = (25.0, 45.0)          # 1/s, echo-to-echo magnitude decay
    iv_factor: int = 4                       # fine-grid refinement of the source boxes
    iv_box: int = 32                         # side (coarse voxels) of the box simulated around a source
    iv_count: tuple = (1, 3)                 # sources per sample
    iv_radius: tuple = (1.0, 3.0)            # mean semi-axis, voxels
    iv_aspect: float = 0.4                   # log-uniform spread of the three semi-axes
    iv_ppm: tuple = (0.6, 2.5)               # |chi| of a source
    iv_positive_prob: float = 0.2
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class Volume:
    """RC1 tensors on one device. `clean` is the unmasked D*chi_cosmos; the target is `mask * clean`."""

    chi: Tensor
    mask: Tensor
    mask_cpu: np.ndarray
    magnitude: Tensor
    clean: Tensor
    tissue: Tensor
    sti: Tensor
    offdiag: Tensor
    kernel: Tensor
    phs_scale: float
    voxel_size: tuple

    @classmethod
    def from_rc1(cls, data: dict, device) -> "Volume":
        voxel = data["voxel_size"]
        sti, pq = sti_fields(data["chi_13"], data["chi_23"], data["chi_33"], voxel)

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)

        return cls(
            chi=t(data["chi"]), mask=t(data["mask"]), mask_cpu=np.asarray(data["mask"], bool),
            magnitude=t(data["magnitude"]), clean=t(clean_field(data["chi"], voxel)), tissue=t(data["phs_tissue"]),
            sti=t(sti), offdiag=t(-pq), kernel=dipole_kernel(data["chi"].shape, voxel, device=device),
            phs_scale=float(data["phs_scale"]), voxel_size=voxel)

    @property
    def shape(self):
        return tuple(self.chi.shape)

    @property
    def target(self) -> Tensor:
        return self.clean * self.mask


def rotation_matrix(angles_deg) -> np.ndarray:
    """Z-Y-X Euler rotation as in the notebook (angles: phi about z, theta about y, psi about x)."""
    phi, theta, psi = np.deg2rad(angles_deg)
    rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    rx = np.array([[1, 0, 0], [0, np.cos(psi), -np.sin(psi)], [0, np.sin(psi), np.cos(psi)]])
    return rz @ ry @ rx


def add_shape(volume: Tensor, center, amplitude: float, radius: float, height: float | None = None,
              angles_deg=(0.0, 0.0, 0.0)) -> None:
    """Add `amplitude` inside a ball (height None) or a cylinder (axis along local z, rotated) in place."""
    half = 0.0 if height is None else height / 2
    reach = int(np.ceil(np.hypot(radius, half))) + 1
    lows = [max(0, int(round(c)) - reach) for c in center]
    highs = [min(n, int(round(c)) + reach + 1) for c, n in zip(center, volume.shape)]
    if any(h <= l for l, h in zip(lows, highs)):
        return
    axes = [torch.arange(l, h, device=volume.device, dtype=torch.float32) - float(c)
            for l, h, c in zip(lows, highs, center)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
    rotation = torch.as_tensor(rotation_matrix(angles_deg).T, device=volume.device, dtype=torch.float32)
    local = torch.einsum("ij,jxyz->ixyz", rotation, grid)
    if height is None:
        inside = local.square().sum(0) <= radius**2
    else:
        inside = (local[0].square() + local[1].square() <= radius**2) & (local[2].abs() <= half)
    region = volume[lows[0]:highs[0], lows[1]:highs[1], lows[2]:highs[2]]
    region += amplitude * inside.to(volume.dtype)


def _uniform(rng, bounds):
    return float(rng.uniform(bounds[0], bounds[1]))


def _count(rng, bounds):
    return int(rng.integers(bounds[0], bounds[1] + 1))


def random_centers(mask_cpu: np.ndarray, rng, count: int, region=None) -> list:
    """`count` voxels inside the mask (and inside `region` = ((lo, hi), ...) when given)."""
    sub = mask_cpu
    offset = (0, 0, 0)
    if region is not None:
        slices = tuple(slice(lo, hi) for lo, hi in region)
        sub, offset = mask_cpu[slices], tuple(lo for lo, _ in region)
    candidates = np.argwhere(sub)
    if len(candidates) == 0:
        candidates = np.argwhere(np.ones_like(sub, bool))
    picks = candidates[rng.integers(0, len(candidates), size=count)]
    return [tuple(int(p + o) for p, o in zip(pick, offset)) for pick in picks]


def sign(rng) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def phase_jumps(v: Volume, rng, cfg: ArtifactConfig, region=None) -> Tensor:
    """psi_star: single-voxel impulses of +-1 or +-2 ppm added to the phase (notebook exp4)."""
    out = torch.zeros_like(v.clean)
    for center in random_centers(v.mask_cpu, rng, _count(rng, cfg.jump_count), region):
        out[center] += sign(rng) * float(rng.choice(cfg.jump_ppm))
    return out


def spheres(v: Volume, rng, cfg: ArtifactConfig, region=None) -> Tensor:
    out = torch.zeros_like(v.clean)
    for center in random_centers(v.mask_cpu, rng, _count(rng, cfg.sphere_count), region):
        add_shape(out, center, sign(rng) * _uniform(rng, cfg.sphere_ppm), _uniform(rng, cfg.sphere_radius))
    return out


def cylinders(v: Volume, rng, cfg: ArtifactConfig, region=None) -> Tensor:
    out = torch.zeros_like(v.clean)
    for center in random_centers(v.mask_cpu, rng, _count(rng, cfg.cylinder_count), region):
        add_shape(out, center, sign(rng) * _uniform(rng, cfg.cylinder_ppm), _uniform(rng, cfg.cylinder_radius),
                  _uniform(rng, cfg.cylinder_height), tuple(rng.uniform(0, 90, size=3)))
    return out


def apply_overrides(v: Volume, info: dict) -> Volume:
    """A family may change the clean field (an added source) and the magnitude of its own sample; crops and targets must follow."""
    changes = {k: info[k] for k in ("clean", "magnitude") if k in info}
    return replace(v, **changes) if changes else v


def dipole_field(v: Volume, chi: Tensor) -> Tensor:
    return torch.fft.ifftn(torch.fft.fftn(chi) * v.kernel).real


def noisy(v: Volume, field: Tensor, snr: float, cfg: ArtifactConfig, generator, spec=None) -> Tensor:
    """Notebook complex-noise recipe inside the mask; outside it the field is left noise-free. `spec` (draw_noise_spec): other statistics."""
    magnitude = v.magnitude.clamp_min(cfg.magnitude_floor) * v.mask
    noise = None if spec is None else complex_noise(field.shape, spec, generator, field.device, field.dtype)
    out = simulate_dirty(field, magnitude, snr, v.phs_scale, generator, noise)
    return torch.where(v.mask > 0, out, field)


def log_uniform(rng, bounds) -> float:
    return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def sample_snr(rng, cfg: ArtifactConfig) -> float:
    if cfg.snr_mix:   # mixture of ranges: pick one by probability, then log-uniform inside it
        probabilities = np.array([r[2] for r in cfg.snr_mix], dtype=float)
        low, high, _ = cfg.snr_mix[int(rng.choice(len(cfg.snr_mix), p=probabilities / probabilities.sum()))]
        return log_uniform(rng, (low, high))
    return float(np.exp(rng.uniform(np.log(cfg.snr_low), np.log(cfg.snr_high))))


def augment_susceptibility(v: Volume, rng, cfg: ArtifactConfig) -> Volume:
    """Paper recipe: random scale of chi, independent scales of its paramagnetic and diamagnetic parts, random sign inversion.

    chi' = flip * scale * (p * max(chi, 0) + q * min(chi, 0)); the field is recomputed from chi' (D is linear). Returns `v` itself
    (no rng draw) when no key is on. The magnitude and the real-psi fields are untouched, so use it with object-agnostic families only.
    """
    if not (cfg.chi_scale or cfg.chi_part_weight or cfg.chi_flip_prob > 0):
        return v
    scale = log_uniform(rng, cfg.chi_scale) if cfg.chi_scale else 1.0
    p, q = (log_uniform(rng, cfg.chi_part_weight), log_uniform(rng, cfg.chi_part_weight)) if cfg.chi_part_weight else (1.0, 1.0)
    flip = -1.0 if cfg.chi_flip_prob > 0 and rng.random() < cfg.chi_flip_prob else 1.0
    chi = flip * scale * (p * v.chi.clamp(min=0) + q * v.chi.clamp(max=0))
    return replace(v, chi=chi, clean=dipole_field(v, chi))


def scale_weight(v: Volume, rng, cfg: ArtifactConfig) -> Volume:
    """Random factor on the weight channel (the magnitude). Everything that draws noise from `v.magnitude` follows (noise ~ 1/magnitude)."""
    if not cfg.weight_scale:
        return v
    return replace(v, magnitude=v.magnitude * log_uniform(rng, cfg.weight_scale))


def calc_shell(v: Volume, rng, generator, cfg: ArtifactConfig, region=None):
    """Strong source(s) added to chi, noise everywhere, and a corrupted shell around each source (dropout).

    Returns (dirty, info) with the NEW clean field and magnitude in info['clean'] / info['magnitude']; the training target is
    mask * info['clean']. Inside the shell the dirty field is `keep * clean + noise` (soft edge), the magnitude dips inside the source.
    """
    sources, core, dip = (torch.zeros_like(v.clean) for _ in range(3))
    for center in random_centers(v.mask_cpu, rng, _count(rng, cfg.calc_count), region):
        radius = _uniform(rng, cfg.calc_radius)
        amplitude = _uniform(rng, cfg.calc_ppm) * (1.0 if rng.random() < cfg.calc_positive_prob else -1.0)
        add_shape(sources, center, amplitude, radius)
        add_shape(core, center, 1.0, radius + _uniform(rng, cfg.calc_extra))
        add_shape(dip, center, 1.0, radius + 1.0)
    soften = lambda t: F.avg_pool3d(t.clamp(max=1.0)[None, None], 3, 1, 1)[0, 0]
    clean = v.clean + dipole_field(v, sources)
    magnitude = v.magnitude * (1.0 - cfg.calc_dip * soften(dip))
    v2 = replace(v, clean=clean, magnitude=magnitude)
    snr = sample_snr(rng, cfg)
    # The notebook recipe returns the WRAPPED phase, which is wrong for fields beyond +-pi/phs_scale (about 0.16 ppm) such as the field
    # next to a strong source; the measured fields (Sim2 Frequency, RC1 phs_tissue) are unwrapped, so only the noise-induced phase
    # deviation is added to the unwrapped clean field.
    s = v.phs_scale
    wrapped_clean = torch.atan2(torch.sin(s * clean), torch.cos(s * clean)) / s
    dirty = clean + v.mask * (noisy(v2, clean, snr, cfg, generator) - wrapped_clean)
    shell = soften(core) * v.mask
    inside = (core.clamp(max=1.0) * v.mask) > 0.5
    rms = clean[inside].square().mean().sqrt() if bool(inside.any()) else clean.new_zeros(())
    keep, relative_noise = _uniform(rng, cfg.calc_keep), _uniform(rng, cfg.calc_noise)
    extra = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype) * (relative_noise * rms)
    dirty = dirty + shell * ((keep - 1.0) * clean + extra)
    return dirty, {"snr": snr, "keep": keep, "shell_noise": relative_noise, "clean": clean, "magnitude": magnitude}


def acquisition(v: Volume, rng, generator, cfg: ArtifactConfig, region=None, sources=True):
    """Multi-echo acquisition of the clean field, optionally with intravoxel dephasing around strong sources (CLAUDE.md 9bp).

    Echo e has magnitude `v.magnitude * exp(-R2* (TE_e - TE_0))` and phase `2 pi gamma B0 TE_e * field`; complex noise of std 1/SNR (the
    notebook's units) is added to every echo and the dirty field is the weighted least-squares frequency over the echoes. Each source is an
    ellipsoid of susceptibility drawn on a grid `iv_factor` times finer inside a box of `iv_box` coarse voxels; the box echoes are the block
    average of the fine signals, so the magnitude drops and the phase departs from the field of the block-averaged susceptibility.
    The new clean field (the training target) is the coarse dipole field of the block-averaged sources; the weight channel dips where the
    echoes dephase (sum m^2 TE / sum m TE of the dephased echoes over that of the undephased ones). Sources are processed in turn; a later
    box overwrites the echoes of an overlapping earlier one (those carry the earlier sources' coarse field, not their sub-voxel dephasing).
    """
    tes, b0 = cfg.iv_echoes, cfg.iv_b0
    p = per_ppm_per_s(b0)
    r2star = _uniform(rng, cfg.iv_r2star)
    decay = torch.tensor([np.exp(-r2star * (te - tes[0])) for te in tes], device=v.clean.device, dtype=v.clean.dtype)
    echo_magnitude = v.magnitude[None] * decay.reshape(-1, 1, 1, 1)
    chi_source = torch.zeros_like(v.clean)
    boxes = []
    if sources:
        side = min(cfg.iv_box, *v.shape)
        for center in random_centers(v.mask_cpu, rng, _count(rng, cfg.iv_count), region):
            low = [min(max(c - side // 2, 0), n - side) for c, n in zip(center, v.shape)]
            box = tuple(slice(l, l + side) for l in low)
            radii = np.minimum(_uniform(rng, cfg.iv_radius) * np.exp(rng.uniform(-cfg.iv_aspect, cfg.iv_aspect, 3)), side / 2 - 2)
            amplitude = _uniform(rng, cfg.iv_ppm) * (1.0 if rng.random() < cfg.iv_positive_prob else -1.0)
            fine = source_chi_fine((side,) * 3, cfg.iv_factor, [c - l for c, l in zip(center, low)], radii, amplitude,
                                   tuple(rng.uniform(0, 90, size=3))).to(v.clean.dtype).to(v.clean.device)
            background = v.clean + dipole_field(v, chi_source)
            echoes, coarse = box_echoes(background[box], echo_magnitude[(slice(None), *box)], fine, cfg.iv_factor, v.voxel_size, tes, p)
            chi_source[box] = chi_source[box] + coarse
            boxes.append((box, echoes))
    clean = v.clean + dipole_field(v, chi_source) if boxes else v.clean
    t = torch.tensor(tes, device=clean.device, dtype=clean.dtype).reshape(-1, 1, 1, 1)
    signal = echo_magnitude * torch.exp(1j * clean[None] * (t * p))
    dip = torch.ones_like(clean)
    for box, echoes in boxes:
        signal[(slice(None), *box)] = echoes
        base = weight_prime(echo_magnitude[(slice(None), *box)], tes)
        dip[box] = torch.where(base > 0, weight_prime(echoes.abs(), tes) / base.clamp_min(1e-30), torch.ones_like(base)).clamp(0, 1)
    snr, spec = sample_snr(rng, cfg), draw_noise_spec(rng, cfg)
    noise = complex_noise(tuple(signal.shape), spec, generator, clean.device, clean.dtype)
    dirty = torch.where(v.mask > 0, estimate_ppm(signal + noise / snr, tes, b0), clean)
    info = {"snr": snr, "r2star": r2star, "sources": len(boxes), "clean": clean, "magnitude": v.magnitude * dip, "chi_source": chi_source}
    if spec:
        info["noise"] = spec["kind"]
    return dirty, info


def make_dirty(family: str, v: Volume, rng, generator, cfg: ArtifactConfig | None = None, region=None):
    """Dirty field (whole volume, ppm) of one family and a small info dict (snr, ...)."""
    cfg = cfg or ArtifactConfig()
    info = {"family": family}
    if family == "noise":
        info["snr"] = sample_snr(rng, cfg)
        spec = draw_noise_spec(rng, cfg)
        dirty = noisy(v, v.clean, info["snr"], cfg, generator, spec)
        if spec:
            info["noise"] = spec["kind"]
    elif family == "jumps":
        dirty = v.clean + phase_jumps(v, rng, cfg, region)
    elif family == "cylinders":
        dirty = v.clean + cylinders(v, rng, cfg, region)
    elif family == "spheres_phase":
        dirty = v.clean + spheres(v, rng, cfg, region)
    elif family == "tissue":
        dirty = v.tissue
    elif family == "sti":
        dirty = v.sti
    elif family == "offdiag":
        dirty = v.clean + v.offdiag
    elif family == IDENTITY:
        dirty = v.clean.clone()
    elif family in REAL_NOISE:
        info["snr"] = sample_snr(rng, cfg)
        base = v.sti if family == STI_NOISE else v.clean + v.offdiag
        dirty = noisy(v, base, info["snr"], cfg, generator)
    elif family == CALC:
        dirty, extra = calc_shell(v, rng, generator, cfg, region)
        info.update(extra)
    elif family in (ACQ, INTRAVOXEL):
        dirty, extra = acquisition(v, rng, generator, cfg, region, sources=family == INTRAVOXEL)
        info.update(extra)
    elif family in ("spheres_chi", "combo"):
        info["snr"] = sample_snr(rng, cfg)
        field_chi = v.clean + dipole_field(v, spheres(v, rng, cfg, region))
        dirty = noisy(v, field_chi, info["snr"], cfg, generator)
        if family == "combo":
            dirty = dirty + phase_jumps(v, rng, cfg, region)
    else:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES + EXTRA_FAMILIES}")
    if cfg.outside == "zero":
        dirty = dirty * v.mask
    return dirty, info


@dataclass
class ObjectConfig:
    """Anatomy augmentation: rotate/rescale the head (chi, mask, magnitude) with B0 fixed, then redo D*chi.

    Physically valid (a head in another pose), unlike rotating the field. Voxels are treated as isotropic
    (RC1's 1.0625/1.0714 mm differ by under 1%).
    """

    probability: float = 0.7
    cube_rotations: bool = True              # random one of the 24 proper cube rotations (z can move to x or y)
    max_angle_deg: float = 20.0              # extra random Euler tilt about each axis
    scale: tuple = (0.9, 1.1)
    max_lost_mask: float = 0.01              # re-draw when more than this share of the brain leaves the grid
    tries: int = 6


def cube_rotation_matrices() -> list:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = np.zeros((3, 3))
            for row, (column, sgn) in enumerate(zip(perm, signs)):
                matrix[row, column] = sgn
            if np.linalg.det(matrix) > 0:
                out.append(matrix)
    return out


CUBE_ROTATIONS = cube_rotation_matrices()


def warp(volume: Tensor, rotation: np.ndarray, scale: float) -> Tensor:
    """Resample [X,Y,Z] with output(x) = input(R x / scale) about the grid center (bilinear, zeros outside)."""
    theta = torch.zeros(1, 3, 4, device=volume.device)
    theta[0, :3, :3] = torch.as_tensor(rotation / scale, dtype=torch.float32)
    grid = F.affine_grid(theta, (1, 1, *volume.shape), align_corners=False)
    return F.grid_sample(volume[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]


def _pad_cube(x: Tensor, side: int) -> Tensor:
    """Zero-pad a 3D tensor, centered, to side^3."""
    lo = [(side - n) // 2 for n in x.shape]
    pad = [p for l, n in reversed(list(zip(lo, x.shape))) for p in (l, side - n - l)]
    return F.pad(x, pad)


def resample_volume(v: Volume, voxel_mm: float, min_side: int = 0) -> Volume:
    """The same head at an isotropic voxel size (area average when coarser, trilinear when finer), zero-padded to a cube of side
    >= `min_side` so a training patch fits. chi and magnitude are averaged over the mask only (no dilution at the edge); the clean
    field is recomputed from the resampled chi with the kernel of the new grid. Real-psi fields do not exist at the new resolution
    (placeholders), so only object-agnostic families may be drawn from the result.
    """
    ratio = float(np.mean(v.voxel_size)) / voxel_mm
    shape = tuple(max(2, int(round(n * ratio))) for n in v.shape)

    def resample(x):
        x = x[None, None]
        x = F.interpolate(x, size=shape, mode="area") if ratio < 1 else F.interpolate(x, size=shape, mode="trilinear", align_corners=False)
        return x[0, 0]

    inside = resample(v.mask)
    mask = (inside > 0.5).to(v.mask.dtype)
    weight = inside.clamp_min(1e-3)
    side = max(max(shape), min_side)
    chi, magnitude, mask = (_pad_cube(t, side) for t in (resample(v.chi * v.mask) / weight * mask,
                                                         resample(v.magnitude * v.mask) / weight * mask, mask))
    kernel = dipole_kernel((side,) * 3, (voxel_mm,) * 3, device=mask.device)
    none = torch.zeros(1, device=mask.device)
    return Volume(chi=chi, mask=mask, mask_cpu=mask.cpu().numpy() > 0, magnitude=magnitude,
                  clean=torch.fft.ifftn(torch.fft.fftn(chi) * kernel).real, tissue=none, sti=none, offdiag=none, kernel=kernel,
                  phs_scale=v.phs_scale, voxel_size=(voxel_mm,) * 3)


@dataclass
class GeometricConfig:
    """Brain-free training volumes of the paper's second experiment: random spheres and cylinders in chi inside an ellipsoidal mask."""

    spheres: tuple = (3, 12)
    cylinders: tuple = (2, 8)
    sphere_radius: tuple = (3.0, 10.0)       # voxels
    cylinder_radius: tuple = (2.0, 8.0)      # voxels
    cylinder_height: tuple = (8.0, 32.0)     # voxels, random orientation
    ppm: tuple = (0.02, 0.12)                # |chi| of a figure, random sign; the clean-field rms in the mask is then 0.0086 (median of 60 volumes)
                                             # against 0.0089 for RC1's training region, so the input scale of the nets is not changed
    mask: str = "ellipsoid"                  # "ellipsoid" (random, 25-50% of the cube; figures centred inside) or "ones"
    ellipsoid_axes: tuple = (0.38, 0.5)      # semi-axes as a fraction of the side
    magnitude: float = 0.264                 # mean of RC1's magnitude in the mask (the weight channel the nets were trained on)
    magnitude_variation: float = 0.3         # log-std of a smooth random modulation of the magnitude (0 = constant)


def geometric_chi(size: int, rng, cfg: GeometricConfig, mask_cpu: np.ndarray, device) -> Tensor:
    """Random spheres and cylinders (additive, random sign) with centres inside the mask."""
    chi = torch.zeros((size,) * 3, device=device)
    for center in random_centers(mask_cpu, rng, _count(rng, cfg.spheres)):
        add_shape(chi, center, sign(rng) * _uniform(rng, cfg.ppm), _uniform(rng, cfg.sphere_radius))
    for center in random_centers(mask_cpu, rng, _count(rng, cfg.cylinders)):
        add_shape(chi, center, sign(rng) * _uniform(rng, cfg.ppm), _uniform(rng, cfg.cylinder_radius),
                  _uniform(rng, cfg.cylinder_height), tuple(rng.uniform(0, 360, size=3)))
    return chi


def geometric_volume(size: int, voxel_size, rng, cfg: GeometricConfig, phs_scale: float, kernel: Tensor, device) -> Volume:
    """One brain-free `Volume` (size^3): geometric chi, its masked dipole field, a smooth weight channel. `kernel` is the cached D."""
    if cfg.mask == "ones":
        mask_cpu = np.ones((size,) * 3, bool)
    else:
        axes = size * rng.uniform(*cfg.ellipsoid_axes, size=3)
        center = size / 2 + rng.uniform(-2.0, 2.0, size=3)
        grid = np.ogrid[tuple(slice(0, size) for _ in range(3))]
        mask_cpu = sum(((g - c) / a) ** 2 for g, c, a in zip(grid, center, axes)) <= 1.0
    mask = torch.from_numpy(mask_cpu.astype(np.float32)).to(device)
    chi = geometric_chi(size, rng, cfg, mask_cpu, device)
    modulation = torch.ones((size,) * 3, device=device)
    if cfg.magnitude_variation > 0:
        coarse = torch.from_numpy(rng.standard_normal((1, 1, 4, 4, 4)).astype(np.float32) * cfg.magnitude_variation).to(device)
        modulation = F.interpolate(coarse, size=(size,) * 3, mode="trilinear", align_corners=False)[0, 0].exp()
    none = torch.zeros(1, device=device)
    clean = torch.fft.ifftn(torch.fft.fftn(chi) * kernel).real
    return Volume(chi=chi, mask=mask, mask_cpu=mask_cpu, magnitude=cfg.magnitude * modulation * mask, clean=clean, tissue=none, sti=none,
                  offdiag=none, kernel=kernel, phs_scale=phs_scale, voxel_size=tuple(voxel_size))


def augment_object(v: Volume, rng, cfg: ObjectConfig) -> Volume | None:
    """A rotated/rescaled copy of the head (None if no draw keeps the brain inside the grid)."""
    original = float(v.mask.sum())
    for _ in range(cfg.tries):
        rotation = rotation_matrix(rng.uniform(-cfg.max_angle_deg, cfg.max_angle_deg, 3))
        if cfg.cube_rotations:
            rotation = rotation @ CUBE_ROTATIONS[int(rng.integers(len(CUBE_ROTATIONS)))]
        scale = float(rng.uniform(*cfg.scale))
        mask = (warp(v.mask, rotation, scale) > 0.5).to(v.mask.dtype)
        if 1 - float(mask.sum()) / (original * scale**3) > cfg.max_lost_mask:
            continue
        chi = warp(v.chi, rotation, scale) * mask
        return replace(v, chi=chi, mask=mask, mask_cpu=mask.cpu().numpy() > 0,
                       magnitude=warp(v.magnitude, rotation, scale) * mask, clean=dipole_field(v, chi))
    return None
