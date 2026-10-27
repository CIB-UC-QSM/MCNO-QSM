"""RC1 training source: augmented patches of a freshly simulated dirty field. Never reads RC2 (Sim2)."""

import numpy as np
import torch

from .artifacts import (EXTRA_FAMILIES, FAMILIES, OBJECT_BOUND, ArtifactConfig, GeometricConfig, ObjectConfig, Volume, apply_overrides,
                        augment_object, augment_susceptibility, geometric_volume, make_dirty, resample_volume, scale_weight)
from .physics import dipole_kernel
from .rc1 import load_rc1


def _batch(x):
    return x[None, None]


class RC1Source:
    """Full-volume RC1 tensors plus random patches.

    Each patch comes from either the original head (real-psi families allowed; windows only in the training slab)
    or, with probability `objects.probability`, a rotated/rescaled head (`artifacts.augment_object`; only the
    object-agnostic families). `holdout` = {axis, train_fraction, gap} as in Sim2Source: patches come from the
    first `train_fraction` of the axis, `gap` slices are skipped, the rest is the check slab. With a hold-out the
    rotated heads are built from the head truncated to the training slab, so no held-out anatomy reaches them
    (the original-pose samples still contain the dipole field of the whole head, as in S2b).
    """

    def __init__(self, root, device, cfg: ArtifactConfig | None = None, holdout=None, min_mask=0.05,
                 objects: ObjectConfig | None = None):
        self.device, self.cfg, self.min_mask = device, cfg or ArtifactConfig(), min_mask
        self.objects = objects or ObjectConfig()
        data = load_rc1(root)
        self.volume = v = Volume.from_rc1(data, device)
        self.shape, self.voxel_size, self.phs_scale = v.shape, v.voxel_size, v.phs_scale
        self.holdout = holdout
        self.axis, train_end, test_start = 0, self.shape[0], 0
        if holdout:
            self.axis = int(holdout["axis"])
            n = self.shape[self.axis]
            train_end = int(round(holdout["train_fraction"] * n))
            test_start = train_end + int(holdout["gap"])
            if not 0 < train_end < test_start < n:
                raise ValueError(f"empty train or test slab: train [0,{train_end}) test [{test_start},{n})")
        self.train_range = (0, train_end if holdout else self.shape[0])
        self.test_range = (test_start, self.shape[self.axis]) if holdout else None
        self.mask, self.weight, self.target = _batch(v.mask), _batch(v.magnitude), _batch(v.target)
        train = self._region(self.train_range)
        # Input scale from the clean in-mask field of the training region (a constant of the data, not of a draw).
        self.scale = float(v.target[torch.from_numpy(v.mask_cpu & train).to(device)].std())
        self.regions = {"": self.mask}
        if holdout:
            self.regions = {"_train": self.mask * _batch(torch.from_numpy(train).to(device).float()),
                            "_test": self.mask * _batch(torch.from_numpy(self._region(self.test_range))
                                                        .to(device).float())}
            self.train_volume = Volume.from_rc1(self._truncated(data, train), device)
        else:
            self.train_volume = v
        self.families = [f for f in (*FAMILIES, *EXTRA_FAMILIES) if self.cfg.weights.get(f, 0) > 0]
        self.agnostic = [f for f in self.families if f not in OBJECT_BOUND]
        self._resampled = {}
        if self.cfg.voxel_sizes and not self.agnostic:
            raise ValueError("voxel_sizes needs at least one object-agnostic family with weight > 0")

    def _truncated(self, data, train):
        keep = np.asarray(train, bool)
        out = dict(data)
        for key in ("chi", "magnitude", "phs_tissue", "chi_13", "chi_23", "chi_33"):
            out[key] = data[key] * keep
        out["mask"] = data["mask"] & keep
        return out

    def _region(self, bounds):
        region = np.zeros(self.shape, bool)
        index = [slice(None)] * 3
        index[self.axis] = slice(*bounds)
        region[tuple(index)] = True
        return region

    def _pick_family(self, rng, pool):
        weights = np.array([self.cfg.weights[f] for f in pool], dtype=float)
        return str(rng.choice(pool, p=weights / weights.sum()))

    def fixed_dirty(self, family, seed=12345):
        """Whole-volume dirty field of one family from a fixed seed on the original head (monitoring).

        With a hold-out, localized artifacts (jumps, spheres, cylinders) are placed inside the check slab, so the
        check region always contains something to remove.
        """
        rng = np.random.default_rng(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        region = None
        if self.holdout:
            region = [(0, n) for n in self.shape]
            region[self.axis] = self.test_range
        dirty, info = make_dirty(family, self.volume, rng, generator, self.cfg, region)
        return _batch(dirty), info

    def window(self, size, rng, volume: Volume | None = None, anywhere=False):
        """Origin of a patch whose mask share exceeds `min_mask` (training slab only unless `anywhere`)."""
        volume = volume or self.volume
        limits = list(volume.shape)
        if not anywhere:
            limits[self.axis] = self.train_range[1]
        high = [n - size for n in limits]
        if min(high) < 0:
            raise ValueError(f"patch {size} does not fit the training region {limits}")
        for _ in range(30):
            origin = [int(rng.integers(0, h + 1)) for h in high]
            if volume.mask_cpu[tuple(slice(o, o + size) for o in origin)].mean() > self.min_mask:
                break
        return origin

    def _draw_voxel(self, rng):
        """Isotropic voxel size (mm) of the next batch, or None (native) unless `voxel_sizes` is set; all patches of a batch share it."""
        return float(rng.choice(self.cfg.voxel_sizes)) if self.cfg.voxel_sizes else None

    @staticmethod
    def fit_size(shape, native, voxel, size):
        """Patch side for a head of `shape` voxels of `native` mm resampled to `voxel` mm: `size`, or the largest multiple of 8 that fits
        (a 96^3 patch of a 3 mm head would be 95% zero padding; the models and the trainer's caches take any shape)."""
        fit = int(min(shape) * float(np.mean(native)) / voxel) // 8 * 8
        return min(size, max(fit, 8))

    def _at_voxel(self, voxel, size):
        """The training head resampled to `voxel` mm and padded to at least `size`^3 (cached)."""
        if (voxel, size) not in self._resampled:
            self._resampled[(voxel, size)] = resample_volume(self.train_volume, voxel, size)
        return self._resampled[(voxel, size)]

    def patch(self, size, rng, generator, family=None, voxel=None):
        """One augmented patch (field, mask, weight, target), each [1,1,P,P,P], and its info dict.

        `voxel` (mm): use the head resampled to that voxel size (object-agnostic families only; the patch shrinks to what fits the
        resampled head, see `fit_size`); None keeps the native grid.
        """
        # Real-psi families (and runs made only of them) have nothing to rotate: they always use the original head.
        eligible = bool(self.agnostic) if family is None else family not in OBJECT_BOUND
        if voxel is not None and not eligible:
            raise ValueError("a resampled voxel size needs an object-agnostic family")
        wants_object = eligible and rng.random() < self.objects.probability
        if voxel is not None:
            size = self.fit_size(self.train_volume.shape, self.train_volume.voxel_size, voxel, size)
            volume = self._at_voxel(voxel, size)
            if wants_object:
                volume = augment_object(volume, rng, self.objects) or volume
            augmented = True
        else:
            volume = augment_object(self.train_volume, rng, self.objects) if wants_object else None
            augmented = volume is not None
            if not augmented:
                volume = self.volume
        family = family or self._pick_family(rng, self.agnostic if augmented else self.families)
        volume = self._augment_recipe(volume, family, rng)
        origin = self.window(size, rng, volume, anywhere=augmented)
        region = tuple((o, o + size) for o in origin)
        dirty, info = make_dirty(family, volume, rng, generator, self.cfg, region)
        out = self._finish(apply_overrides(volume, info), dirty, origin, size, rng)
        info["origin"], info["object_augmented"] = origin, augmented
        return (*out, info)

    def _augment_recipe(self, volume, family, rng):
        """Paper recipe (default off, no rng draw then): chi scaling / part weights / sign flip, and the weight-channel scale.
        The measured real-psi fields (tissue, sti, offdiag) stay as they are."""
        if family in ("tissue", "sti", "offdiag"):
            return volume
        if family not in OBJECT_BOUND:
            volume = augment_susceptibility(volume, rng, self.cfg)
        return scale_weight(volume, rng, self.cfg)

    @staticmethod
    def _finish(volume, dirty, origin, size, rng):
        """Crop the window, then random flips and quarter turns in the plane (z stays the B0 axis)."""
        window = (slice(None), slice(None), *[slice(o, o + size) for o in origin])
        flips = [d for d in (2, 3, 4) if rng.random() < 0.5]
        quarter_turns = int(rng.integers(0, 4))
        out = []
        for t in (_batch(dirty), _batch(volume.mask), _batch(volume.magnitude), _batch(volume.target)):
            t = t[window]
            if flips:
                t = torch.flip(t, flips)
            out.append(torch.rot90(t, quarter_turns, (2, 3)))
        return out

    def next_batch(self, size, rng, generator, n):
        """`n` patches stacked into (field, mask, weight, target) and the voxel size they share."""
        voxel = self._draw_voxel(rng)
        batch = [self.patch(size, rng, generator, voxel=voxel)[:4] for _ in range(n)]
        return (*(torch.cat(items) for items in zip(*batch)), self.voxel_size if voxel is None else (voxel,) * 3)


class GeometricSource(RC1Source):
    """Brain-free training (paper experiment 2): every patch is a fresh volume of random spheres and cylinders in chi.

    RC1 is still loaded, but only for the monitoring/selection side (check slab, input scale, phase scale, native voxel size), which
    keeps the run comparable with the brain-trained ones; no RC1 voxel reaches a training patch. A patch is a whole `size`^3 volume.
    """

    def __init__(self, root, device, cfg=None, holdout=None, min_mask=0.05, objects=None, geometric=None):
        super().__init__(root, device, cfg, holdout, min_mask, objects)
        if not self.agnostic:
            raise ValueError("geometric training needs at least one object-agnostic family with weight > 0")
        self.geometric = GeometricConfig(**(geometric or {}))
        self._kernels = {}

    def _kernel(self, size, voxel_size):
        if (size, voxel_size) not in self._kernels:
            self._kernels[(size, voxel_size)] = dipole_kernel((size,) * 3, voxel_size, device=self.device)
        return self._kernels[(size, voxel_size)]

    def patch(self, size, rng, generator, family=None, voxel=None):
        voxel_size = self.voxel_size if voxel is None else (voxel,) * 3
        if family in OBJECT_BOUND:
            raise ValueError(f"{family!r} needs a head; geometric training uses object-agnostic families only")
        volume = geometric_volume(size, voxel_size, rng, self.geometric, self.phs_scale, self._kernel(size, voxel_size), self.device)
        family = family or self._pick_family(rng, self.agnostic)
        volume = self._augment_recipe(volume, family, rng)
        origin = [0, 0, 0]
        dirty, info = make_dirty(family, volume, rng, generator, self.cfg, tuple((0, size) for _ in range(3)))
        out = self._finish(apply_overrides(volume, info), dirty, origin, size, rng)
        info["origin"], info["object_augmented"] = origin, False
        return (*out, info)
