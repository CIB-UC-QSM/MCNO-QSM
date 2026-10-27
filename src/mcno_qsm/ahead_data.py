"""S4c training source: RC1 patches mixed with patches of AHEAD heads (CLAUDE.md 9ab). Never reads RC2 (Sim2).

AHEAD heads come from the cache of `scripts/preprocess_ahead.py` (1.0 mm, B0 along z). They carry no real psi, so only the
object-agnostic families (noise, clean, spheres_chi, ...) are used on them. Each training step draws ONE source (RC1 with probability
1 - fraction, AHEAD with `fraction`) so a batch shares one voxel size. The phase scale is RC1's, i.e. the noise recipe relates the
weight channel to the phase noise exactly as in the earlier runs. Checkpoint selection stays on the RC1 check slab (comparable with
`noise_clean`); the AHEAD validation subjects are logged only.
"""

from pathlib import Path

import numpy as np
import torch

from .ahead import load_subject
from .artifacts import IDENTITY, Volume, apply_overrides, augment_object, make_dirty, resample_volume
from .metrics import masked_nrmse
from .physics import dipole_kernel
from .rc1_data import RC1Source


def split_subjects(names, val, spare, seed=0):
    """Deterministic split by subject: (train, validation, spare), each sorted."""
    if len(names) < val + spare + 1:
        raise ValueError(f"{len(names)} subjects cannot give {val} validation + {spare} spare + a training set")
    order = np.random.default_rng(seed).permutation(len(names))
    pick = lambda idx: sorted(names[i] for i in idx)
    return pick(order[val + spare:]), pick(order[:val]), pick(order[val:val + spare])


def volume_from_arrays(chi, weight, mask, voxel_size, phs_scale, device, margin=16) -> Volume:
    """A `Volume` of one AHEAD head, zero-padded to a cube of side max(shape) + margin so cube rotations stay inside the grid."""
    side = max(chi.shape) + margin
    lo = [(side - n) // 2 for n in chi.shape]

    def t(a, dtype=torch.float32):
        out = torch.zeros((side,) * 3, dtype=dtype)
        out[tuple(slice(l, l + n) for l, n in zip(lo, chi.shape))] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return out.to(device)

    chi_t, mask_t, weight_t = t(chi), t(mask), t(weight)
    kernel = dipole_kernel((side,) * 3, voxel_size, device=device)
    clean = torch.fft.ifftn(torch.fft.fftn(chi_t) * kernel).real
    none = torch.zeros(1, device=device)    # real-psi fields do not exist for AHEAD (those families are never sampled)
    return Volume(chi=chi_t, mask=mask_t, mask_cpu=mask_t.cpu().numpy() > 0, magnitude=weight_t * mask_t, clean=clean,
                  tissue=none, sti=none, offdiag=none, kernel=kernel, phs_scale=phs_scale, voxel_size=tuple(voxel_size))


class MixedSource(RC1Source):
    """RC1Source plus AHEAD training heads; `ahead` = {root, fraction, val_subjects, spare_subjects, split_seed, eval_subjects, crop}."""

    def __init__(self, root, device, cfg=None, holdout=None, min_mask=0.05, objects=None, ahead=None):
        super().__init__(root, device, cfg, holdout, min_mask, objects)
        ahead = dict(ahead or {})
        self.fraction = float(ahead.get("fraction", 0.5))
        self.crop = int(ahead.get("crop", 128))
        files = {p.stem: p for p in sorted(Path(ahead["root"]).glob("sub-*.npz"))}
        self.train_names, self.val_names, self.spare_names = split_subjects(
            list(files), int(ahead.get("val_subjects", 10)), int(ahead.get("spare_subjects", 10)), int(ahead.get("split_seed", 0)))
        if ahead.get("n_train"):   # paper recipe: train on a few heads only (seeded pick from the training split)
            pick = np.random.default_rng(int(ahead.get("split_seed", 0)) + 1).permutation(len(self.train_names))[: int(ahead["n_train"])]
            self.train_names = sorted(self.train_names[i] for i in pick)
        self.eval_names = self.val_names[: int(ahead.get("eval_subjects", 3))]
        self.files = files
        self.ahead_voxel = (1.0, 1.0, 1.0)
        if not self.agnostic:
            raise ValueError("AHEAD heads need at least one object-agnostic family with weight > 0")
        # float16 arrays of the training subjects stay in RAM (about 40 MB each); volumes are built on the GPU per patch.
        self.raw = {name: load_subject(files[name]) for name in self.train_names}
        self.ahead_meta = {"root": str(ahead["root"]), "fraction": self.fraction, "train": self.train_names,
                           "validation": self.val_names, "spare": self.spare_names, "eval_subjects": self.eval_names,
                           "split_seed": int(ahead.get("split_seed", 0)), "n_train": ahead.get("n_train")}

    def _volume(self, arrays):
        chi, weight, mask, voxel = arrays
        return volume_from_arrays(chi, weight, mask, voxel, self.volume.phs_scale, self.device)

    def _ahead_patch(self, size, rng, generator, voxel=None):
        name = self.train_names[int(rng.integers(len(self.train_names)))]
        volume = self._volume(self.raw[name])
        if voxel is not None:   # paper recipe: the head at another isotropic voxel size (smaller patch when the head no longer fills one)
            size = self.fit_size(volume.shape, volume.voxel_size, voxel, size)
            volume = resample_volume(volume, voxel, size)
        if rng.random() < self.objects.probability:
            volume = augment_object(volume, rng, self.objects) or volume
        family = self._pick_family(rng, self.agnostic)
        volume = self._augment_recipe(volume, family, rng)
        origin = self.window(size, rng, volume, anywhere=True)
        dirty, info = make_dirty(family, volume, rng, generator, self.cfg, tuple((o, o + size) for o in origin))
        return self._finish(apply_overrides(volume, info), dirty, origin, size, rng)

    def next_batch(self, size, rng, generator, n):
        if rng.random() >= self.fraction:
            return super().next_batch(size, rng, generator, n)
        voxel = self._draw_voxel(rng)
        batch = [self._ahead_patch(size, rng, generator, voxel) for _ in range(n)]
        return (*(torch.cat(items) for items in zip(*batch)), self.ahead_voxel if voxel is None else (voxel,) * 3)

    def _centered_crop(self, volume):
        """Corner of a `crop`^3 window centered on the brain (clamped to the volume)."""
        centroid = np.argwhere(volume.mask_cpu).mean(0)
        return [int(np.clip(round(c - self.crop / 2), 0, max(volume.shape[i] - self.crop, 0))) for i, c in enumerate(centroid)]

    @torch.no_grad()
    def ahead_scores(self, model, seed=12345):
        """Mean cleaned and input phase NRMSE (percent) on a crop of each AHEAD validation subject: noise family and identity."""
        rows = {f"ahead_{k}": [] for k in ("noise", "noise_in", "clean")}
        for name in self.eval_names:
            volume = self._volume(load_subject(self.files[name]))
            origin = self._centered_crop(volume)
            window = (slice(None), slice(None), *[slice(o, o + self.crop) for o in origin])
            mask = volume.mask[None, None][window]
            weight = volume.magnitude[None, None][window]
            target = volume.target[None, None][window]
            for family, key in (("noise", "noise"), (IDENTITY, "clean")):
                dirty, _ = make_dirty(family, volume, np.random.default_rng(seed), torch.Generator(device=self.device).manual_seed(seed),
                                      self.cfg)
                field = dirty[None, None][window]
                prediction = model(field, mask, weight, voxel_size=self.ahead_voxel)
                rows[f"ahead_{key}"].append(masked_nrmse(prediction, target, mask).item())
                if family == "noise":
                    rows["ahead_noise_in"].append(masked_nrmse(field * mask, target, mask).item())
        return {k: float(np.mean(v)) for k, v in rows.items() if v}
