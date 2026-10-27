"""Phase cleaner: residual spectral blocks whose Fourier branch is split by dipolelet segments."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .dipolelets import SIM2_DIPOLELET, DipoleletTransform, PlainTransform, gaussian_smooth
from .physics import dipole_kernel


def k_features(shape, voxel_size, b0=(0.0, 0.0, 1.0), *, device=None, dtype=torch.float32) -> Tensor:
    """[5,X,Y,Z] physical-frequency features: D(k), 2|k| (cycles/mm) and the unit direction of k."""
    kernel = dipole_kernel(shape, voxel_size, b0, device=device, dtype=dtype)
    freqs = [torch.fft.fftfreq(n, d=v, device=device, dtype=dtype) for n, v in zip(shape, voxel_size)]
    grid = torch.meshgrid(*freqs, indexing="ij")
    radius = sum(k.square() for k in grid).sqrt()
    unit = [k / radius.clamp_min(1e-12) for k in grid]
    return torch.stack([kernel, 2 * radius, *unit])


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose mean and variance are computed over the voxels INSIDE the mask only (per sample and group), so the statistics do not depend on how
    much background a patch or a full volume contains (a 96^3 RC1 patch is about 58 % mask, the Sim2 volume 17 %). Per-channel affine, identity at init."""

    def __init__(self, channels: int, groups: int = 4, eps: float = 1e-5):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.groups, self.eps = groups, eps
        self.weight, self.bias = nn.Parameter(torch.ones(channels)), nn.Parameter(torch.zeros(channels))

    def forward(self, v: Tensor, mask: Tensor) -> Tensor:
        b, c = v.shape[:2]
        g = v.reshape(b, self.groups, c // self.groups, *v.shape[2:])
        m = mask.reshape(b, 1, 1, *mask.shape[2:]).to(v.dtype)
        count = (m.sum(dim=(2, 3, 4, 5), keepdim=True) * (c // self.groups)).clamp_min(1.0)
        mean = (g * m).sum(dim=(2, 3, 4, 5), keepdim=True) / count
        var = (((g - mean) ** 2) * m).sum(dim=(2, 3, 4, 5), keepdim=True) / count
        out = ((g - mean) * torch.rsqrt(var + self.eps)).reshape(v.shape)
        return out * self.weight.view(1, c, 1, 1, 1) + self.bias.view(1, c, 1, 1, 1)


class DipoleletBlock(nn.Module):
    """v <- v + GELU(W u + Re iFFT( sum_s g_s(k) R_s FFT(segment_s(u)) ) [+ L u]), u = v, or u = MaskedGroupNorm(v) (pre-norm residual, CLAUDE.md 9bd).

    L is the optional local refinement (option 2): a zero-initialized dilated 3x3x3 conv, so the block is
    unchanged at init. `local_dilation` None or 0 = off (no parameters, old state dicts load).
    """

    def __init__(self, channels: int, segments: int, k_conditioning: bool = True, hidden: int = 16,
                 local_dilation: int | None = None, norm_groups: int | None = None):
        super().__init__()
        self.norm = MaskedGroupNorm(channels, norm_groups) if norm_groups else None
        self.skip = nn.Conv3d(channels, channels, 1)
        self.local = None
        if local_dilation:
            d = int(local_dilation)
            self.local = nn.Conv3d(channels, channels, 3, padding=d, dilation=d)
            nn.init.zeros_(self.local.weight)
            nn.init.zeros_(self.local.bias)
        std = 0.1 / channels**0.5
        self.real = nn.Parameter(torch.randn(segments, channels, channels) * std)
        self.imag = nn.Parameter(torch.randn(segments, channels, channels) * std)
        self.gain = None
        if k_conditioning:
            self.gain = nn.Sequential(nn.Conv3d(5, hidden, 1), nn.GELU(), nn.Conv3d(hidden, segments, 1))
            nn.init.zeros_(self.gain[-1].weight)
            nn.init.zeros_(self.gain[-1].bias)

    @staticmethod
    def _segment_term(spectrum: Tensor, window, weight: Tensor, gain) -> Tensor:
        segment = spectrum if window is None else spectrum * window
        term = torch.einsum("oc,bcxyz->boxyz", weight, segment)
        return term if gain is None else term * gain

    def forward(self, v: Tensor, transform: DipoleletTransform, features: Tensor,
                segment_checkpoint: bool = False, mask: Tensor | None = None) -> Tensor:
        if self.norm is not None and mask is None:
            raise ValueError("a block with normalization needs the mask")
        u = v if self.norm is None else self.norm(v, mask)
        weights = torch.complex(self.real, self.imag)
        gains = None if self.gain is None else 1 + self.gain(features)
        total = None
        for s, (spectrum, window) in enumerate(transform.analysis_parts(u)):
            gain = None if gains is None else gains[:, s:s + 1]
            if segment_checkpoint:
                term = checkpoint(self._segment_term, spectrum, window, weights[s], gain, use_reentrant=False)
            else:
                term = self._segment_term(spectrum, window, weights[s], gain)
            total = term if total is None else total + term
        spectral = torch.fft.ifftn(total, dim=(-3, -2, -1), norm="ortho").real
        pre = self.skip(u) + spectral
        if self.local is not None:
            pre = pre + self.local(u)
        return v + F.gelu(pre)


class UNetBranch(nn.Module):
    """Small spatial 3D U-Net (option 1): full resolution, /2 and /4, base channels doubling per level.

    A HYBRID component (local convolutions with down/up-sampling, not Fourier); it runs in parallel with
    the Dipolelet blocks and its output is added to their result. The output conv is zero-initialized, so
    the branch is silent at init. Any input shape works (ceil-mode pooling, upsampling to the skip size).
    """

    def __init__(self, channels: int, base: int = 8):
        super().__init__()
        c0, c1, c2 = base, 2 * base, 4 * base
        self.enc0 = nn.Conv3d(channels, c0, 3, padding=1)
        self.enc1 = nn.Conv3d(c0, c1, 3, padding=1)
        self.enc2 = nn.Conv3d(c1, c2, 3, padding=1)
        self.mid = nn.Conv3d(c2, c2, 3, padding=1)
        self.dec1 = nn.Conv3d(c2 + c1, c1, 3, padding=1)
        self.dec0 = nn.Conv3d(c1 + c0, c0, 3, padding=1)
        self.out = nn.Conv3d(c0, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @staticmethod
    def _up(x: Tensor, like: Tensor) -> Tensor:
        return F.interpolate(x, size=like.shape[-3:], mode="trilinear", align_corners=False)

    def forward(self, v: Tensor) -> Tensor:
        e0 = F.gelu(self.enc0(v))
        e1 = F.gelu(self.enc1(F.avg_pool3d(e0, 2, ceil_mode=True)))
        e2 = F.gelu(self.enc2(F.avg_pool3d(e1, 2, ceil_mode=True)))
        m = F.gelu(self.mid(e2))
        d1 = F.gelu(self.dec1(torch.cat([self._up(m, e1), e1], dim=1)))
        d0 = F.gelu(self.dec0(torch.cat([self._up(d1, e0), e0], dim=1)))
        return self.out(d0)


class RegimeFilter(nn.Module):
    """Regime-gated linear Dipole-let filter (CLAUDE.md 9ar): two gains per segment blended by a learned local regime map.

    Real in-vivo-like residuals sit in the near-cone bands at EVERY scale (angular separability) while noise separates by scale (radial),
    and the two want opposite gains for the cone segments (9aq). The regime is read from the input alone: per radial group (each detail
    scale, and the coarse remainder when it is split into |D| bands) the local cone excess phi_g = log( G*E_cone / G*E_other ) - log(mass ratio of
    white noise), i.e. 0 for white noise and strongly positive when the cone carries more energy than its k-space share. r = sigmoid(w.phi + c)
    in [0,1]; the gain of segment s is g_s = g0_s + (g1_s - g0_s) r and the filter REMOVES sum_s (1 - g_s) comp_s from the input
    (comp_s = image component of segment s, they sum to the input). g0 = g1 = 1 is the identity.
    """

    def __init__(self, names, rho=6.0, g0=None, g1=None, w=None, c=0.0):
        super().__init__()
        groups = {}
        for index, name in enumerate(names):
            key, _, band = name.partition("/")
            if band.startswith("band"):
                groups.setdefault(key, []).append(index)
        self.groups = [(idx[0], idx[1:]) for idx in groups.values() if len(idx) >= 2]
        segments = len(names)
        init = lambda value, default: torch.full((segments,), default) if value is None else torch.as_tensor(value, dtype=torch.float32)
        self.g0, self.g1 = nn.Parameter(init(g0, 1.0)), nn.Parameter(init(g1, 1.0))
        if self.g0.shape != (segments,) or self.g1.shape != (segments,):
            raise ValueError("g0 and g1 need one gain per segment")
        w0 = torch.zeros(len(self.groups)) if w is None else torch.as_tensor(w, dtype=torch.float32)
        if w0.shape != (len(self.groups),):
            raise ValueError(f"w needs one weight per radial group ({len(self.groups)})")
        self.w, self.c = nn.Parameter(w0), nn.Parameter(torch.tensor(float(c)))
        self.rho = float(rho)

    def mass_ratios(self, energies) -> list:
        return [float(energies[c] / max(sum(energies[o] for o in others), 1e-30)) for c, others in self.groups]

    def cone_excess(self, comps: Tensor, mass: list) -> Tensor:
        """phi [B,G,X,Y,Z]: local log cone excess per radial group (0 for white noise)."""
        feats = []
        for (cone, others), m in zip(self.groups, mass):
            e_cone = gaussian_smooth(comps[:, cone:cone + 1].square(), self.rho)
            e_other = gaussian_smooth(comps[:, others].square().sum(1, keepdim=True), self.rho)
            eps = 1e-4 * e_other.mean(dim=(-3, -2, -1), keepdim=True).detach().clamp_min(1e-12)
            feats.append(torch.log((e_cone + eps) / (e_other + eps)) - math.log(m))
        return torch.cat(feats, dim=1) if feats else comps[:, :0]

    def forward(self, comps: Tensor, mass: list):
        """comps [B,S,X,Y,Z] (image components of the masked input) -> (removal [B,1,X,Y,Z], regime map r [B,1,X,Y,Z])."""
        phi = self.cone_excess(comps, mass)
        r = torch.sigmoid((phi * self.w.view(1, -1, 1, 1, 1)).sum(1, keepdim=True) + self.c)
        base = torch.einsum("s,bsxyz->bxyz", 1 - self.g0, comps)[:, None]
        delta = torch.einsum("s,bsxyz->bxyz", self.g1 - self.g0, comps)[:, None]
        return base - r * delta, r


class PhaseCleaner(nn.Module):
    """f(b_dirty, mask, weight) -> mask * (b_dirty - n_hat), all fields in ppm.

    The head is zero-initialized, so the untrained model is the masked identity. Dipolelet
    windows and k-features are built lazily per input shape (patches for training, the full
    volume for inference) and are not part of the state dict.
    """

    def __init__(self, voxel_size, b0=(0.0, 0.0, 1.0), *, channels=16, blocks=4, dilations=(1, 2, 4),
                 k_conditioning=True, input_scale=1.0, checkpoint=True, dipolelet=None, plain=False,
                 block_dilations=None, unet_base=None, regime=None, norm=None, norm_groups=4):
        super().__init__()
        if channels < 1 or blocks < 1 or input_scale <= 0:
            raise ValueError("channels and blocks must be positive, input_scale > 0")
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.b0 = tuple(float(v) for v in b0)
        self.dipolelet = dict(SIM2_DIPOLELET if dipolelet is None else dipolelet)
        self.use_checkpoint = checkpoint
        self.plain = bool(plain)  # ablation: one segment (the whole spectrum) instead of the dipolelet split
        names = ["all"] if self.plain else DipoleletTransform((8, 8, 8), self.voxel_size, self.b0, **self.dipolelet).segment_names
        segments = len(names)
        self.regime, self.regime_feed = None, False
        if regime is not None:      # CLAUDE.md 9ar; default off (no parameters, old state dicts load)
            if self.plain:
                raise ValueError("the regime filter needs the Dipolelet segments (plain=False)")
            options = dict(regime)
            self.regime_feed = bool(options.pop("feed", True))
            self.regime = RegimeFilter(names, **options)
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))
        self.lift = nn.ModuleList(nn.Conv3d(3 + int(self.regime_feed), channels, k, padding=k // 2) for k in (1, 3, 5, 7))
        self.spatial = nn.ModuleList(nn.Conv3d(channels, channels, 3, padding=d, dilation=d)
                                     for d in dilations)
        if block_dilations is not None and len(block_dilations) != blocks:
            raise ValueError("block_dilations needs one entry per block (0 = no local conv in that block)")
        local = [None] * blocks if block_dilations is None else list(block_dilations)
        if norm not in (None, "masked_group"):
            raise ValueError("norm must be None or 'masked_group'")
        self.blocks = nn.ModuleList(DipoleletBlock(channels, segments, k_conditioning, local_dilation=d,
                                                   norm_groups=norm_groups if norm else None) for d in local)
        self.unet = UNetBranch(channels, int(unet_base)) if unet_base else None
        self.unet_active = True  # test-time ablation switch: False drops the branch output
        self.last_branch_ratio = None  # ||branch output|| / ||v|| of the last forward (diagnostic)
        self.head = nn.Conv3d(channels, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._geometry_cache = {}

    def _geometry(self, shape, device, voxel_size=None):
        voxel_size = self.voxel_size if voxel_size is None else tuple(float(v) for v in voxel_size)
        key = (tuple(shape), voxel_size, str(device))
        if key not in self._geometry_cache:
            if self.plain:
                transform = PlainTransform(shape).to(device)
            else:
                transform = DipoleletTransform(shape, voxel_size, self.b0, **self.dipolelet).to(device)
            features = k_features(shape, voxel_size, self.b0, device=device)[None]
            mass = None
            if self.regime is not None:   # cone / non-cone energy ratio of white noise per radial group (the 'no excess' reference)
                white = torch.randn((1, 1, *shape), generator=torch.Generator().manual_seed(0)).to(device)
                energies = [float(s.abs().square().sum()) for s in transform.analysis(white)]
                mass = self.regime.mass_ratios(energies)
            self._geometry_cache[key] = (transform, features, mass)
        return self._geometry_cache[key]

    def forward(self, field: Tensor, mask: Tensor, weight: Tensor, voxel_size=None) -> Tensor:
        """`voxel_size` (mm) of this volume; defaults to the one the model was built with."""
        if field.ndim != 5 or field.shape[1] != 1 or mask.shape != field.shape or weight.shape != field.shape:
            raise ValueError("field, mask and weight must all have shape [B,1,X,Y,Z]")
        x = field / self.input_scale
        mask = mask.to(x.dtype)
        transform, features, mass = self._geometry(field.shape[-3:], field.device, voxel_size)
        removal, extra = 0.0, []
        if self.regime is not None:
            comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0]
                                 for s in transform.analysis(x * mask)], dim=1)
            removal, regime_map = self.regime(comps, mass)
            if self.regime_feed:
                extra = [regime_map]
        v = F.gelu(sum(conv(torch.cat([x, mask, weight, *extra], dim=1)) for conv in self.lift))
        for conv in self.spatial:
            v = v + F.gelu(conv(v))
        branch = None
        if self.unet is not None:
            branch = checkpoint(self.unet, v, use_reentrant=False)                 if self.use_checkpoint and torch.is_grad_enabled() else self.unet(v)
        for block in self.blocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                # Nested: per block, and inside it per segment (same math, far fewer live spectra).
                v = checkpoint(block, v, transform, features, True, mask, use_reentrant=False)
            else:
                v = block(v, transform, features, mask=mask)
        if branch is not None:
            with torch.no_grad():
                self.last_branch_ratio = float(branch.norm() / v.norm().clamp_min(1e-12))
            if self.unet_active:
                v = v + branch
        return mask * (x - self.head(v) - removal) * self.input_scale
