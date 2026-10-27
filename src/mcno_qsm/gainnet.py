"""Dipole-let gain network (CLAUDE.md 9az): a 3D U-Net that predicts spatially varying PASSIVE gains for the Dipole-let segments.

It is not an FNO: the Fourier side is only the fixed analysis (the segment images D_s psi, which sum to the input) and the network is a
convolutional U-Net that reads those images, the weight and the local regime cues and outputs one gain g_s(x) in (0,1) per segment,
    psi_hat = m (psi - sum_s (1 - g_s(x)) D_s psi),
so it can attenuate but never amplify a segment, and it can switch regime voxel by voxel. The gains start exactly at the regime-gated
passive linear filter of the cascade (g_s = 1 - r(x) (1 - g1_s), CLAUDE.md 9au): the last U-Net layer is zero-initialized and
g_s = sigmoid(logit(g_base_s) + z_s). `GainCascade` puts the frozen noise network (stage B) behind it and is trained through it.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .cleaner import PhaseCleaner, RegimeFilter
from .dipolelets import DipoleletTransform, gaussian_smooth


class UNet3d(nn.Module):
    """Small 3-level 3D U-Net (full, /2, /4; base channels doubling), any input shape; zero-initialized 1x1x1 output."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 16):
        super().__init__()
        c0, c1, c2 = base, 2 * base, 4 * base
        self.enc0 = nn.Conv3d(in_channels, c0, 3, padding=1)
        self.enc1 = nn.Conv3d(c0, c1, 3, padding=1)
        self.enc2 = nn.Conv3d(c1, c2, 3, padding=1)
        self.mid = nn.Conv3d(c2, c2, 3, padding=1)
        self.dec1 = nn.Conv3d(c2 + c1, c1, 3, padding=1)
        self.dec0 = nn.Conv3d(c1 + c0, c0, 3, padding=1)
        self.out = nn.Conv3d(c0, out_channels, 1)
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


class GainNet(nn.Module):
    """f(b, mask, weight) -> mask * (b - sum_s (1 - g_s(x)) D_s b) with g_s(x) in (eps, 1 - eps). Fields in ppm.

    Input channels of the U-Net (all computed from the input only): the S segment images D_s b / input_scale, mask, weight, the regime map r,
    the local log cone excess phi_g of each radial group and the log local rms of the field. `g1` is the passive gain of the real regime
    (one per segment, in [0, 1]); `w`, `c` define r = sigmoid(w . phi + c) as in RegimeFilter and, like g1, are fixed.
    """

    def __init__(self, voxel_size, b0=(0.0, 0.0, 1.0), *, input_scale=1.0, dipolelet=None, g1, w, c, rho=6.0, base=16,
                 amp_sigma=3.0, eps=1e-3, checkpoint=True):
        super().__init__()
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.b0 = tuple(float(v) for v in b0)
        self.dipolelet = dict(dipolelet)
        self.use_checkpoint, self.amp_sigma, self.eps = checkpoint, float(amp_sigma), float(eps)
        names = DipoleletTransform((8, 8, 8), self.voxel_size, self.b0, **self.dipolelet).segment_names
        self.segments = len(names)
        g1 = torch.as_tensor(g1, dtype=torch.float32)
        if g1.shape != (self.segments,) or bool((g1 < 0).any() or (g1 > 1).any()):
            raise ValueError(f"g1 needs {self.segments} passive gains in [0, 1]")
        self.regime = RegimeFilter(names, rho=rho, g0=[1.0] * self.segments, g1=g1, w=w, c=c)
        self.regime.requires_grad_(False)       # the regime map and the base gains are the fixed prior, the U-Net learns the modulation
        self.register_buffer("input_scale", torch.tensor(float(input_scale)))
        self.groups = len(self.regime.groups)
        self.unet = UNet3d(self.segments + 3 + self.groups + 1, self.segments, base)
        self.keep_gain = True       # store the last gain map (diagnostics, tests); GainCascade turns it off: 12 x 164x205x205 floats is 330 MB
        self.neutralize = ()        # inference diagnostics only: names among 'weight', 'amp', 'phi' replaced by their in-mask median (CLAUDE.md 9bb)
        self.last_gain = None
        self._geometry_cache = {}

    def _geometry(self, shape, device, voxel_size=None):
        voxel_size = self.voxel_size if voxel_size is None else tuple(float(v) for v in voxel_size)
        key = (tuple(shape), voxel_size, str(device))
        if key not in self._geometry_cache:
            transform = DipoleletTransform(shape, voxel_size, self.b0, **self.dipolelet).to(device)
            white = torch.randn((1, 1, *shape), generator=torch.Generator().manual_seed(0)).to(device)
            mass = self.regime.mass_ratios([float(s.abs().square().sum()) for s in transform.analysis(white)])
            self._geometry_cache[key] = (transform, mass)
        return self._geometry_cache[key]

    def forward(self, field: Tensor, mask: Tensor, weight: Tensor, voxel_size=None) -> Tensor:
        if field.ndim != 5 or field.shape[1] != 1 or mask.shape != field.shape or weight.shape != field.shape:
            raise ValueError("field, mask and weight must all have shape [B,1,X,Y,Z]")
        mask = mask.to(field.dtype)
        transform, mass = self._geometry(field.shape[-3:], field.device, voxel_size)
        with torch.no_grad():
            x = field * mask / self.input_scale
            comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0] for s in transform.analysis(x)], dim=1)
            phi = self.regime.cone_excess(comps, mass)
            r = torch.sigmoid((phi * self.regime.w.view(1, -1, 1, 1, 1)).sum(1, keepdim=True) + self.regime.c)
            amp = 0.5 * torch.log10(gaussian_smooth(x.square(), self.amp_sigma) + 1e-4)
            if self.neutralize:
                unknown = set(self.neutralize) - {"weight", "amp", "phi"}
                if unknown:
                    raise ValueError(f"cannot neutralize {sorted(unknown)}")
                flat = lambda t: torch.where(mask.expand_as(t) > 0, t.flatten(2)[..., mask.flatten(2)[0, 0] > 0].median(dim=-1).values[..., None, None, None].expand_as(t), torch.zeros_like(t))
                weight, amp, phi = (flat(t) if name in self.neutralize else t for name, t in (("weight", weight), ("amp", amp), ("phi", phi)))
            features = torch.cat([comps, mask, weight, r, phi, amp], dim=1)
            base = (1 - r * (1 - self.regime.g1.view(1, -1, 1, 1, 1))).clamp(self.eps, 1 - self.eps)
            base_logit = torch.log(base) - torch.log1p(-base)
        z = checkpoint(self.unet, features, use_reentrant=False) if self.use_checkpoint and torch.is_grad_enabled() else self.unet(features)
        gain = torch.sigmoid(base_logit + z)
        self.last_gain = gain.detach() if self.keep_gain else None
        out = mask * (x - ((1 - gain) * comps).sum(1, keepdim=True)) * self.input_scale
        if not torch.is_grad_enabled() and out.is_cuda:
            del features, z, gain, comps, phi, r, amp, base, base_logit, x      # full-volume inference: free stage A' before stage B runs
            torch.cuda.empty_cache()
        return out


def load_stage_b(path, device="cpu") -> PhaseCleaner:
    state = torch.load(path, map_location=device, weights_only=False)
    model = PhaseCleaner(state["voxel_size"], input_scale=state["input_scale"], dipolelet=state["config"]["dipolelet"], **state["config"]["model"])
    model.load_state_dict(state["model"])
    return model


class GainCascade(nn.Module):
    """Stage A' (`GainNet`, trained) then stage B (a frozen PhaseCleaner, the noise_clean network): f = B(A'(b) + outside), trained through B.

    B sees what it saw in training: the A' output inside the mask and the input field outside it (noise-free D*chi in the RC1 recipe, zeros for the
    real Sim2). Only the GainNet parameters get gradients; the checkpoint of B is part of the state dict (a saved run is self-contained).
    """

    def __init__(self, voxel_size, *, stage_b, input_scale=1.0, dipolelet, gain=None):
        super().__init__()
        self.gain = GainNet(voxel_size, input_scale=input_scale, dipolelet=dipolelet, **(gain or {}))
        self.gain.keep_gain = False
        self.stage_b = stage_b
        self.stage_b.requires_grad_(False)
        self.stage_b.eval()

    @classmethod
    def from_config(cls, voxel_size, input_scale, dipolelet, model_config, device="cpu"):
        options = dict(model_config)
        if options.pop("kind") != "gain_cascade":
            raise ValueError("not a gain_cascade model configuration")
        stage_b = load_stage_b(options.pop("stage_b"), device)
        return cls(voxel_size, stage_b=stage_b, input_scale=input_scale, dipolelet=dipolelet, gain=options)

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage_b.eval()
        return self

    def stage_a(self, field, mask, weight, voxel_size=None) -> Tensor:
        return self.gain(field, mask, weight, voxel_size)

    def forward(self, field: Tensor, mask: Tensor, weight: Tensor, voxel_size=None) -> Tensor:
        mask = mask.to(field.dtype)
        a = self.gain(field, mask, weight, voxel_size)
        return self.stage_b(a + field * (1 - mask), mask, weight, voxel_size=voxel_size)
