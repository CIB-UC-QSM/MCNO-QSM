"""Parallel spatial convolutions followed by independent spectral MLPs."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .dipolelets import DipoleletTransform
from .physics import hermitian_projection


class ComplexMLP(nn.Module):
    """Ten complex linear layers, six input channels and one output channel."""

    def __init__(self, width=16):
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive")
        dimensions = [6] + [width] * 9 + [1]
        self.layers = nn.ModuleList(
            nn.Linear(a, b, bias=False, dtype=torch.complex64)
            for a, b in zip(dimensions, dimensions[1:])
        )
        # Compensate attenuation from nine split GELU activations.
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.complex(F.gelu(x.real), F.gelu(x.imag)) * 1.5
        return self.layers[-1](x)


class DipoleletNet(nn.Module):
    def __init__(self, shape, voxel_size, b0=(0.0, 0.0, 1.0), *, scales=3,
                 angular_bands=(8, 4, 2), delta=0.03, tightness=100.0,
                 width=16, chunk_size=262144, checkpoint=True):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.convolutions = nn.ModuleList(nn.Conv3d(1, 2, k, padding=k // 2)
                                         for k in (3, 5, 7))
        self.transform = DipoleletTransform(
            shape, voxel_size, b0, scales=scales, angular_bands=angular_bands,
            delta=delta, tightness=tightness,
        )
        self.mlps = nn.ModuleList(ComplexMLP(width) for _ in self.transform.segment_names)
        self.chunk_size = chunk_size
        self.use_checkpoint = checkpoint

    def cnn(self, x: Tensor) -> Tensor:
        return torch.cat([F.gelu(conv(x)) for conv in self.convolutions], dim=1)

    def process_segment(self, segment: Tensor, mlp: ComplexMLP) -> Tensor:
        # Each row is one frequency of one sample; only six CNN channels mix.
        rows = segment.movedim(1, -1).reshape(-1, 6)
        outputs = []
        for chunk in rows.split(self.chunk_size):
            if self.use_checkpoint and torch.is_grad_enabled():
                outputs.append(checkpoint(mlp, chunk, use_reentrant=False))
            else:
                outputs.append(mlp(chunk))
        return torch.cat(outputs).reshape(segment.shape[0], *segment.shape[-3:]).unsqueeze(1)

    def forward(self, field: Tensor) -> Tensor:
        if field.ndim != 5 or field.shape[1] != 1:
            raise ValueError("field must have shape [B,1,X,Y,Z]")
        spectrum = None
        for segment, mlp in zip(self.transform.analysis(self.cnn(field)), self.mlps, strict=True):
            output = self.process_segment(segment, mlp)
            spectrum = output if spectrum is None else spectrum + output
        spectrum = hermitian_projection(spectrum)
        return torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="ortho").real


class StackedDipoleletNet(nn.Module):
    """Compose complete CNN/dipolelet layers, each with independent weights.

    Only the measured input is normalized. Intermediate single-channel volumes
    pass directly to the next layer, without masking or a physical forward.
    """

    def __init__(self, shape, voxel_size, b0=(0.0, 0.0, 1.0), *, depth=2, **kwargs):
        super().__init__()
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")
        self.layers = nn.ModuleList(DipoleletNet(shape, voxel_size, b0, **kwargs)
                                    for _ in range(depth))

    @property
    def transform(self):
        return self.layers[0].transform

    def forward(self, field: Tensor) -> Tensor:
        for layer in self.layers:
            field = layer(field)
        return field


def build_model(shape, voxel_size, b0=(0.0, 0.0, 1.0), *, depth=1, **kwargs):
    """Keep depth=1 identical to the original model and checkpoint format."""
    if not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if depth == 1:
        return DipoleletNet(shape, voxel_size, b0, **kwargs)
    return StackedDipoleletNet(shape, voxel_size, b0, depth=depth, **kwargs)
