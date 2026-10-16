"""Masked metrics with no fitted scale, offset, or susceptibility reference."""

import torch
from torch import Tensor, nn

from .physics import DipoleForward


def masked_nrmse(prediction: Tensor, ground_truth: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != ground_truth.shape or mask.shape != ground_truth.shape:
        raise ValueError("prediction, ground truth and mask must have identical shapes")
    difference = (prediction - ground_truth)[mask.bool()]
    denominator = torch.linalg.vector_norm(ground_truth[mask.bool()])
    if denominator.item() == 0:
        raise ValueError("masked ground truth must have a nonzero norm")
    return torch.linalg.vector_norm(difference) / denominator


class QSMObjective(nn.Module):
    """Normalized physics MSE plus normalized susceptibility L1 inside mask."""

    def __init__(self, forward: DipoleForward, ground_truth: Tensor, mask: Tensor,
                 physics_weight=1.0, l1_weight=1.0):
        super().__init__()
        if (not all(0 <= w < float("inf") for w in (physics_weight, l1_weight))
                or physics_weight + l1_weight == 0):
            raise ValueError("loss weights must be finite, nonnegative and not both zero")
        self.physics = forward
        self.register_buffer("ground_truth", ground_truth.detach())
        self.register_buffer("mask", mask.bool())
        with torch.no_grad():
            target = forward(ground_truth * mask)
            epsilon = torch.finfo(ground_truth.dtype).tiny
            self.register_buffer("target_field", target)
            self.register_buffer("field_energy", target[self.mask].square().mean().clamp_min(epsilon))
            self.register_buffer("chi_magnitude", ground_truth[self.mask].abs().mean().clamp_min(epsilon))
        self.physics_weight = physics_weight
        self.l1_weight = l1_weight

    def forward(self, prediction: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        prediction = prediction * self.mask
        error = self.physics(prediction) - self.target_field
        physics = error[self.mask].square().mean() / self.field_energy
        l1 = (prediction - self.ground_truth)[self.mask].abs().mean() / self.chi_magnitude
        total = self.physics_weight * physics + self.l1_weight * l1
        return total, physics, l1
