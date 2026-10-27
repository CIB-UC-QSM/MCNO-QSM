"""Peak CUDA memory of one training step (96^3, batch 2) for the configs of the regime experiment (CLAUDE.md 9ar). Server only."""
import sys
import tomllib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.cleaner import PhaseCleaner  # noqa: E402

size, batch = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (96, 2)
device = "cuda" if torch.cuda.is_available() else "cpu"
for name in ("s4b_noise_clean", "s4b_realmix_plain", "s4b_regime"):
    config = tomllib.loads(Path(f"configs/{name}.toml").read_text())
    model = PhaseCleaner((1.0625, 1.0625, 1.0714), input_scale=0.0084, dipolelet=config["dipolelet"], **config["model"]).to(device)
    shape = (batch, 1, size, size, size)
    field = torch.randn(shape, device=device) * 0.01
    mask, weight = torch.ones_like(field), torch.full_like(field, 0.26)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model(field, mask, weight).square().mean().backward()
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    print(f"{name:20s} parameters {sum(p.numel() for p in model.parameters()):7d}  peak {peak:.2f} GB  ({size}^3, batch {batch})", flush=True)
    del model, field
