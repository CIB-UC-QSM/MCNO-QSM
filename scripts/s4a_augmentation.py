"""Show object (anatomy) augmentation on RC1: example heads, rejection rate, cost, family mix.

Run: uv run python -I scripts/s4a_augmentation.py --out runs/s4a_gate
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mcno_qsm.artifacts import ObjectConfig, augment_object
from mcno_qsm.rc1_data import RC1Source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--out", type=Path, default=Path("runs/s4a_gate"))
    parser.add_argument("--heads", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = RC1Source(args.rc1, args.device, holdout={"axis": 0, "train_fraction": 0.625, "gap": 12})
    cfg = ObjectConfig()
    rng = np.random.default_rng(0)
    heads, started = [], time.perf_counter()
    for _ in range(args.heads):
        heads.append(augment_object(source.train_volume, rng, cfg))
    seconds = (time.perf_counter() - started) / args.heads
    ok = [h for h in heads if h is not None]
    original = float(source.train_volume.mask.sum())
    print(f"object draws kept: {len(ok)}/{len(heads)}; {seconds:.2f} s per head on {args.device} (160^3, incl. D*chi)")
    print("brain voxels vs the (truncated) original: mean ratio "
          f"{np.mean([float(h.mask.sum()) / original for h in ok]):.3f}, range "
          f"{min(float(h.mask.sum()) / original for h in ok):.3f}-{max(float(h.mask.sum()) / original for h in ok):.3f}")

    rng, generator = np.random.default_rng(1), torch.Generator(device=args.device).manual_seed(1)
    infos, started = [], time.perf_counter()
    for _ in range(60):
        infos.append(source.patch(96, rng, generator)[-1])
    if args.device != "cpu":
        torch.cuda.synchronize()
    print(f"patch cost: {(time.perf_counter() - started) / 60:.2f} s on {args.device}")
    print("object-augmented share:", np.mean([i["object_augmented"] for i in infos]))
    print("family mix:", dict(Counter(i["family"] for i in infos)))

    fig, axes = plt.subplots(2, 6, figsize=(18, 6.2))
    shown = [source.train_volume] + ok[:5]
    for column, head in enumerate(shown):
        for row, (data, title) in enumerate(((head.target, "target mask*D*chi"), (head.chi, "chi"))):
            image = data.cpu().numpy()[:, :, 70].T
            lim = 0.04 if row == 0 else 0.12
            axes[row, column].imshow(image, cmap="gray", vmin=-lim, vmax=lim, origin="lower")
            axes[row, column].set_title(("original (train slab only)" if column == 0 else f"rotated head {column}")
                                        + f"\n{title}, z=70", fontsize=8)
            axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(args.out / "object_augmentation.png", dpi=75)


if __name__ == "__main__":
    main()
