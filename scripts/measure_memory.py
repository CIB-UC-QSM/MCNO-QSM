"""Peak CUDA memory of one PhaseCleaner training step per patch size / batch / checkpoint mode."""

import argparse

import torch

from mcno_qsm.cleaner import PhaseCleaner


def peak_gb(size, batch, channels, segment_checkpoint):
    model = PhaseCleaner((0.999,) * 3, channels=channels).cuda()
    if not segment_checkpoint:
        # Reference mode: block-level checkpoint only (previous behavior).
        from mcno_qsm import cleaner
        original = cleaner.checkpoint
        cleaner.checkpoint = lambda f, v, t, ft, _seg, **kw: original(f, v, t, ft, False, **kw)
    shape = (batch, 1, size, size, size)
    field = torch.randn(shape, device="cuda") * 0.01
    mask, weight = torch.ones_like(field), torch.ones_like(field)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model(field, mask, weight).square().mean().backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    if not segment_checkpoint:
        cleaner.checkpoint = original
    return peak


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 112, 128])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    for size in args.sizes:
        for batch in args.batches:
            for seg in (False, True):
                try:
                    print(f"size {size}^3 batch {batch} segment_ckpt={seg}: "
                          f"{peak_gb(size, batch, args.channels, seg):.2f} GB", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"size {size}^3 batch {batch} segment_ckpt={seg}: OOM", flush=True)
                torch.cuda.empty_cache()
