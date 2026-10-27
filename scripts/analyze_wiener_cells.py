"""Per-(|k|,|D|) cell: residual energy of the trained cleaner vs the best oracle scalar linear filter per cell (CLAUDE.md 9ap).
Floor_cell = E_b - <b,d>^2/E_d (fitted on the volume itself: optimistic). Reports totals as phase NRMSE (% of ||b||) and where the net
sits above / below that floor."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

data = load_rc2("dataRC2", 7.0)
mask, chi, voxel = data["mask"].astype(np.float32), data["chi"], data["voxel_size"]
shape = chi.shape
b = forward_field(chi, mask, voxel)
d = data["field"] * mask
c = nib.load("runs/s4b_noise_clean/best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
kern = np.abs(dipole_kernel(shape, voxel, dtype=torch.float64).numpy())
fr = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
kmag = np.sqrt(sum(f ** 2 for f in fr))
B, Dd, C = (np.fft.fftn(x) for x in (b, d, c))
ke = np.geomspace(0.01, 0.9, 25)
de = np.array([0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.27, 0.333, 0.5, 0.67])
ik = np.digitize(kmag.ravel(), ke) - 1
idd = np.digitize(kern.ravel(), de) - 1
nk, nd = len(ke) - 1, len(de) - 1
ok = (ik >= 0) & (ik < nk) & (idd >= 0) & (idd < nd)
flat = ik[ok] * nd + idd[ok]
def acc(w):
    return np.bincount(flat, weights=w[ok], minlength=nk * nd).reshape(nk, nd)
Bf, Df, Cf = B.ravel(), Dd.ravel(), C.ravel()
Eb = acc(np.abs(Bf) ** 2); Ed = acc(np.abs(Df) ** 2)
Ebd = acc(np.real(Bf * np.conj(Df)))
Er = acc(np.abs(Cf - Bf) ** 2)           # net residual
Eres_dirty = acc(np.abs(Df - Bf) ** 2)
floor = Eb - Ebd ** 2 / np.maximum(Ed, 1e-30)
wien_simple = Eb * (Ed - Eb) / np.maximum(Ed, 1e-30)  # unused placeholder
tot_b = np.abs(Bf).sum() and (np.abs(Bf[ok]) ** 2).sum()
f = lambda x: 100 * np.sqrt(x.sum() / tot_b)
print(f"phase NRMSE % within the binned k-range: dirty {f(Eres_dirty):.2f}  net(noise_clean) {f(Er):.2f}  oracle per-cell scalar filter {f(floor):.2f}")
print("(energy outside the bins is DC / beyond 0.9)")
print("\nResidual energy per |k| row, % of ||b||^2 : dirty | net | floor | net-floor; and per |D| col")
for i in range(nk):
    print(f"{ke[i]:5.3f}-{ke[i+1]:5.3f}  {100*Eres_dirty[i].sum()/tot_b:7.3f} {100*Er[i].sum()/tot_b:7.3f} {100*floor[i].sum()/tot_b:7.3f} {100*(Er[i].sum()-floor[i].sum())/tot_b:+7.3f}")
print(f"\n{'|D| bin':>12s}   dirty     net   floor   net-floor")
for j in range(nd):
    print(f"{de[j]:5.3f}-{de[j+1]:5.3f}  {100*Eres_dirty[:,j].sum()/tot_b:7.3f} {100*Er[:,j].sum()/tot_b:7.3f} {100*floor[:,j].sum()/tot_b:7.3f} {100*(Er[:,j].sum()-floor[:,j].sum())/tot_b:+7.3f}")
print("\nnet - floor per cell (x0.01 % of ||b||^2); rows |k|, cols |D| upper edge")
print(" " * 12 + " ".join(f"{e:6.3f}" for e in de[1:]))
for i in range(nk):
    print(f"{ke[i]:5.3f}-{ke[i+1]:5.3f} " + " ".join(f"{10000*(Er[i,j]-floor[i,j])/tot_b:6.1f}" for j in range(nd)))
neg = (Er - floor)
print(f"\ncells where net is BELOW the oracle scalar floor: {int((neg<0).sum())} of {nk*nd}; energy below {100*neg[neg<0].sum()/tot_b:.3f}, above {100*neg[neg>0].sum()/tot_b:.3f} (% of ||b||^2)")
