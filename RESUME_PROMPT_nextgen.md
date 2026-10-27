# Prompt for a new model: a Dipole-let physics-informed network beyond the FNO, scored by eNDI at 25 fixed iterations

You are taking over a research project (MCNO-QSM, folder `C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM`). Read `CLAUDE.md` (sections 9l to 9ay are the recent ones), `technicalReport/results_report.pdf` (findings, 10 pages) and `technicalReport/report.pdf` (the FNO architecture and data simulation) before proposing anything. The user is the author of the Dipole-lets paper (`technicalReport/paper`) and of the eNDI inversion; they are an expert, they want rigor, honest negatives and no mild variants of what already failed.

## 1. Aim

Design and test a **Dipole-let, physics-informed network that is NOT an FNO** (CNN, 3D U-Net, a hybrid, or an unrolled network; your choice, argued from the findings below) that **extracts the non-dipolar phase** `psi_N` (and noise) from a single-orientation local field `psi = psi_D + psi_N + eps`, so that **the eNDI reconstruction after 25 fixed iterations (the protocol of every table in this project) is closer to the ground-truth susceptibility than the current best (the passive gated cascade)**. The streaking of the eNDI reconstruction is the problem; scope is streaking, nothing else (do not build background-remnant models unless asked).

Notation (the paper's): `D(k) = 1/3 - kz^2/|k|^2`, magic cone `Sigma = {D=0}`, radial filters `R_j`, angular windows `A_m` (thresholds on `|D|`), Dipole-let `D_{j,m} psi = F^-1 R_j A_m F psi`, residual `D_res`, `S = JM+1` (here J=3, M=3, delta=0.05, 1/3); the split residual `D_{res,m}` gives S'=12. Fields in ppm. `mcno_qsm.dipolelets.DipoleletTransform` and `image_components` give the segments; `mcno_qsm.physics.dipole_kernel` is the correctly aligned kernel (the FANSI kernel is wrong for odd sizes).

## 2. What is established (single seed, RC1 head for training, Sim2 volume for test; eNDI NRMSE %, 25 fixed iterations)

Baselines to reproduce first: dirty, `B` = noise_clean FNO (`runs/s4b_noise_clean/best.pt`), and the passive gated cascade (`scripts/cascade_gate.py --passive`, table `runs/cascade_passive/table.txt`):

| field | dirty | B (FNO) | cascade A then B |
|---|---|---|---|
| exp7 (measured tissue residual, RC1) | 93.6 | 71.7 | 58.6 |
| exp8 (STI) / exp9 (off-diag.) | 67.7 / 83.3 | 61.2 / 76.3 | 52.4 / 60.5 |
| real Sim2 (calcification inside) | 53.9 | 47.5 | 48.7 |
| Sim2 forward-simulated SNR 100 / 40 / jumps | 61.5 / 121 / 66.8 | 46.7 / 52.1 / 48.4 | 46.9 / 52.5 / 48.6 |
| exp1 noiseless / exp3 SNR 40 | 35.8 / 61.2 | 36.1 / 43.0 | 36.1 / 43.1 |
| exp5 cylinders / exp6 phase spheres (strong localized) | 94 / 36 | 116 / 47 | 114 / 63 |

Oracle (eNDI of the clean masked field `m*psi_D`) at the same 25 iterations: 35.8 (RC1), 43.1 (Sim2); this is the reference for how much of the gap a cleaner can close (it is not a strict ceiling, because the mask removes signal that any masked field loses).

Findings you can rely on:
1. **Geometry.** In the Dipole-let domain real-like residuals (tissue, STI, off-diagonal) are separable by **angle** (best passive gain per (|k|,|D|) cell depends on |D|: ~0.1 at |D|<0.05, 0.4 at 0.15, 0.7 at 0.27, 0.85 at 0.33; near-cone energy share 13-37% against 0.2-0.9% for the clean field; local signal-to-residual -13..-17 dB in |D|<0.05 at every scale), while noise is separable by **scale** (gain ~1 below 0.1 cycles/voxel, ~0.01 above 0.6, weak |D| dependence). One set of gains cannot serve both: the wrong regime costs 8-12 eNDI points. The regime is readable from the input by the scale dependence of the local cone excess (`RegimeFilter`, `cleaner.py`).
2. **Information floor for noise.** A trained FNO is within 0.2 points of the best per-cell scalar filter (27.2% phase NRMSE on Sim2). Loss re-weighting (band, per-segment sensitivity), more anatomy (85 AHEAD heads), local convs, a U-Net branch, a calcification-shell family: all within ~0.5 eNDI points of `noise_clean` (47.3). Do not spend effort here.
3. **Phase NRMSE does not predict eNDI error.** Four decompositions differ by 3 points of phase error and <0.9 of eNDI; a network that lowers phase error can raise eNDI. Always judge by eNDI.
4. **A 12-parameter linear passive Dipole-let filter** (one gain in [0,1] per segment, split residual, gated by the regime map) beats the trained FNO on real residuals by 9-19 points, no training. The split of the coarse residual by |D| is worth 1.4-2.7 eNDI points; more radial scales add nothing on top.
5. **Open failure: strong compact sources** (the Sim2 calcification: +1.2 over B; exp6: 63 vs 47). The filter deletes near-cone content of true strong fields. In the 3 mm shell around the calcification the INPUT keeps only 32% of the true field (T2*-type dropout), so part of this is information.
6. A data-driven redesign of the decomposition (DP over radial groups x |D| bands) gives 6 segments equal to the hand-set 12 on real residuals in eNDI but worse on strong sources; real-regime best |D| edges ~0.10/0.15/0.20/0.27, noise-regime radial edges ~0.15/0.27/0.40/0.49 cycles/voxel.

## 3. Negative results: do not repeat these

Reweighting the field loss; extra anatomy; U-Net branch or local convs *inside the same noise recipe*; calcification-shell training family; mixing structured families with noise in one network (task interference: 49.05 vs 46.96); amplitude gating of stage A (two scales); passivity alone (halves but does not fix strong-source damage); plug-and-play residual refinement inside eNDI with these cleaners (worse on every row, probably out-of-distribution residual amplitude); clipping the field; a data-driven decomposition as a lever on eNDI. The user explicitly does not want "the same experiment with other noise levels/seeds/weights".

## 4. Rules (non-negotiable)

- **Sim2 is a test volume only**: never for training, checkpoint selection, threshold or architecture choices. Read gains, gates, thresholds from RC1 (and AHEAD if used).
- **Write success criteria before each run** (they are in CLAUDE.md for every earlier run), report negatives as negatives, say "single seed / one head / one volume" next to every number.
- **Protocol**: `ndiCG` (`external/eNDI_module_torch.py`, never modify the user's modules), alpha 1e-5, zero padding (192^3 RC1, 256^3 Sim2), TE/B0 from `scripts/cascade_gate.py` (SETTINGS), score = NRMSE of chi inside the mask after **25 fixed iterations** (history index 25, i.e. 26 gradient steps; no automatic stop, because dirty fields diverge after 4-17 iterations). Use the helpers in `scripts/rescore_fixed.py` / `scripts/cascade_gate.py`; add new methods as extra columns of the existing tables so that results stay comparable.
- The user's mask is BET; there are no background remnants in the data. The user has in-vivo data with real phase but no ground truth; ask before using it.
- Compute: laptop CPU for small tests (RC1 eNDI ~1-2 min), server RTX 5090 shared (`MCNO_VRAM_GB=10`, `ssh igna@100.125.57.28`, project `~/mcno/MCNO-QSM`, `.venv/bin/python`, `nohup`, no tmux). When you give the user commands: two separately labelled blocks, "PowerShell on your PC" (start with `cd` to the project folder, real address, one-line scp) and "Server"; check `nvidia-smi` free memory first; never edit a script a queue is running.
- New code needs tests; `uv run pytest -q`-style tests exist in `tests/`; configs are TOML in `configs/`; document every run in `CLAUDE.md` with the next section number (9az...).

## 5. Where the new idea should push (hypotheses to test, not instructions)

a. **Dipole-lets as the input representation of a spatial network.** The FNO ties spectral weights to the segments; a CNN/U-Net may do better at what the findings say is missing: *local*, *source-aware* decisions. Feed the S'=12 (or learned) segment images `D_s psi` (and the regime map `r`, the local cone excess, the weight/magnitude, ideally the four echo magnitudes of Sim2 and their noise levels) as channels to a 3D U-Net or a multiscale CNN; predict **spatially varying passive gains** `g_s(x) in [0,1]` (sigmoid) per segment, `psi_hat = m (psi - sum_s (1-g_s(x)) D_s psi)`, so the network can attenuate but never amplify a strong source, and can switch regime per voxel. Compare with the fixed-gain stage A (12 numbers) as the baseline to beat. Consider separate banks (angular for real-like, radial for noise) with a router, since one decomposition cannot serve both.
b. **Source awareness.** Strong compact sources are the unsolved case. Candidate cues the field amplitude alone does not give: the magnitude/weight dropout (T2*) and multi-echo decay, the local consistency of the field with a dipole of a compact source (the field of a localized chi has a known angular pattern and zero cone content up to mask leakage), and the residual of a few eNDI iterations. Validate on RC1 constructions (exp5/exp6 and new compact-source phantoms with magnitude dropout) before looking at Sim2.
c. **A task-aware objective (optional, costly).** The loss could be the **chi error after the fixed-iteration eNDI** through a differentiable unrolled `ndiCG` (nonlinear in `sin` but differentiable; memory grows with the number of unrolled steps, so test with a few steps and a patch first, then check against the full 25-iteration score). For near-linear fields the unrolled operator is linear and the loss is a positive-definite quadratic, so it can only matter through (i) the early-stopping bias of a finite-step solver, (ii) the weight `w^2`, (iii) the `sin` nonlinearity near strong sources. A variant: the network also outputs a **warm start `x_0`**. Keep a field-domain term as regulariser. Risks to test explicitly: hallucination of chi, patch-edge mismatch with the padded full-volume score, prior unrolling work finding no gain from unrolling alone. Phase-only training is the cheaper default; use the task loss only if a prototype shows a reason (e.g. the strong-source failure).
d. **Better residual training data.** Only exp7 is a *measured* non-dipolar phase (one head). STI and off-diagonal residuals are modelled. If a learned model is trained on modelled residuals, hold exp7 out and report it separately; a self-supervised option for the user's in-vivo data (cone-masked Noise2Self: predict the cone content of the field from the field with the cone removed) is an idea worth a feasibility test before any large training.
e. Keep the useful priors: passivity, regime awareness, the split of the coarse residual by |D|, `delta` is irrelevant for the linear part, scales matter for noise.

## 6. Suggested order of work (each stage ends with a go/no-go and the user's approval for anything long)

0. Reproduce the baseline table (dirty, B, cascade) at 25 iterations and verify that your scoring code gives the same numbers (laptop/server, no training).
1. Pre-register criteria for the new network (e.g. beat the cascade on exp7, exp8, exp9 at 25 iterations by a stated margin, with no harm on noise-like fields, exp1, exp3 and the forward Sim2 fields; a separate criterion for strong sources, exp5/exp6 and the Sim2 calcification; Sim2 real only as a final test).
2. Small, fast prototypes on RC1 (patches 96^3, <= 10 GB): variant (a) with fixed segment inputs, then, only if justified, (c) with an unrolled-eNDI loss; ablate each ingredient against the 12-parameter stage A.
3. Only if a prototype passes its gate: scale, seeds, then the full evaluation including the real Sim2 and the user's in-vivo data.
4. Write the results in `technicalReport/results_report.tex` style (same notation) and in `CLAUDE.md`.

Be direct about what fails. If the evidence says the FNO-based cascade is already near what the data support, say so with numbers.
