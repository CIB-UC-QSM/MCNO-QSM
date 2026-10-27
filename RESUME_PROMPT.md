# OBSOLETE (S4a is done): the current prompt is `RESUME_PROMPT_S4b.md`.

# Prompt for the next session: S4 (paste into Claude Code, working dir `MCNO-QSM`)

The text between the lines is the prompt. (The previous session-1 prompt is obsolete: its questions were answered in `CLAUDE.md` §9b.)

---

```markdown
Read `CLAUDE.md` completely first (especially §8 server procedure, §9b/9c/9d, §5, §6), then `utils/Dipole-lets_Paper_Plots_3.ipynb`
cells "Data from FANSI" and the artifact cell (the one defining `psi_star`, `sphere_phantom_*`, `cylinder_phantom_*`, `exp1`..`exp11`).
Do not re-derive what CLAUDE.md already settles. No commits unless I ask. Report negative results plainly.

## How to talk to me about commands (important)
Whenever I must run something, give TWO separate, clearly labeled blocks and nothing ambiguous:
 (A) "PowerShell on my PC": starts with `cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM`,
     uses the real address `igna@100.125.57.28`, never placeholders like `<host>`.
 (B) "Server": commands I type after `ssh igna@100.125.57.28` (project at `~/mcno/MCNO-QSM`, always `.venv/bin/python`,
     long jobs with `nohup`, never tmux). Follow CLAUDE.md §8 exactly. Say which files changed so I upload only those.
Tell me what output to paste back (e.g. `cat runs/<name>/eval.csv`).

## Where we are (2026-10-19)
Details and tables are in CLAUDE.md §9d. Short version:
- The PhaseCleaner (dipolelet-FNO, 67k parameters) trained on Sim2 only FITS that volume (phase NRMSE 42% -> 10%, eNDI 50.9% -> 45.8%)
  but does not generalize: on a z-slab never seen (S2b) the phase improves 65% -> 55% yet eNDI(test) is 65.5% vs 64.1% for the dirty
  field. Test error is best after ~500 steps then rises while train error falls; using only simulated noise (no real psi) changes
  nothing. Likely cause: a single brain. So S4 = many different volumes + artifacts.
- Plan decided with me: **train on Challenge 1 (`dataRC1`) plus simulated artifacts; use Challenge 2 (Sim2, `dataRC2`) ONLY as the
  untouched test target** (no Sim2 voxel in training, none for checkpoint selection; Sim2 numbers are reported at the end and as
  monitoring curves only). Select checkpoints on a held-out RC1 region (e.g. a z-slab with a gap, as in `[data.holdout]`).

## Phasing (decided 2026-10-20)
S4a/S4b use RC1 + artifacts ONLY. AHEAD (105 in-vivo 7 T subjects already on the server in `dataAHEAD/`, see CLAUDE.md §9e) is added
later as S4c, after I have seen the RC1-only result and refined the architecture. Do not build AHEAD code before I ask.
The cleaner now checkpoints per segment (`scripts/measure_memory.py`: 128^3 batch 2 = 12 GB, 96^3 batch 1 = 2.7 GB).

## Stage S4a — build the data + GATE (no training before I approve)
1. `src/mcno_qsm/rc1.py`: loader for `dataRC1/*.mat` (chi_cosmos minus its mean, chi_sti, magn, msk, phs_tissue, spatial_res[0]);
   units in ppm; RC1: TE 25 ms, B0 2.8936 T, 160^3, voxel ~[1.0625, 1.0625, 1.0714] mm; clean target
   `b_clean = mask * D*chi_cosmos` with the repo kernel (correct for odd N; RC1 is even).
2. `src/mcno_qsm/artifacts.py`: seeded, individually toggleable generators following the notebook, in ppm, with RANDOM placement/
   sign/size per sample (the notebook uses fixed positions):
   - exp2/exp3 complex noise SNR 100 / 40: `signal = mag*exp(i*b*phs_scale) + (1/SNR)(randn + i randn)`, `phase = angle/phs_scale`.
     NOTE the notebook uses `mag_base = normalized chi_cosmos * mask` as magnitude, not `magn`: decide with me which to use
     (compare with Sim2, where I use the real magnitude echo 2).
   - exp4 phase jumps `psi_star`: single-voxel impulses (+-1, +-2 ppm) added directly to the phase.
   - exp5 cylinders (r=8, height 8, +-0.5 ppm, rotated 45/45/45 and 30/30/30) added to the phase.
   - exp6 spheres (r=4, +-0.33 ppm) added to the phase.
   - exp7 real `phs_tissue` (target COSMOS, psi_real = phs_tissue - b_clean).
   - exp8 `pSTI = p33 - (p13 + p23)` and exp9 `-pQ` (STI off-diagonal terms; kernels as in the notebook).
   - exp11 spheres (r=4; +0.33, -0.33, +0.33 ppm) added to the SUSCEPTIBILITY (`D*(chi_cosmos + spheres)`) plus noise SNR 40.
   - **exp12 (new, requested by me): spheres in chi (as exp11) + `psi_star` phase jumps + noise, all together** — this should match
     the artifacts seen in Sim2 best. Sample SNR from {100, 40} or a range, and calibrate so psi rms spans Sim2's (~0.0037 ppm,
     CLAUDE.md §9c; RC1 phs_scale 19.4 rad/ppm vs Sim2 15.0 rad/ppm, so the same SNR gives different ppm noise).
   Working assumption (confirm with me): in every experiment the TARGET stays `mask * D*chi_cosmos`; spheres/cylinders/jumps are
   artifacts to remove (the notebook scores against `chi_cosmos`).
   Known notebook quirks to handle deliberately, not copy blindly: `create_geometric_phantom` does `z, y, x = np.indices(...)` and
   then uses `center[0]` for `x`, i.e. the axes are permuted (harmless for a cube, matters for placement); `signal3` uses `noise2`
   (equal to 1/40 so harmless).
3. Model/voxel handling: `PhaseCleaner` is built with one `voxel_size` and caches dipolelets per shape. RC1 is anisotropic and slightly
   different from Sim2 (0.999 mm). Make the cleaner take the voxel size per volume (or resample), with tests, so that training on
   RC1 and testing on Sim2 uses the correct physical k-features and dipolelet windows.
4. **GATE: sanity report before any training** (show me numbers and PNG slices; wait for my OK):
   - Architecture: parameter count, receptive field (dilations 1/2/4 plus global FFT branch), identity at init, memory at 96^3,
     behavior on RC1 and on Sim2 shapes/voxel sizes, gradient flow; whether 10 segments x scalar gain is sensible or needs the
     per-k C x C mix; confirm the plain-FNO ablation is wired (a config switch with one segment).
   - Training set: slices of every experiment (input, target, difference = psi), histograms of ppm values, psi rms and spectral
     energy split (|k|>0.2 cyc/mm etc.) compared with Sim2's measured psi (CLAUDE.md §9c), SNR actually achieved, fraction of
     patches with enough mask, that the target equals `mask * D*chi_cosmos`, determinism by seed, and that no RC2 file is read by the
     training code path. Also check the number of distinct anatomies (RC1 is ONE volume; say honestly what diversity the augmentations
     really add and whether more sources — phantoms from `Segmentation.nii` of Sim2 is NOT allowed since RC2 is the test — are needed).
   - Baselines on RC2 with the same settings as before: dirty, low-pass, oracle (CLAUDE.md §9d), so the new numbers are comparable.

## Stage S4b — training and tests (after my approval)
- Experiments (same seed/steps/optimizer, new `runs/<name>` each, ask before numbering variants): train on each artifact family alone
  (exp2/3, exp4, exp5, exp6, exp7, exp8+9, exp11), on the all-in-one mix, and on exp12 (spheres in chi + psi_star + noise) alone and
  in the mix. For each: RC1 held-out region (phase NRMSE, eNDI) and RC2 full volume (phase NRMSE, eNDI NRMSE/XSIM vs dirty, low-pass,
  oracle). Evaluate every 250 steps; keep the early-stopping lesson from S2b (best fresh-noise test error was at ~500 steps).
- Scoring is always my eNDI (`ndiCG_auto`, settings in CLAUDE.md §9b) via `scoring.py`; report a table with all rows, single-volume/
  single-seed caveats, and the step of the best checkpoint.
- Then S5 ablations (plain FNO vs dipolelet split; with/without k-conditioning and spatial branch; magnitude input).

## Housekeeping I want
- Keep CLAUDE.md current (it is the memory of this project): add S4 results and decisions; fix remaining stale text in §2, §3.3, §4, §5.2.
- Tests must pass on the laptop (`uv run pytest -q tests`) and on the server before any long run.
- README is Spanish; code, comments and CLAUDE.md are English.
```

---

## Notes for me (not part of the prompt)

- S2b result: negative for generalization; details in CLAUDE.md §9d.
- Server: `ssh igna@100.125.57.28`; procedure in CLAUDE.md §8.
