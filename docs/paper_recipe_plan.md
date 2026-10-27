# Paper-recipe replication: plan, choices, criteria (written 2026-10-25, BEFORE any result)

Paper: Lambert, Silva, Milovic, Tejos, "Deep Fourier-Space Inversion", ISMRM 2023, abstract 4171. We copy its TRAINING RECIPE and tests onto our
phase cleaner; we do not adopt its inversion (the plan stays phase cleaning + eNDI). Reference for every run: `noise_clean`
(`configs/s4b_noise_clean.toml`: seed 42, 6000 steps, batch 2, 96^3 patches, 67,305-parameter `PhaseCleaner`, same RC1 check slab for the
checkpoint selection, Sim2 monitored only). Only the data / augmentation changes. It is summarized as ledger L8 / section 9bo of CLAUDE.md (compacted file in place since 2026-10-25).

## What was built (all keys default-off; the patches of the old configs are bit-identical, checked with a recorded 6-batch reference; 160 tests pass)

| run (config `configs/s4b_<name>.toml`) | paper item | what changes vs `noise_clean` |
|---|---|---|
| `paper_e1_rc1` | E1 single brain | SNR 30-300; `chi_scale` [0.5, 2] (log-uniform factor on chi, field recomputed); `voxel_sizes` 1, 1.25, 1.5, 2, 2.5, 3 mm (head resampled, one size per batch, kernel and k-features per batch); axis permutation = the 24 cube rotations + 20 deg tilts that `objects` already applies (test: the z axis of an object reaches x and y) |
| `paper_e1_ahead1` | E1 on one AHEAD subject | same, on a single AHEAD subject (`[data.ahead] n_train = 1`, `fraction = 1.0`) instead of the RC1 head |
| `paper_e2_geo` | E2 geometric figures only | no brain: 64^3 volumes of random spheres and cylinders in chi (`artifacts.geometric_chi`, ellipsoidal mask), SNR 30-120, 15000 steps x batch 2 = 30,000 volumes |
| `paper_e3_ft` | E3 fine-tune | E2 `best.pt` fine-tuned on 2 AHEAD subjects: `chi_part_weight` [0.5, 2] (para- and diamagnetic parts scaled independently), `chi_flip_prob` 0.5, `snr_mix` 30-50 (30%) / 90-150 (70%), chi scale, cube rotations; 3000 steps, lr 3e-4 |
| `paper_e4_l2only` | E4 loss | masked L2 only (`l1_weight = 0`); RC1 data of `noise_clean` |
| `paper_ws` | weight-scale test (item 6) | `weight_scale` [0.5, 3]: random factor on the weight channel; the noise is drawn from the scaled magnitude (about 1/weight), as the paper-style tie |
| `paper_e1_rc1_ws` | E1 + weight scale | `paper_e1_rc1` plus `weight_scale` |

Paper tests (item 5): `scripts/export_paper_tests.py` builds forward simulations from COSMOS at SNR 40, SNR 100 and SNR 100 plus a smooth non-dipolar
phase (`paper_snr100_wrap`), reports the phase NRMSE of each cleaner, and writes the same npz layout as `export_generalization.py`, so
`rescore_fixed.py` gives the eNDI numbers (side note only).

## Choices that are ours (the abstract does not give them) and limits

- Susceptibility ranges (x0.5..2, parts x0.5..2), the voxel-size list, the fine-tune length and rate (3000 steps, 3e-4), 2 AHEAD subjects (the paper used 2 synthetic T1 brains), the geometric-figure ranges (counts 3-12 / 2-8, radii 3-10 / 2-8 voxels, |chi| 0.02-0.12 ppm, chosen so that the clean-field rms in the mask is 0.0086 ppm against 0.0089 for the RC1 training region) and the ellipsoidal mask are our choices. None was tuned on Sim2 (Sim2 is monitored only).
- "Value mapping" of the paper (T1 intensity classes to chi) has no analogue here: our heads already carry chi; chi scaling and the part weights cover the amplitude side.
- Voxel size: the dipole kernel is voxel-size independent in normalized frequency; what changes is the anatomy in voxels and the physical |k| fed to the network. Patches shrink to the largest multiple of 8 that fits the resampled head (96 at 1-1.5 mm, 80 at 2, 64 at 2.5, 56 at 3 mm for a 160^3 RC1 head) because a 96^3 patch of a 3 mm head would be 95% padding. The SNR is per voxel (not scaled with voxel volume).
- E2 still loads RC1, only for the monitoring side (check slab, input scale 0.0089 ppm, phase scale, native voxel size); no RC1 voxel reaches a training patch (test). Selection is the RC1 check-slab score of the trained families (noise 1.0, clean 0.25 as in `noise_clean`), so it is comparable.
- The wrap test: the abstract gives no amplitude for the "strong phase inconsistency". The RC1 clean field has rms 0.0084 ppm and wraps only beyond 0.162 ppm, so any real wrapping needs a non-dipolar phase about 10x the signal: default rms 0.1 ppm (12% of the brain wrapped; 0.08 -> 5%, 0.15 -> 29%). The cleaner was never trained on unwrapped-versus-wrapped data and is not expected to unwrap; the report gives the plain phase NRMSE and a wrap-invariant one (differences modulo 2 pi), because eNDI uses sin and does not see exact 2 pi wraps.
- AHEAD is processed QSM (target = reconstruction, not COSMOS, no real psi). One seed per run; one RC1 head; one Sim2 volume. Single seed is labeled single seed.
- Not built (ask first, per the project's rule against new inversion methods): the paper's Fourier-space input representation (Re/Im of the FFT of the field + kernel as 3 channels) as an alternative front end. Not available: the abdominal QSM phantom (Silva et al., ISMRM 2022).
- GPU time is an estimate, not measured: E1 about 2 h each, E2 about 2-5 h (64^3 patches, 15000 steps), E3 about 1 h, on a shared GPU; scoring about 1-2 h.

## Success criteria (CONFIRMED by the user on 2026-10-25, before any result; also in CLAUDE.md DECISION RULES)

Read from ONE fresh fixed-iteration scoring (`after_paper.sh` scores `noise_clean` in the same pipeline, so compare with that number, not with a quoted one:
the project has seen 47.31, 47.38 and 47.48 for the same net from different scripts). Real Sim2 means `best_clean.nii.gz` through eNDI (ndiCG, alpha 1e-5).
1. A run counts as BETTER than `noise_clean` only if real Sim2 @25 is at least 0.5 points below it, OR the weight-scale test improves (field domain, `sim2_snrcal` and real Sim2 at weight x2.8: error ratio to the x1 condition at least 0.10 lower than `noise_clean`'s 1.20 / 1.08 of 9bf) AND, in both cases, no harm: the forward Sim2 fields and RC1 exp1 are not worse than `noise_clean` by more than 0.5 (eNDI @25).
2. The same reading at 100 iterations is reported next to it; it changes the verdict only if @25 and @100 disagree by more than 1.0, and then it is said so.
3. Paper tests: on `paper_snr40` and `paper_snr100` the phase NRMSE of a run is not worse than `noise_clean`'s by more than 1.0 point (percent); `paper_snr100_wrap` is reported without a criterion (both phase NRMSE and the wrap-invariant one).
4. Verdicts: E2 and E3 are reported whatever the outcome (a null is a result: it would say that anatomy is not needed or that it is, depending on the direction). E2 "works" in the paper's sense if its real Sim2 @25 is within 0.5 of `noise_clean`. Nothing is tuned after reading a result.

## Commands (copy exactly; real address, `cd` first)

PowerShell on your PC (upload the changed files, one command per line):
```powershell
cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM
scp src\mcno_qsm\artifacts.py src\mcno_qsm\rc1_data.py src\mcno_qsm\ahead_data.py src\mcno_qsm\rc1_train.py igna@100.125.57.28:~/mcno/MCNO-QSM/src/mcno_qsm/
scp configs\s4b_paper_e1_rc1.toml configs\s4b_paper_e1_ahead1.toml configs\s4b_paper_e2_geo.toml configs\s4b_paper_e3_ft.toml configs\s4b_paper_e4_l2only.toml configs\s4b_paper_ws.toml configs\s4b_paper_e1_rc1_ws.toml igna@100.125.57.28:~/mcno/MCNO-QSM/configs/
scp scripts\export_paper_tests.py scripts\make_paper_configs.py scripts\after_paper.sh scripts\rescore_fixed.py igna@100.125.57.28:~/mcno/MCNO-QSM/scripts/
scp tests\test_paper_recipe.py igna@100.125.57.28:~/mcno/MCNO-QSM/tests/
ssh igna@100.125.57.28
```
Server (after `ssh`):
```bash
cd ~/mcno/MCNO-QSM
sed -i 's/\r$//' scripts/after_paper.sh
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
pgrep -af "mcno_qsm.train"
.venv/bin/python -m pytest -q tests
nohup bash scripts/after_paper.sh > runs/after_paper.log 2>&1 &
tail -2 runs/after_paper.log
```
`after_paper.sh` trains `paper_e1_rc1`, `paper_e1_ahead1`, `paper_e2_geo`, `paper_e3_ft` one after another (E3 needs E2), then scores them and `noise_clean` and packs `~/mcno/results_paper.tgz`.
Optional afterwards (same launcher, name the runs): `nohup bash scripts/after_paper.sh paper_ws paper_e4_l2only paper_e1_rc1_ws > runs/after_paper2.log 2>&1 &`.
Do not edit a script while it runs. Resume after a reboot: run the same command again (finished runs are skipped, interrupted ones resume).
