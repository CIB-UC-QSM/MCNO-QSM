# RESUME: intravoxel / acquisition-noise runs (9bp), results extraction and scoring (written 2026-10-25)

Read `CLAUDE.md` first (STATE OF THE ART item 8b, ledger L9, DECISION RULES entry "9bp ... PROPOSED", OPEN ITEMS 0 and 5). Memory note: `project_intravoxel.md`.

## Where things stand
On 2026-10-25 the user started on the server (without formally confirming the criteria; they stand as written before any result):
`nohup bash scripts/after_intravoxel.sh > runs/after_intravoxel.log 2>&1 &`   (GPU shared, cap `MCNO_VRAM_GB=10`)
It trains `iv_noisestat` (noise_clean recipe, SNR 15-400, four noise statistics, no sources) and `iv_full` (noise 1.0, clean 0.5, acq 0.5, intravoxel 0.5),
6000 steps each (guess: 2-3 h each, the intravoxel family is about 2.4x slower per patch on the laptop GPU), then:
`export_generalization.py` (runs/generalization_iv) -> `rescore_fixed.py --iters 100` (log `runs/rescore_fixed_iv.log`) -> `export_paper_tests.py`
(runs/generalization_iv_tests, `runs/iv_paper_tests.log`) -> `rescore_fixed.py` on it (`runs/rescore_fixed_iv_tests.log`) -> `export_iv_tests.py --set noise`
and `--set src` (runs/generalization_iv_noise, runs/generalization_iv_src; logs `runs/iv_tests_noise.log`, `runs/iv_tests_src.log`; eNDI in
`runs/rescore_fixed_iv_noise.log`, `runs/rescore_fixed_iv_src.log`) -> `test_scale_robustness.py` (`runs/scale_robustness_iv/table.txt`) ->
`test_clipping.py` (calcification ROI, `runs/clipping_iv/table.txt`, read the row `none`) -> `score_shell.py` (`runs/shell_iv.txt`) -> `~/mcno/results_iv.tgz`.
The last line of `runs/after_intravoxel.log` is `after_intravoxel finished <date>` when everything is done. A run that did not finish is listed as `NOT DONE`
and left out of the scoring.

## Commands for the user (separate windows; real address)

Server, to check that it is finished (run first; nothing else needed if it does not say finished):
```bash
cd ~/mcno/MCNO-QSM
tail -3 runs/after_intravoxel.log
pgrep -af "mcno_qsm.train"
ls ~/mcno/results_iv.tgz
ls runs/s4b_iv_noisestat runs/s4b_iv_full
```
If a run folder holds `FAILED`, send `cat runs/s4b_iv_full/FAILED | tail -20` and `tail -20 runs/s4b_iv_full.log` (an out-of-memory error at the 10 GB cap is the
likely cause: the source boxes need about 2 GB for generation on top of the 8.6 GB training peak). Do not edit a running script; resume an interrupted run with
`bash scripts/run_s4b.sh iv_full` (it passes `--resume` when `last.pt` exists).

Server, ONE block that prints everything the assistant needs (paste the whole output into the chat; split at the `#####` lines if too long):
```bash
cd ~/mcno/MCNO-QSM
echo "##### after_intravoxel.log"; tail -15 runs/after_intravoxel.log
for n in iv_noisestat iv_full; do [ -d runs/s4b_$n ] && { echo "##### $n"; cat runs/s4b_$n/DONE 2>/dev/null; cat runs/s4b_$n/FAILED 2>/dev/null; head -1 runs/s4b_$n/eval.csv | tr ',' '\n' | grep -n "sel_trained\|intravoxel\|acq" | head -8; tail -2 runs/s4b_$n/eval.csv | cut -d, -f1; }; done
echo "##### rescore_fixed_iv.log"; cat runs/rescore_fixed_iv.log
echo "##### iv_paper_tests.log"; cat runs/iv_paper_tests.log
echo "##### iv_tests_noise.log"; cat runs/iv_tests_noise.log
echo "##### rescore_fixed_iv_noise.log"; cat runs/rescore_fixed_iv_noise.log
echo "##### iv_tests_src.log"; cat runs/iv_tests_src.log
echo "##### rescore_fixed_iv_src.log"; cat runs/rescore_fixed_iv_src.log
echo "##### scale_robustness_iv"; cat runs/scale_robustness_iv/table.txt
echo "##### clipping_iv"; cat runs/clipping_iv/table.txt
echo "##### shell_iv"; cat runs/shell_iv.txt
```
(The `rescore_fixed_*.log` lines read like `rc1 exp3 noise_clean @10: .. @25: .. @40: .. @100: .. min ..% @i XSIM@25 ..`; `@25` is the judge, `@100` is
reported next to it.)

PowerShell on the PC, to bring the small result files to the laptop (optional, the paste above is enough for the scoring; the tarball keeps its `runs/` paths):
```powershell
cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM
mkdir results_iv
scp igna@100.125.57.28:~/mcno/results_iv.tgz results_iv\
tar -xzf results_iv\results_iv.tgz -C results_iv
```

## What the assistant does with it (in this order)
1. Score against the criteria in DECISION RULES "9bp" (read the exact text in CLAUDE.md; thresholds were fixed before any result), always against the
   `noise_clean` rows of the SAME logs, never against a quoted number:
   - (1) no harm, `iv_full` AND `iv_noisestat`: real Sim2 (`sim2 sim2_real`) @25 within 0.5; forward fields `sim2_snr100`, `sim2_snrcal`, `sim2_snr40`,
     `sim2_jumps`, `sim2_spheres` within 0.5; RC1 `exp1` within 0.5, `exp3` within 1.0 (`rescore_fixed_iv.log`); paper tests `paper_snr40`, `paper_snr100` phase NRMSE
     within 1.0 (`iv_paper_tests.log`).
   - (2) near sources (`iv_full`): `iv_src100`, `iv_src40` `near_source` phase NRMSE at least 5.0 points below `noise_clean` (`iv_tests_src.log`), OR eNDI @25 at least 1.0
     below (`rescore_fixed_iv_src.log`).
   - (3) transfer to Sim2: shell <= 3 mm relative error <= 0.89 and gain >= 0.40 (`shell_iv.txt`; noise_clean 0.986 / 0.29, dirty 1.029 / 0.32), AND/OR calcification-ROI eNDI @25
     at least 3.0 below noise_clean (`clipping_iv/table.txt`, row `none`, column calcROI).
   - (4) noise statistics: `iv_colored40`, `iv_heavy40`, `iv_hetero40` phase NRMSE at least 0.5 below noise_clean on at least two of three (`iv_tests_noise.log`);
     `iv_acq40` and the weight-scale ratios (`scale_robustness_iv/table.txt`, x2.8 rows; noise_clean 1.20 / 1.08) are reported without a criterion.
   - Read @100 next to @25; a verdict changes only if they disagree by more than 1.0 (say so).
2. Verdict per run in the project's terms (positive / null / negative) with the READING rules of 9bp, the caveats (single seed; one RC1 head; one Sim2 volume; the family is
   Sim2-informed through the echo train, the amplitude range and the realism check; `iv_src*` and `iv_*40` are in-distribution tests; Sim2's early-echo magnitude loss near the
   source is not reproduced by dephasing alone). Compare `iv_noisestat` with `iv_full` to separate the sweep/statistics from the sources. Report nulls as results. Nothing
   is re-tuned after reading.
3. Write the result as CLAUDE.md section "9bp-bis" (short paragraph with the table in the ledger style), update ledger L9 status from "pending", STATE OF THE ART item 8b,
   OPEN ITEMS 0, `project_intravoxel.md` and its `MEMORY.md` line, and tell the user.
4. Ask before anything new. The user's TO DO (CLAUDE.md OPEN ITEMS 5): IF (1) and (2) are met, the BACKGROUND-REMOVAL REMNANT family is next. Start from
   `utils/Dipole-lets_Paper_Plots_3.ipynb` cell 16 (air-like spheres chi 9.4 ppm outside the mask: frontal sinus radius 30, two ear-canal spheres radius 12, forward field
   `phase_bkg`), read the cells after it for the removal step, agree the design with the user, write the criteria into DECISION RULES BEFORE any run, calibrate on RC1 only.
   Other options to offer: adopt `iv_full` and rescore AB and the rim option on it (only if (1)+(2)+(3) are met), a second seed of a positive run, lower weights only after
   asking.

## Reminders
- Commands to the user: separate "PowerShell on the PC" and "Server" blocks, `cd` first, real address `igna@100.125.57.28`, one command per line.
- Server project `~/mcno/MCNO-QSM`, `.venv/bin/python`, `nohup`, no tmux, never edit a running script, no `sleep` in commands for the user.
- Judge by eNDI at 25 AND 100 fixed iterations; the auto-stop is not a judge. Core idea: clean the field, then simple eNDI; do not propose new inversion methods as the plan.
- On the laptop run tests with `.venv/Scripts/python.exe -m pytest -q tests` (174 pass at 2026-10-25); the venv torch is CPU-only, the global `python` has CUDA.
- Windows-edited files are CRLF; bash scripts must be LF (`sed -i 's/\r$//'` on the server after scp).
- The 9bo main chain (E1-E3, `results_paper.tgz`) is still not scored in this repo's notes; `paper_ws` was scored (NULL on accuracy) and is in CLAUDE.md L8.
