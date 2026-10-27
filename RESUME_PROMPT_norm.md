# Prompt: score the normalized-layers run (noise_clean_norm) against CLAUDE.md 9bd

Project folder: `C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM`. Read `CLAUDE.md` sections 9av (passive cascade, the baseline), 9az to 9be (gain network, diagnostics, feasibility, normalized layers, session state). The user is the author of the Dipole-lets paper and of eNDI: rigor, honest negatives, no mild variants of what failed, judge only by eNDI at 25 FIXED iterations, Sim2 is a test volume only, single seed / one head / one volume next to every number.

## State
The server job `nohup bash scripts/after_norm.sh` (train `noise_clean_norm`, scale-robustness check, `cascade_gate.py --norm-run noise_clean_norm`) was started on 2026-10-24. The criteria and reading rules are in CLAUDE.md 9bd and were fixed BEFORE the result. Do not change them.

## Server commands to get the results (the user runs them; two separately labelled blocks, never mixed)
Server (check it finished; `after_norm.log` must say "finished"):
```bash
cd ~/mcno/MCNO-QSM
tail -2 runs/after_norm.log
ls runs/s4b_noise_clean_norm
cat runs/s4b_noise_clean_norm/DONE
cat runs/s4b_noise_clean_norm/FAILED 2>/dev/null | tail -15
cat runs/scale_robustness/table.txt
cat runs/cascade_norm/table.txt
```
If `FAILED` exists with an out-of-memory error: the cap is 10 GB (`MCNO_VRAM_GB`); the first remedy is chunked or checkpointed inference, not a smaller model. If the job died after training, `bash scripts/run_s4b.sh noise_clean_norm` skips finished runs and the remaining steps of `after_norm.sh` can be run by hand.

PowerShell on the user's PC (only if files must be copied back): `cd` to the project folder first, real address `igna@100.125.57.28`, one-line `scp`.

## What to do with the output
1. Score `Bn` (column of `runs/cascade_norm/table.txt`) against the criteria (1) no harm, (2) scale robustness (from `runs/scale_robustness/table.txt`), (3) strong localized sources, with the numbers of 9bd; apply the reading rules; write the result as CLAUDE.md section 9bf (numbers, which criteria met or failed, caveats).
2. If (1) met and (2) or (3) met: the next step is the cascade with the normalized B (A then Bn), criteria as in 9av, to be pre-registered before running. Otherwise state the null/negative plainly and stop architecture variants on B.
3. Optionally check whether the training curve of `noise_clean_norm` (`eval.csv`, column `sel_trained`) is still falling at step 6000.
4. Give the user the next commands as separate PowerShell / Server blocks, and ask before anything long.
