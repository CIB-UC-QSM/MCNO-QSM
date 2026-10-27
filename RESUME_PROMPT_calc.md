# Prompt for the next session: `noise_clean_calc` result (paste into Claude Code, working dir `MCNO-QSM`)

Written 2026-10-23 at the end of the session. The text between the lines is the prompt; the commands below it are for me to run first.

---

```markdown
Read `CLAUDE.md` completely first (especially section 8 server procedure and sections 9ad-9an), then the technical report only if needed. Do not re-derive what CLAUDE.md settles.
No commits unless I ask. Report negative results plainly. Ask before creating numbered variants of an experiment.

## Command-block rule
Whenever I must run something, give TWO separate, labeled blocks: (A) "PowerShell on my PC" (starts with `cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM`,
real address `igna@100.125.57.28`, each scp on ONE line) and (B) "Server" (after `ssh igna@100.125.57.28`, project `~/mcno/MCNO-QSM`, always `.venv/bin/python`, long jobs with `nohup`, never tmux).
No trailing `# comments` in pasted commands. Never edit a script on the server while a queue runs it. Before a long launch give `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` (the GPU is shared; a launch can die at once for lack of memory; a stale FAILED file stays in a run folder after a successful relaunch: trust DONE and eval.csv).

## Where we are (2026-10-23)
- Everything tried on the noise recipe lands within about 0.5 eNDI points of `noise_clean` (real Sim2 47.31 at 25 fixed iterations, dirty 53.94, oracle 43.10): AHEAD anatomy, band loss, Dipole-let segment-sensitivity loss, local conv, U-Net. The field-domain noise task is at an information floor (train/held-out ratio 0.466/0.449). Capacity is not the limit (localized artifacts are removed to 97% when trained on). Sim2's calcification problem is a ~3 mm shell where the INPUT keeps only 32% of the field.
- The run in progress: `noise_clean_calc` (noise_clean recipe + `calc_shell` family, CLAUDE.md 9am), launched with `nohup bash scripts/after_calc.sh > runs/after_calc.log 2>&1 &`; it writes `~/mcno/results_calc.tgz` (eval.csv, cleaned Sim2 volumes, fixed-iteration tables, `chi_fixed25.npz`).
- Success criteria fixed BEFORE the result (9am): (1) shell within 3 mm of the 37 calcification voxels: relative field error < 0.8 (dirty 1.03, noise_clean 0.99) and gain > 0.5 (0.32 / 0.29); (2) calcification-ROI eNDI NRMSE (Chi < -0.5 dilated 8 mm, 25 iterations) below noise_clean's, ideally below dirty's (old auto-stop maps: dirty 75.2, noise 76.9, oracle 68.3); (3) real Sim2 not above 47.6, RC1 exp1 not above 36.5, forward fields within 0.3.
- DECISION RULE (9an): if (1) or (2) fails, stop varying the field-domain recipe and change the problem: R1 unrolled-eNDI task loss against chi, R2 plug-and-play (cleaner interleaved with eNDI), R3 cascade (detect/inpaint sources, then the noise net), R4 richer input (four echo magnitudes). I want a recommendation, not a survey, and I decide.
- Caveat to state with any result: the family ranges are modelled on Sim2's own calcification, so a Sim2 gain is not an independent test; one seed, one volume.

## Tasks for this session
1. I will extract `results_calc.tgz`. Read `runs/generalization_calc/fixed_meta.json`, `runs/s4b_noise_clean_calc/eval.csv`, `runs/chi_fixed25.npz` and the cleaned volumes; check criteria (1)-(3) honestly (shell gain and error: the analysis of CLAUDE.md 9al on `best_clean.nii.gz`; ROI NRMSE: the cell of section H of the notebook).
2. Rebuild the notebook (`python notebooks/build_performance_notebook.py`) and the HTML report (`python notebooks/render_performance_report.py`); write the CLAUDE.md section with the numbers.
3. Apply the decision rule and recommend the next step.
```

---

## Recovery: commands for me to run first

**(B) Server** (after `ssh igna@100.125.57.28`):
```bash
cd ~/mcno/MCNO-QSM
tail -3 runs/after_calc.log
ls runs/s4b_noise_clean_calc
tail -2 runs/s4b_noise_clean_calc.log
tail -2 runs/rescore_fixed_calc.log
tail -3 runs/chi_fixed25.log
ls -la ~/mcno/results_calc.tgz
date
```
- Finished = `after_calc finished <date>`, `DONE` in the run folder, `rescore finished`, `wrote runs/chi_fixed25.npz`, and a `results_calc.tgz` newer than the end of training. If nothing is running and the tgz is missing (reboot), relaunch the same chain: it skips finished steps and resumes training from `last.pt`.

**(A) PowerShell on my PC**:
```powershell
cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM
scp igna@100.125.57.28:~/mcno/results_calc.tgz .
tar -xzf results_calc.tgz
```
