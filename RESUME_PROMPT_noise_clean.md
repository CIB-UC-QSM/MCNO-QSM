# Prompt for the next session: `noise_clean` result (paste into Claude Code, working dir `MCNO-QSM`)

Written 2026-10-22 at the end of the session. The text between the lines is the prompt; the commands below it are for me to run first.

---

```markdown
Read `CLAUDE.md` completely first (especially §8 server procedure, §9u-§9x), then the technical report only if needed. Do not re-derive what CLAUDE.md settles.
No commits unless I ask. Report negative results plainly. Ask before creating numbered variants of an experiment.

## Command-block rule
Whenever I must run something, give TWO separate, labeled blocks: (A) "PowerShell on my PC" (starts with `cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM`,
real address `igna@100.125.57.28`, each scp on ONE line) and (B) "Server" (after `ssh igna@100.125.57.28`, project `~/mcno/MCNO-QSM`, always `.venv/bin/python`, long jobs with `nohup`, never tmux).
No trailing `# comments` in pasted commands (zsh passes them as arguments). Never edit a script on the server while a queue runs it.

## Where we are (2026-10-22)
- User's verdict: cleaning must be judged by eNDI at a FIXED iteration count (25), not with the auto-stop; the `noise` net does not generalize even on its training head (it damages near-clean fields).
- Fixed-iteration results (CLAUDE.md 9w, `runs/generalization/fixed_meta.json`, notebook `notebooks/generalization_paper_experiments.ipynb` section B2): at 25 iterations real Sim2 dirty 53.9 / noise 47.3 / mix 50.0 (oracle 43.1);
  RC1 exp1 (noiseless) dirty 35.8 / noise 39.7 / mix 40.6 (the cleaner costs 4-5 points); exp2 (SNR 100) 39.4 / 39.2 / 46.0; exp3 61.2 / 43.1 / 49.3; exp5 cylinders 94.1 / 124.1 / 47.7; exp7-9 real residuals mix 63.9 / 55.7 / 66.7 vs dirty 93.6 / 67.7 / 83.3.
  Dirty fields diverge with iterations; cleaned fields keep improving.
- Test bugs found and fixed (9u/9v): the weight channel's SCALE matters to the nets (RC1 `magn` mean 0.26); rescaling Sim2's weight to 0..1 broke them.
- THE RUN IN PROGRESS: `noise_clean` (CLAUDE.md 9x): noise recipe + 20% identity examples (dirty = clean) + SNR 40-200, seed 42, 6000 steps. Launched on the server with `nohup bash scripts/after_clean.sh > runs/after_clean.log 2>&1 &`
  (train -> clean the generalization fields with `noise` and `noise_clean` into `runs/generalization_nc` -> fixed-iteration rescoring `runs/rescore_fixed_nc.log` -> pack `~/mcno/results_nc.tgz`). The 87 tests were NOT yet run on the server (the pasted command had a `#` comment; run them).
- Success criteria decided BEFORE seeing the results (compare with `noise` at 25 iterations): exp1 not worse than about 37; real Sim2 not worse than about 47.8; exp3 and the Sim2 forward fields not worse than `noise` by more than about 1.
  If not met: discuss new ideas with me (candidates: weight-channel scale augmentation, larger network or two heads, a gain-map output).

## Tasks for this session
1. I will extract `results_nc.tgz` (or paste `runs/rescore_fixed_nc.log`). Read `runs/generalization_nc/fixed_meta.json` and `runs/s4b_noise_clean/eval.csv`; build the same table as 9w (dirty / noise / noise_clean at 10, 25, 40 iterations and min), check the success criteria honestly, and also look at the clean-example damage and the RC1 check-slab ratios in eval.csv.
2. Write a new CLAUDE.md section with the numbers; add noise_clean to the notebook section B2 if useful (rebuild with `python -I notebooks/build_generalization_notebook.py`).
3. Recommend the next step (a recommendation, not a survey). Keep visible: one head, one Sim2 volume, single seed, jumps and phase spheres are invisible to this eNDI (they say nothing about calcification streaking), the 'oracle' is not a floor.
```

---

## Recovery: commands for me to run first

**(B) Server** — after `ssh igna@100.125.57.28` (type each line without trailing comments):
```bash
cd ~/mcno/MCNO-QSM
.venv/bin/python -m pytest -q tests
pgrep -af "after_clean|mcno_qsm.train|export_generalization|rescore_fixed"
tail -3 runs/after_clean.log
tail -3 runs/s4b_noise_clean.log
ls runs/s4b_noise_clean
ls -la ~/mcno/results_nc.tgz
```
- `pytest` must end with `87 passed` (not run yet on the server).
- `after_clean.log` ends with `after_clean finished <date>` when everything is done; `ls runs/s4b_noise_clean` shows `DONE` when training finished (`FAILED` = crashed, `cat runs/s4b_noise_clean/FAILED`).
- If nothing is running and `results_nc.tgz` is missing (server rebooted): relaunch the same chain, it skips finished steps and resumes training from `last.pt`:
```bash
cd ~/mcno/MCNO-QSM
nohup bash scripts/after_clean.sh > runs/after_clean.log 2>&1 &
```
- If training is DONE but the later steps failed, run them by hand:
```bash
cd ~/mcno/MCNO-QSM
MCNO_VRAM_GB=10 .venv/bin/python scripts/export_generalization.py --runs noise noise_clean --endi none --force --out runs/generalization_nc > runs/generalization_nc.log 2>&1
MCNO_VRAM_GB=10 .venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_nc --methods dirty noise noise_clean --real-runs noise noise_clean > runs/rescore_fixed_nc.log 2>&1
tar czf ~/mcno/results_nc.tgz runs/s4b_noise_clean/eval.csv runs/s4b_noise_clean/metadata.json runs/s4b_noise_clean/DONE runs/generalization_nc/fixed_meta.json runs/generalization_nc.log runs/rescore_fixed_nc.log runs/s4b_noise_clean.log
```

**(A) PowerShell on my PC** — bring the results:
```powershell
cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM
scp igna@100.125.57.28:~/mcno/results_nc.tgz .
tar -xzf results_nc.tgz
```
Optional, to look at the cleaned fields of the new net in the notebook (about 1 GB; skip unless needed): `scp -r igna@100.125.57.28:~/mcno/MCNO-QSM/runs/generalization_nc .\runs\`
