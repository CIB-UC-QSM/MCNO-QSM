# Prompt for the next session: S4b results (paste into Claude Code, working dir `MCNO-QSM`)

Written 2026-10-20 at the end of a long session. The text between the lines is the prompt; the recovery commands below it are for me to run first.

---

```markdown
Read `CLAUDE.md` completely first (especially §8 server procedure, §9d-§9j), then `technicalReport/report.tex` (sections 3-5 and 7). Do not re-derive what CLAUDE.md settles.
No commits unless I ask. Report negative results plainly. Ask before creating numbered variants of an experiment.

## Command-block rule
Whenever I must run something, give TWO separate, labeled blocks: (A) "PowerShell on my PC" (starts with `cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM`,
real address `igna@100.125.57.28`, each scp on ONE line) and (B) "Server" (after `ssh igna@100.125.57.28`, project `~/mcno/MCNO-QSM`, always `.venv/bin/python`, long jobs with `nohup`, never tmux).
Never edit `scripts/run_s4b.sh` on the server while a queue runs (bash reads scripts incrementally).

## Where we are (2026-10-20)
- S4a done (RC1 loader, 9 artifact families, anatomy augmentation, family weights, per-volume voxel size, hold-out axis 0, nested checkpointing; 76 tests pass on laptop and server).
- S4b queue on the server (`runs/s4b_all.log`): combo (done), mix (done), noise, spheres_chi, jumps, tissue, sti_offdiag, spheres_phase, cylinders; then a waiter job runs mix_seed43 and mix_seed44 (`runs/s4b_seeds.log`).
  About 1 h per run. Training is on RC1 only; Sim2 is monitored only; checkpoint chosen on the RC1 check slab only.
- Results so far (CLAUDE.md §9i, §9j): combo alone is slightly WORSE than the dirty field on Sim2 (eNDI 53.34% vs 52.60%); the MIX is better than dirty (eNDI 48.78% vs 52.60% masked;
  47.74% vs 50.87% in the unmasked protocol; XSIM 0.547 vs 0.497) and about equal to the low-pass sigma 0.5 (49.03%, XSIM 0.525), single seed.
- Technical report (`technicalReport/report.pdf`) still says no S4b result is available: update section 7 after the queue and the seeds finish.

## Tasks for this session
1. I will paste `runs/s4b_summary.txt` (and maybe `cat runs/s4b_all.log runs/s4b_seeds.log`). Interpret it honestly: which families matter (per-family runs vs mix), the seed spread of the mix
   (3 seeds: is the gain over dirty and over the low-pass larger than the spread?), whether the RC1 selection tracks the Sim2 monitoring curve, per-family check-slab ratios.
2. Update CLAUDE.md (new section with all S4b numbers) and the technical report's results section (+ rebuild the PDF with `latexmk -pdf report.tex` in `technicalReport/`).
3. Propose the next step with a recommendation, not a survey. Candidates: (a) diagnosis of what the model removes on Sim2 (spectrum of the removed component vs the true psi, per band);
   (b) eNDI score on the RC1 check slab; (c) S4c: add AHEAD heads (CLAUDE.md §9e; open question: is the NIfTI affine in scanner coordinates?); (d) S5 ablations (single segment vs Dipole-let, k-conditioning
   with real P_s because the imaginary parts are inert without it, magnitude input); (e) longer/shorter training or per-eval checkpoints for post-hoc diagnosis (never for selection).
4. Keep the open caveats visible: one training head, single Sim2 volume, auto-stop iteration differs (53 vs 16-22), the low-pass baseline was tuned on the target.
```

---

## Recovery: commands for me to run first (I paste the outputs into the new session)

**(B) Server** — after `ssh igna@100.125.57.28`:
```bash
cd ~/mcno/MCNO-QSM
cat runs/s4b_all.log runs/s4b_seeds.log
pgrep -af "mcno_qsm.train|run_s4b"          # empty = everything finished
for d in runs/s4b_*/; do [ -f $d/DONE ] && echo "$d DONE" || echo "$d NOT DONE"; done
```
If something is NOT DONE and nothing is running (server rebooted): `nohup bash scripts/run_s4b.sh mix_seed43 mix_seed44 > runs/s4b_seeds.log 2>&1 &` resumes/skips as needed (list the missing names as arguments; finished runs are skipped, interrupted ones resume from `last.pt`).

When everything is DONE, score all runs with eNDI (about 5 min each, GPU) and make the table:
```bash
cd ~/mcno/MCNO-QSM
nohup bash scripts/score_s4b.sh > runs/s4b_scoring.log 2>&1 &
tail -3 runs/s4b_scoring.log                 # wait until it says "scoring finished"
.venv/bin/python scripts/summarize_s4b.py | tee runs/s4b_summary.txt
```
Upload `scripts/score_s4b.sh` and `scripts/summarize_s4b.py` first if they are not on the server (they are new; do NOT touch `run_s4b.sh`).

Pack the small result files to bring them to the PC (no big `.nii.gz`):
```bash
cd ~/mcno/MCNO-QSM
tar czf ~/mcno/s4b_results.tgz $(find runs -maxdepth 2 -path 'runs/s4b_*' \( -name '*.csv' -o -name '*.json' -o -name DONE \)) runs/s0/score.json $(ls runs/*.log) runs/s4b_summary.txt
ls -la ~/mcno/s4b_results.tgz
```

**(A) PowerShell on my PC**:
```powershell
cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM
scp igna@100.125.57.28:~/mcno/s4b_results.tgz .
tar -xzf s4b_results.tgz
```
Upload for the first time (before the server steps above, if not yet done):
```powershell
cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM
scp scripts\score_s4b.sh scripts\summarize_s4b.py igna@100.125.57.28:~/mcno/MCNO-QSM/scripts/
```
