# RESUME: paper-recipe replication (9bo), results extraction and scoring (written 2026-10-25)

Read `CLAUDE.md` first (STATE OF THE ART block, ledger L8, DECISION RULES entry "9bo ... CONFIRMED"), then `docs/paper_recipe_plan.md`
(runs, our choices, limits, criteria, commands). Memory note: `project_paper_replication.md`.

## Where things stand
`nohup bash scripts/after_paper.sh > runs/after_paper.log 2>&1 &` was started on the server on 2026-10-25 (GPU shared, cap `MCNO_VRAM_GB=10`).
It trains `paper_e1_rc1`, `paper_e1_ahead1`, `paper_e2_geo` (15000 steps, 64^3), `paper_e3_ft` (fine-tune of the E2 `best.pt`), then:
`export_generalization.py` (runs/generalization_paper) -> `rescore_fixed.py --iters 100` (fixed_meta.json, log `runs/rescore_fixed_paper.log`)
-> `export_paper_tests.py` (runs/generalization_paper_tests, `runs/paper_tests.log`) -> `rescore_fixed.py` on it (`runs/rescore_fixed_paper_tests.log`)
-> `test_scale_robustness.py` (`runs/scale_robustness_paper/table.txt`) -> `~/mcno/results_paper.tgz`. The last line of `runs/after_paper.log`
is `after_paper finished <date>` when everything is done. At 15:28 -03 on 2026-10-25 E1 (RC1 head) and E1 (AHEAD subject) were DONE and E2 had just started.
Optional runs, only after the main chain and only if the user wants them: `paper_ws`, `paper_e4_l2only`, `paper_e1_rc1_ws`
(`nohup bash scripts/after_paper.sh paper_ws paper_e4_l2only paper_e1_rc1_ws > runs/after_paper2.log 2>&1 &`; their tarball is again `~/mcno/results_paper.tgz`, so rename the first one before).

## Commands for the user (separate windows; real address)

Server, to check that it is finished (run first; nothing else needed if it does not say finished):
```bash
cd ~/mcno/MCNO-QSM
tail -2 runs/after_paper.log
pgrep -af "mcno_qsm.train"
ls ~/mcno/results_paper.tgz
```

Server, ONE block that prints everything the assistant needs (paste the whole output into the chat):
```bash
cd ~/mcno/MCNO-QSM
echo "##### after_paper.log"; tail -15 runs/after_paper.log
for n in paper_e1_rc1 paper_e1_ahead1 paper_e2_geo paper_e3_ft paper_ws paper_e4_l2only paper_e1_rc1_ws; do [ -d runs/s4b_$n ] && { echo "##### $n"; ls runs/s4b_$n | tr '\n' ' '; echo; cat runs/s4b_$n/DONE 2>/dev/null; cat runs/s4b_$n/FAILED 2>/dev/null; head -1 runs/s4b_$n/eval.csv; tail -2 runs/s4b_$n/eval.csv; }; done
echo "##### rescore_fixed_paper.log"; cat runs/rescore_fixed_paper.log
echo "##### paper_tests.log"; cat runs/paper_tests.log
echo "##### rescore_fixed_paper_tests.log"; cat runs/rescore_fixed_paper_tests.log
echo "##### scale_robustness_paper"; cat runs/scale_robustness_paper/table.txt
echo "##### generalization_paper.log (tail)"; tail -5 runs/generalization_paper.log
```
(If the output is too long for one paste, split it at the `#####` lines. The `rescore_fixed_paper.log` lines read like
`rc1 exp3 noise_clean  @10: .. @25: .. @40: .. @100: .. min ..% @i  XSIM@25 ..`; `@25` is the judge.)

PowerShell on the PC, to bring the small result files to the laptop (optional, for the notebook and the files; the paste above is enough for the scoring):
```powershell
cd C:\Users\ignac\OneDrive\Desktop\Projects\Research\2025\QSM\DipoleNet\MCNO-QSM
scp igna@100.125.57.28:~/mcno/results_paper.tgz .
tar -xzf results_paper.tgz
```

## What the assistant does with it (in this order)
1. Score against the criteria CONFIRMED in CLAUDE.md (9bo entry): compare with the `noise_clean` row of the SAME `rescore_fixed_paper.log`
   (do not use a quoted 47.31/47.48). Per run: real Sim2 (`sim2 sim2_real`) @25 and @100; forward Sim2 fields and RC1 exp1 @25 (no-harm, 0.5);
   weight-scale ratios at x2.8 from `scale_robustness_paper/table.txt` (needs the `sim2_snrcal` and real Sim2 rows of `noise_clean` there too);
   paper tests phase NRMSE (`paper_tests.log`: snr40 / snr100 not worse than `noise_clean` by more than 1.0; `paper_snr100_wrap` reported,
   plain and wrap-invariant); RC1 check-slab `sel_trained` and the AHEAD validation columns from `eval.csv` as context only.
2. Verdict per run in the project's terms (positive / null / negative), with the caveats: single seed, one RC1 head, one Sim2 volume, AHEAD is
   processed QSM, E2 is judged on the RC1 check slab and Sim2 only. Report nulls as results. Nothing is re-tuned after reading.
3. Write the result as CLAUDE.md section "9bo-bis" (a short paragraph with the table, in the ledger style; update ledger L8 status from
   "pending" to the verdicts and the STATE OF THE ART item 8), update `project_paper_replication.md` and the `MEMORY.md` line, and tell the user.
4. Ask before anything new: optional runs (ws, E4, E1+ws), the Re/Im-FFT front end (item 7), a second seed of any run that looks positive,
   and what to do next (CLAUDE.md OPEN ITEMS: final model on the full RC1 head, in-vivo validation, write-up).

## Reminders
- Commands to the user: separate "PowerShell on the PC" and "Server" blocks, `cd` first, real address `igna@100.125.57.28`, one command per line.
- Server project `~/mcno/MCNO-QSM`, `.venv/bin/python`, `nohup`, no tmux, never edit a running script, no `sleep` in commands for the user.
- Judge by eNDI at 25 AND 100 fixed iterations; the auto-stop is not a judge. Do not propose new inversion methods as the plan.
- Windows-edited files are CRLF; bash scripts must be LF (`sed -i 's/\r$//'` on the server after scp).
