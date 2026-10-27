# TO-DO for 2026-10-26 (written 2026-10-25)

Read this first, then `CLAUDE.md` §1-8 and §9bn (state of the art). Two tasks: (A) compact CLAUDE.md, (B) replicate everything the
Deep Fourier-Space Inversion paper tried (needs the GPU, which is free tomorrow). Do A first: it needs no GPU and can be done while
the server is still busy.

## A. Compact CLAUDE.md (1085 lines, 223 KB: too long to read in one pass; the tail was never read in full on 2026-10-25)

Rules: lose NO number, NO success criterion fixed beforehand, NO caveat that limits a claim. Keep the section IDs (9a ... 9bn), because
memory notes, scripts and the user refer to them (e.g. "9an", "9ap", "9bg").
1. Copy the current file verbatim to `docs/CLAUDE_archive_2026-10-25.md` (create `docs/`). Do not edit the archive afterwards.
2. Rewrite `CLAUDE.md` to about 250-350 lines:
   - Keep as they are: §1 goal and honest limits, §3.4 units, §6 eNDI scoring, §8 server procedure (verified commands), §9 working rules.
   - Condense §2, §3.3, §4, §5, §7, §10 to what is still true (the superseded text is in the archive; say so in one line each).
   - Replace the chronological 9a-9bn by a RESULTS LEDGER: one table per topic (data and families, architecture variants, losses, cascade /
     regime / filters, calcification, mask and rim, iteration counts) with columns: id, what was run, setting (seed, steps, head), key numbers
     (@25 / @100 fixed eNDI iterations, real Sim2, exp7-9), verdict (positive / null / negative), caveat, archive section.
   - A "settled, do not repeat" list (band loss, segment loss, local conv, U-Net, masked norm, AHEAD anatomy, calcification family, clipping, TV test ...)
     with the one-line reason and the section id.
   - A "decision rules and criteria" list (the rules fixed before results: 9am, 9an, 9ar, 9bd, 9bm ...), and the current baselines table
     (dirty, low-pass, oracle, noise, noise_clean, A12, AB, regime, B_rim2) with the iteration count of each number.
   - "Open items" (final model on the full RC1 head, AHEAD license, technical report without results after 9o, background-remnant family).
3. Show the user the compacted file (diff of headings and a check that every number of the last 5 sections is present: `grep` the archive
   numbers against the new file) BEFORE it replaces the old one. Update the pointer in `MEMORY.md` if the "read first" section changes.

## B. Replicate what the paper tried (Lambert, Silva, Milovic, Tejos; ISMRM 2023, abstract 4171)

Context: the paper trains a network that gets the dipole kernel as an input channel, from one brain or from spheres and cylinders only, and tests
on an abdomen phantom and on Sim2. We do NOT adopt their inversion (the plan stays phase cleaning + eNDI); we copy their TRAINING RECIPE and
tests onto our cleaner. Every run: `noise_clean` recipe as the reference (seed 42, 6000 steps, batch 2, 96^3 patches, 67k-parameter
`PhaseCleaner`, same RC1 check slab), only the data/augmentation changes; cap `MCNO_VRAM_GB=10`; judge by eNDI at 25 AND 100 fixed iterations
(`scripts/rescore_fixed.py`, `scripts/rerun_100.py`), real Sim2 and the rim variant first.
Write the success criteria into CLAUDE.md BEFORE the results (proposal, user to confirm): a run counts as better only if real Sim2 @25 is at
least 0.5 below noise_clean (47.31) or the weight-scale test (below) improves by more than the 9bf spread, AND forward Sim2 fields and RC1 exp1
are not worse than noise_clean by more than 0.5. Single seed is labeled single seed.

New code needed (all default-off keys, old configs byte-identical, tests for each; `artifacts.py`, `rc1_data.py`, `scripts/make_s4b_configs.py`):
1. **E1 single-brain training**: only ONE head (RC1 head, and separately one AHEAD subject), families noise + clean, the paper's augmentations:
   susceptibility scaling (random factor on chi, field recomputed), axis permutation (we already have the 24 cube rotations: check that
   z can move to x/y, 9f), SNR 30-300, ISOTROPIC VOXEL SIZE 1-3 mm (kernel and k-features per sample; batches share one voxel size). Test on Sim2.
2. **E2 geometric figures only**: no brain; 64^3 volumes of random cylinders and spheres in chi (30,000 volumes or generated online),
   forward by the dipole kernel, complex noise SNR 30-120, 15000 steps. Needs a new chi generator (`artifacts.geometric_chi`), no head.
3. **E3 fine-tune** of E2 on brains (our AHEAD subjects replace their two synthetic T1 brains): chi value-mapping, axis permutation, RANDOM
   WEIGHTING of the paramagnetic and diamagnetic parts of chi (positive and negative parts scaled independently), SIGN INVERSION of chi,
   noise SNR mixture 30-50 (30% of the draws) and 90-150 (70%).
4. **E4 loss ablation**: their loss is MSE only; ours is masked L1 + L2 (expect null, 9p-ter / 9ak; cheap, run only if the GPU is free).
5. **Their tests on our side**: Sim2 real and forward (done tables), forward simulations from COSMOS at SNR 40, 100, 100 with the last one
   containing STRONG PHASE INCONSISTENCY (wraps of +-pi): build as a new RC1 experiment, report the phase NRMSE of the cleaner (wraps are
   invisible to eNDI because of sin, so the eNDI number alone says nothing), and the eNDI number as a side note.
6. **Weight-scale test** (from the 9bf conditions): the nets break when the weight channel leaves 0.13-0.38 (9u). Add random scaling of the weight
   channel with the noise tied to it (noise proportional to 1/weight) as an augmentation key, and run it as E1-E3 side option on the reference
   recipe. Compare with the 9bf weight-scale table (`runs/scale_robustness/table.txt`).
7. **OPTIONAL, ask the user first (memory: no new inversion methods as the plan)**: their input representation (Re/Im of the FFT of the field +
   the kernel as 3 input channels) as an alternative cleaner front end; ablation only.
Not available: the abdominal QSM phantom (Silva et al., ISMRM 2022) is not in the repo; the 2016-challenge COSMOS data are RC1.

Order: (1) A, (2) unit tests + the augmentation keys of E1/E3/6 on the laptop, (3) dry-run, (4) on the server: E1, E2, E3 sequentially via a new
`scripts/after_paper.sh` (never edit a running script, 8), (5) rescoring at 25/100, (6) summary table + section 9bo in the new CLAUDE.md.
Estimated GPU time: about 2 h per 6000-step run (E2 15000 steps about 4-5 h); the GPU is shared, check `nvidia-smi` first.

## Reminders
- Command blocks: separate "PowerShell on the PC" and "Server" blocks, `cd` first, real address `igna@100.125.57.28` (CLAUDE.md §8).
- Server project: `~/mcno/MCNO-QSM`, call `.venv/bin/python`, `nohup`, no tmux; run `pytest` on the server before long runs.
- Report faithfully: nulls are results; label single seed / one head / one Sim2 volume.
