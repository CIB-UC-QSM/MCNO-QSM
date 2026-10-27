"""Option 1 ablation: clean Sim2 with the trained U-Net branch switched off, and report how big its output is.

For each run (a trained `*_unet*` config) writes runs/<run>_nobranch/best_clean.nii.gz (cleaned Sim2, branch dropped)
so that `scripts/score_run.py --run runs/<run>_nobranch` gives the eNDI without the branch. Also prints
||branch output|| / ||v|| with the branch on (the share of the final features the hybrid part carries).
Run: .venv/bin/python scripts/ablate_unet.py --runs s4b_noise_unet s4b_noise_unet_seed43
"""

import argparse
import json
from pathlib import Path

import torch

from mcno_qsm.cleaner import PhaseCleaner
from mcno_qsm.rc1_train import Sim2Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--field-strength", type=float, default=7.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = Sim2Monitor(args.data, args.field_strength, device)
    report = {}
    for run in args.runs:
        state = torch.load(args.runs_dir / run / "best.pt", map_location=device, weights_only=False)
        config = state["config"]
        model = PhaseCleaner(state["voxel_size"], input_scale=float(state["input_scale"]),
                             dipolelet=config["dipolelet"], **config["model"]).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        if model.unet is None:
            print(f"{run}: no U-Net branch, skipped")
            continue
        out = args.runs_dir / f"{run}_nobranch"
        out.mkdir(exist_ok=True)
        model.unet_active = True
        with_branch = monitor.scores(model)
        ratio = model.last_branch_ratio
        model.unet_active = False
        without = monitor.scores(model)
        monitor.save(model, out / "best_clean.nii.gz")
        report[run] = {"branch_ratio": ratio, "sim2_phase_nrmse_with": with_branch["sim2_nrmse"],
                       "sim2_phase_nrmse_without": without["sim2_nrmse"], "step": state["step"]}
        print(f"{run}: branch/features {ratio:.3f}  Sim2 phase NRMSE with {with_branch['sim2_nrmse']:.4f} "
              f"without {without['sim2_nrmse']:.4f}", flush=True)
    (args.runs_dir / "ablate_unet.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
