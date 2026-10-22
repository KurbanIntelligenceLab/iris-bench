"""
Run main.py's baseline (src/models/PhysModels.py blocks) on Delfys75, clip by clip,
from a pinned code snapshot, recording failures instead of writing zeros.

What it calls: <code_dir>/main.py execute_experiment(path, dynamics, experiment_name,
dt, loss_name) and get_dynamics(path), i.e. the released training code unchanged.
What the wrapper changes, and why:
  * an exception is recorded as status=failed with its message (main.py writes
    alpha=beta=...=0, which entered the published MAE as a real estimate);
  * Results/<experiment>/best_model.pt is deleted before each clip, so a clip whose
    loss is NaN before its first checkpoint cannot silently load the previous clip's
    model (main.py loads whatever best_model.pt exists);
  * clips are processed in sorted order (main.py uses os.walk order);
  * --init_phys overrides the hard-coded init_phys=10.0 (sensitivity runs only);
  * --seed N (N != 42) replaces the hard-coded torch.manual_seed(42) in execute_experiment
    (seed runs only; the 1500-window subsample keeps main.py's fixed default_rng(42));
  * the per-epoch parameter log returned by train.train is saved per clip;
  * status is ok / failed / nonfinite / not_trained (train.train returned the initial
    parameters) / nan_during_training (best-so-far parameters before a NaN loss).
    Only "ok" clips enter the MAE; the others are counted in n_total.

Usage (cwd is irrelevant; the wrapper chdirs into <out_dir>/work):
  python repro_d75/run_baseline.py --code_dir CODE --data_root DATA --dt 0.1 \
      --loss latent_loss --out_dir RUN_DIR [--init_phys 5] [--only pendulum/pendulum_45/01] [--resume]
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import sys
import time
import traceback

FIELDS = ["dynamics", "setting", "run", "npy", "n_samples", "dt", "status", "error", "alpha", "beta",
          "max_z", "min_z", "z0", "z1", "alpha_init", "beta_init", "alpha_last", "beta_last",
          "epochs_logged", "checkpoint_saved", "seconds"]


def dt_resolver(dt_arg, data_root):
    """--dt NUMBER: same dt for every clip. --dt protocol: per-dynamics dt from
    <data_root>/protocol.json written by video2npy_fixed.py (stride / 60)."""
    if dt_arg != "protocol":
        dt = float(dt_arg)
        return lambda dynamics: dt
    with open(os.path.join(data_root, "protocol.json")) as f:
        proto = json.load(f)["dynamics"]
    return lambda dynamics: float(proto[dynamics]["dt"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dt", required=True, help="seconds per frame step, or 'protocol' (per dynamics from protocol.json)")
    ap.add_argument("--loss", required=True, choices=["latent_loss", "latent_loss_multistep"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--experiment", default="d75", help="experiment_name passed to main.py (must not contain 'dropped_ball')")
    ap.add_argument("--init_phys", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42, help="replaces main.py's torch.manual_seed(42)")
    ap.add_argument("--only", action="append", default=[], help="dynamics/setting/run to include (repeatable)")
    ap.add_argument("--resume", action="store_true", help="skip clips already in clips.csv")
    args = ap.parse_args()

    if "dropped_ball" in args.experiment:
        sys.exit("experiment name must not contain 'dropped_ball' (train.py would switch to 1000 epochs)")
    code_dir = os.path.abspath(args.code_dir)
    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    clips_csv = os.path.join(out_dir, "clips.csv")
    if os.path.exists(clips_csv) and not args.resume:
        sys.exit(f"{clips_csv} exists; refusing to overwrite (use --resume to continue a run)")

    # main.py/train.py read ./config.yaml and write ./Results/<experiment>/ relative to cwd.
    work = os.path.join(out_dir, "work")
    os.makedirs(os.path.join(work, "Results", args.experiment), exist_ok=True)
    shutil.copy2(os.path.join(code_dir, "config.yaml"), os.path.join(work, "config.yaml"))
    os.chdir(work)
    sys.path.insert(0, code_dir)
    import numpy as np
    import main as M  # <code_dir>/main.py

    captured = {}
    orig_train = M.train.train

    def train_capture(*a, **k):
        out = orig_train(*a, **k)
        captured["log"] = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        return out

    M.train.train = train_capture

    if args.seed != 42:
        import torch
        orig_seed = torch.manual_seed

        def seed_override(_):
            return orig_seed(args.seed)

        torch.manual_seed = seed_override  # main.py calls torch.manual_seed(42) per clip

    if args.init_phys is not None:
        orig_endphys = M.mainmodel.EndPhys

        def endphys_override(*a, **k):
            k["init_phys"] = args.init_phys
            return orig_endphys(*a, **k)

        M.mainmodel.EndPhys = endphys_override

    dt_for = dt_resolver(args.dt, data_root)
    files = sorted(glob.glob(os.path.join(data_root, "*", "*", "*", "*.npy")))
    if args.only:
        files = [f for f in files if "/".join(os.path.relpath(f, data_root).split(os.sep)[:3]) in args.only]
    if not files:
        sys.exit(f"no .npy files selected under {data_root}")

    done = set()
    if args.resume and os.path.exists(clips_csv):
        with open(clips_csv) as f:
            done = {(r["dynamics"], r["setting"], r["run"]) for r in csv.DictReader(f)}
    new_file = not os.path.exists(clips_csv)
    traj_dir = os.path.join(out_dir, "param_logs")
    os.makedirs(traj_dir, exist_ok=True)
    best_path = os.path.join("Results", args.experiment, "best_model.pt")

    n_fail = 0
    with open(clips_csv, "a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for path in files:
            dyn_dir, setting, run = os.path.relpath(path, data_root).split(os.sep)[:3]
            if (dyn_dir, setting, run) in done:
                continue
            row = {k: "" for k in FIELDS}
            row.update(dynamics=dyn_dir, setting=setting, run=run, npy=path)
            print(f"=== {dyn_dir}/{setting}/{run}", flush=True)
            if os.path.exists(best_path):
                os.remove(best_path)
            captured.clear()
            t0 = time.time()
            try:
                row["n_samples"] = np.load(path, mmap_mode="r").shape[0]
                dynamics = M.get_dynamics(path)
                if dynamics != dyn_dir:
                    raise ValueError(f"path keyword gives dynamics={dynamics!r}, folder is {dyn_dir!r}")
                M.video_number += 1
                row["dt"] = dt_for(dynamics)
                _, vals = M.execute_experiment(path, dynamics, args.experiment, row["dt"], loss_name=args.loss)
                row.update(zip(["alpha", "beta", "max_z", "min_z", "z0", "z1"], vals))
                row["status"] = "ok" if all(math.isfinite(float(v)) for v in vals) else "nonfinite"
            except Exception as e:
                row["status"] = "failed"
                row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:500]
                traceback.print_exc()
            row["seconds"] = round(time.time() - t0, 1)
            row["checkpoint_saved"] = os.path.exists(best_path)
            log = captured.get("log")
            if row["status"] == "ok":
                # train.train returns the INITIAL parameters if the first evaluation raises,
                # and the best-so-far parameters if the loss turns NaN; neither is a clean fit.
                if not log or len(log) <= 1:
                    row["status"], row["error"] = "not_trained", "train.train returned before epoch 1 (estimate = init)"
                elif any(not math.isfinite(float(e.get("train_loss", 0.0))) for e in log):
                    row["status"], row["error"] = "nan_during_training", f"NaN loss after {len(log) - 1} epochs"
            if log:
                row["epochs_logged"] = len(log) - 1
                row["alpha_init"], row["alpha_last"] = log[0].get("alpha", ""), log[-1].get("alpha", "")
                row["beta_init"], row["beta_last"] = log[0].get("beta", ""), log[-1].get("beta", "")
                with open(os.path.join(traj_dir, f"{dyn_dir}__{setting}__{run}.json"), "w") as fj:
                    json.dump(log, fj)
            if row["status"] != "ok":
                n_fail += 1
            writer.writerow(row)
            fcsv.flush()
            print(f"    status={row['status']} alpha={row['alpha']} beta={row['beta']} "
                  f"({row['seconds']} s) {row['error']}", flush=True)

    # main.py-format copy with successful clips only, for the old comparison script.
    with open(clips_csv) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(out_dir, "mainpy_format_ok_only.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "alpha", "beta", "max_z", "min_z", "z0", "z1"])
        for r in rows:
            if r["status"] == "ok":
                w.writerow([r["dynamics"], r["setting"], r["run"], r["alpha"], r["beta"],
                            r["max_z"], r["min_z"], r["z0"], r["z1"]])
    total_fail = sum(r["status"] != "ok" for r in rows)
    print(f"Done: {len(rows)} clips in {clips_csv}; {total_fail} not ok.")
    sys.exit(3 if total_fail else 0)


if __name__ == "__main__":
    main()
