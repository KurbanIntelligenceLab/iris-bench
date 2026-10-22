"""
Run scripts/run_unified_delfys75.py's unified model ("Corr." in paper Table 4) on
Delfys75, clip by clip, from a pinned code snapshot, recording failures.

What it calls, unchanged: load_video_tensor, train_one_video, config_for_dynamics and
get_dynamics_from_path from <code_dir>/scripts/run_unified_delfys75.py, with the config
built exactly as its run_pipeline() does (config_unified.yaml + dt, epochs, multistep).
What the wrapper changes, and why:
  * exceptions are recorded as status=failed (the script writes zeros);
  * torch/numpy are seeded with --seed before every clip (the script sets no seed,
    so the published unified runs are not reproducible run-to-run);
  * clips are processed in sorted order.
Behaviour kept as released and recorded in the output, not fixed:
  * load_video_tensor() returns only the FIRST 10-frame window of each .npy
    (data[0:1]), so each clip is fit on 10 frames. n_windows_in_file shows how many
    windows were available.

Usage:
  python repro_d75/run_unified.py --code_dir CODE --data_root DATA --dt 0.1 \
      --out_dir RUN_DIR [--multistep] [--seed 42] [--only dyn/setting/run] [--resume]
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
import traceback

FIELDS = ["dynamics", "setting", "run", "npy", "n_windows_in_file", "frames_used", "dt", "status", "error",
          "alpha", "beta", "max_z", "min_z", "z0", "z1", "seconds"]


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
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--multistep", action="store_true")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    code_dir = os.path.abspath(args.code_dir)
    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    clips_csv = os.path.join(out_dir, "clips.csv")
    if os.path.exists(clips_csv) and not args.resume:
        sys.exit(f"{clips_csv} exists; refusing to overwrite (use --resume to continue a run)")
    work = os.path.join(out_dir, "work")
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    sys.path.insert(0, os.path.join(code_dir, "scripts"))
    sys.path.insert(0, code_dir)
    import numpy as np
    import torch
    import yaml
    import run_unified_delfys75 as U  # <code_dir>/scripts/run_unified_delfys75.py

    # Same config construction as U.run_pipeline().
    with open(os.path.join(code_dir, "config_unified.yaml")) as f:
        config = yaml.safe_load(f)
    config["input_dim"] = config.get("input_dim", 5600)
    config["dt"] = None  # set per clip below (protocol may differ per dynamics)
    config["epochs"] = args.epochs
    config["use_multistep_loss"] = args.multistep
    if args.multistep:
        config.setdefault("multistep_num_steps", 5)
        config.setdefault("multistep_step_weights", [1.0, 1.0, 0.5, 0.5, 0.25])
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            t0 = time.time()
            try:
                row["n_windows_in_file"] = np.load(path, mmap_mode="r").shape[0]
                dynamics = U.get_dynamics_from_path(path)
                if dynamics != dyn_dir:
                    raise ValueError(f"path keyword gives dynamics={dynamics!r}, folder is {dyn_dir!r}")
                video = U.load_video_tensor(path).to(device)
                row["frames_used"] = video.shape[1]
                if video.shape[1] < 3:
                    raise ValueError(f"too few frames ({video.shape[1]})")
                row["dt"] = dt_for(dynamics)
                # config_for_dynamics exists only in newer code versions (it changes multi-object systems only)
                config_for = getattr(U, "config_for_dynamics", lambda c, d: c)
                run_config = config_for(dict(config, dt=row["dt"]), dynamics)
                _, vals = U.train_one_video(run_config, video, device, epochs=args.epochs, log_interval=50)
                row.update(zip(["alpha", "beta", "max_z", "min_z", "z0", "z1"], vals))
                row["status"] = "ok" if all(math.isfinite(float(v)) for v in vals) else "nonfinite"
            except Exception as e:
                row["status"] = "failed"
                row["error"] = f"{type(e).__name__}: {e}".replace("\n", " ")[:500]
                traceback.print_exc()
            row["seconds"] = round(time.time() - t0, 1)
            writer.writerow(row)
            fcsv.flush()
            print(f"    status={row['status']} gamma0={row['alpha']} gamma1={row['beta']} "
                  f"({row['seconds']} s) {row['error']}", flush=True)

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
