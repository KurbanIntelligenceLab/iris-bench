"""
Which frame stride produced the .npy files behind the archived runs?

Results/delfys75_1/delfys75.log prints "Data shape: (N, 10, 1, 56, 100)" per clip.
With the windowing used by every video2npy version, N = frames - 10 * stride.
Given the true frame count of each video (from video2npy_fixed.py's report),
this script checks which stride reproduces each logged N.

Usage (the log is in this repository's history: git show d41564e:Results/delfys75_1/delfys75.log):
  python -I repro_d75/stride_evidence.py --log Results/delfys75_1/delfys75.log \
      --report DATA/preprocess_s6_report.csv --out stride_evidence.csv
"""

import argparse
import csv
import re

STRIDES = (1, 2, 3, 6, 24)  # 6/2/1: adaptive video2npy; 3: dt 0.05 at 60 fps; 24: Garcia et al.'s first version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    logged, current = {}, None
    with open(args.log, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Processing (.+\.npy)\.\.\.", line)
            if m:
                parts = re.split(r"[\\/]+", m.group(1))
                current = tuple(p.lower() for p in parts[-4:-1])  # dynamics, setting, run
                continue
            m = re.match(r"Data shape:\s*\((\d+),", line)
            if m and current:
                logged[current] = int(m.group(1))
                current = None

    frames = {}
    with open(args.report) as f:
        for r in csv.DictReader(f):
            frames[(r["dynamics"], r["setting"], r["run"].lower())] = (int(r["frames"]), r["fps"])

    rows, tally = [], {}
    for key, n in sorted(logged.items()):
        fr = frames.get(key)
        row = {"dynamics": key[0], "setting": key[1], "run": key[2], "logged_N": n,
               "frames": fr[0] if fr else "", "fps": fr[1] if fr else ""}
        match = []
        for s in STRIDES:
            pred = (fr[0] - 10 * s) if fr else ""
            row[f"N_if_stride_{s}"] = pred
            if fr and pred == n:
                match.append(str(s))
        row["matching_strides"] = ",".join(match) if fr else "no frame count"
        tally[row["matching_strides"]] = tally.get(row["matching_strides"], 0) + 1
        rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} clips with a logged shape (clips that failed to load have none).")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  matching stride(s) = {k or 'none'}: {v} clips")
    print(f"Per-clip table: {args.out}")


if __name__ == "__main__":
    main()
