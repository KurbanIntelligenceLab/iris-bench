"""
Check every Delfys75 .npy before training and write a SHA-256 manifest.

Fails (exit 1) unless every file under --root loads with shape (N, nf, 1, 56, 100),
N > 0, a float dtype, finite values in [0, 1], and the layout is
<root>/<dynamics>/<setting>/<run>/<run>.npy. With --report (from video2npy_fixed.py),
nf and N must equal that clip's reported values, every "ok" clip in the report must
exist, and the SHA-256 must match the one recorded at write time.

Usage:
  python -I repro_d75/verify_npy.py --root DATA --manifest DATA.sha256.tsv [--report REPORT.csv]
"""

import argparse
import csv
import hashlib
import os
import sys

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True, help="output TSV: relpath, shape, dtype, sha256")
    ap.add_argument("--report", default=None, help="video2npy_fixed.py report (expected nf, N, sha256)")
    ap.add_argument("--nf", type=int, default=10, help="expected frames per sample when no --report")
    args = ap.parse_args()

    expected = {}
    if args.report:
        with open(args.report) as f:
            for r in csv.DictReader(f):
                if r["status"] == "ok":
                    expected[os.path.join(r["dynamics"], r["setting"], r["run"], r["run"] + ".npy")] = r

    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.root) for f in fs if f.endswith(".npy"))
    if not files:
        sys.exit(f"No .npy files under {args.root}")
    bad, rows, per_setting = [], [], {}
    for p in files:
        rel = os.path.relpath(p, args.root)
        parts = rel.split(os.sep)
        try:
            a = np.load(p, mmap_mode="r", allow_pickle=False)
            problems = []
            if len(parts) != 4 or parts[3] != parts[2] + ".npy":
                problems.append("layout is not dynamics/setting/run/run.npy")
            exp = expected.get(rel)
            nf = int(exp["nf"]) if exp else args.nf
            if args.report and exp is None:
                problems.append("not listed as ok in the report")
            if a.ndim != 5 or a.shape[1:] != (nf, 1, 56, 100):
                problems.append(f"shape {a.shape}, expected (N, {nf}, 1, 56, 100)")
            elif exp and a.shape[0] != int(exp["n_samples"]):
                problems.append(f"N={a.shape[0]} but report says {exp['n_samples']}")
            elif a.shape[0] == 0:
                problems.append("zero samples")
            if not np.issubdtype(a.dtype, np.floating):
                problems.append(f"dtype {a.dtype}")
            if not problems:
                probe = np.asarray(a[:: max(1, a.shape[0] // 20)])
                if not np.isfinite(probe).all() or probe.min() < 0 or probe.max() > 1:
                    problems.append(f"values outside [0,1] or non-finite (min {probe.min()}, max {probe.max()})")
        except Exception as e:  # unreadable file is a failure, never skipped
            problems = [f"load error: {e}"]
            a = None
        digest = sha256(p)
        if expected.get(rel) and expected[rel]["sha256"] != digest:
            problems.append("sha256 differs from the one recorded when the file was written")
        if problems:
            bad.append((rel, problems))
        per_setting.setdefault(tuple(parts[:2]), 0)
        per_setting[tuple(parts[:2])] += 1
        rows.append((rel, "x".join(map(str, a.shape)) if a is not None else "?", str(a.dtype) if a is not None else "?", digest))

    seen = {r[0] for r in rows}
    for rel in sorted(set(expected) - seen):
        bad.append((rel, ["listed as ok in the report but missing on disk"]))
    with open(args.manifest, "w") as f:
        f.write("relpath\tshape\tdtype\tsha256\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"{len(files)} files checked; manifest: {args.manifest}")
    for k, n in sorted(per_setting.items()):
        flag = "" if n == 5 else "   <-- expected 5"
        print(f"  {'/'.join(k):<30}{n}{flag}")
    if bad:
        print(f"FAILED: {len(bad)} bad files", file=sys.stderr)
        for rel, probs in bad:
            print(f"  {rel}: {'; '.join(probs)}", file=sys.stderr)
        sys.exit(1)
    print("All files OK.")


if __name__ == "__main__":
    main()
