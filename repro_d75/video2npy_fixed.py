"""
Delfys75 video -> .npy preprocessing with a fixed, logged stride and window length.

Same frame processing as src/utils/video2npy.py and Garcia et al.'s original
(resize to 100x56 with cv2.resize, BGR->gray, /255, windows of nf frames taken every
`stride` frames, window start i in range(nframes - nf*stride)), except:
  * stride and nf are fixed per dynamics (no adaptive 6/2/1 choice):
      --stride S      the same stride S and nf=10 for every system (old pipeline layout)
      --preset garcia per-system stride/nf from Garcia et al. (CVPR 2025), Table 5
  * clips too short to give one window are LISTED, not silently skipped;
  * arrays are saved as float32 (training casts to float32 anyway);
  * every clip gets a report row (fps, frames, stride, nf, dt, samples, sha256), and
    <output_dir>/protocol.json records stride, nf and dt per dynamics for the runners.

Output layout expected by main.py / run_unified_delfys75.py:
    <output_dir>/<dynamics>/<setting>/<run>/<run>.npy

Input layout is discovered: each video's path is mapped to (dynamics, setting, run)
from its folder names. Anything that cannot be mapped is an error (exit 2) and is
listed, so the mapping can be fixed rather than guessed.

Usage:
  python -I repro_d75/video2npy_fixed.py --input_dir RAW --output_dir OUT \
      (--stride 6 | --preset garcia) --report OUT_report.csv [--workers 8] [--dry_run]
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

W, H = 100, 56
FPS_NOMINAL = 60.0
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")

SETTINGS = {
    "dropped_ball": ["large", "medium", "small"],
    "free_fall": ["mousepad", "table", "tennis"],
    "led": ["led_2s", "led_5s", "led_10s"],
    "pendulum": ["pendulum_45", "pendulum_90", "pendulum_150"],
    "sliding_block": ["low", "mid", "high"],
    "torricelli": ["large", "medium", "small"],
}

# Garcia et al., "Learning Physics From Video" (CVPR 2025, arXiv:2410.01376v2), Table 5:
# dt and frames per sample per system, at 60 fps. free_fall (Kaggle's bouncing-ball
# clips, 16-23 frames) is not one of their 75 videos; it gets the shortest stride and
# nf=10 so that every clip yields windows.
PRESETS = {
    "garcia": {
        "pendulum": {"stride": 6, "nf": 20},       # dt 1/10
        "torricelli": {"stride": 6, "nf": 20},     # dt 1/10
        "led": {"stride": 1, "nf": 20},            # dt 1/60
        "sliding_block": {"stride": 2, "nf": 10},  # dt 1/30
        "dropped_ball": {"stride": 2, "nf": 4},    # dt 1/30 ("free fall, scale")
        "free_fall": {"stride": 1, "nf": 10},      # dt 1/60 (not in Garcia et al.)
    },
}


def norm(s):
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def map_path(rel_path):
    """(dynamics, setting, run) from a path relative to input_dir, or None."""
    parts = [norm(p) for p in rel_path.split(os.sep)]
    stem = os.path.splitext(parts[-1])[0]
    folders = parts[:-1]
    dyn = next((p for p in folders if p in SETTINGS), None)
    if dyn is None:
        return None
    after = folders[folders.index(dyn) + 1:]
    setting, idx = None, None
    for j, p in enumerate(after):
        for s in SETTINGS[dyn]:
            # exact, or a short alias such as "45" for pendulum_45 / "2s" for led_2s
            if p == s or (s.startswith(dyn.split("_")[0] + "_") and p == s.split("_", 1)[1]):
                setting, idx = s, j
                break
        if setting:
            break
    if setting is None:
        return None
    rest = after[idx + 1:]
    run = rest[-1] if rest else stem
    return dyn, setting, run


def process(job):
    video_path, out_path, stride, nf, dry_run = job
    import cv2  # imported in the worker

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "unreadable", "fps": "", "frames": 0, "n_samples": 0, "sha256": ""}
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (W, H)), cv2.COLOR_BGR2GRAY)
        frames.append(gray / 255.0)
    cap.release()
    nframes = len(frames)
    n_samples = max(0, nframes - nf * stride)  # same range() as the original code
    row = {"fps": fps, "frames": nframes, "n_samples": n_samples, "sha256": ""}
    if n_samples == 0:
        row["status"] = "too_short"
        return row
    if dry_run:
        row["status"] = "dry_run"
        return row
    arr = np.asarray(frames, dtype=np.float32)
    data = np.stack([arr[i:i + nf * stride:stride][:, None] for i in range(n_samples)])
    assert data.shape == (n_samples, nf, 1, H, W), data.shape
    os.makedirs(os.path.dirname(out_path), exist_ok=False)
    np.save(out_path, data)
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    row["sha256"] = h.hexdigest()
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True, help="must not exist yet")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stride", type=int, help="same stride for every system, nf=10")
    g.add_argument("--preset", choices=sorted(PRESETS), help="per-system stride and nf")
    ap.add_argument("--report", required=True, help="per-clip CSV report")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true", help="map paths and count frames; write no .npy")
    args = ap.parse_args()

    if args.preset:
        protocol = {d: dict(v) for d, v in PRESETS[args.preset].items()}
    else:
        protocol = {d: {"stride": args.stride, "nf": 10} for d in SETTINGS}
    for v in protocol.values():
        v["dt"] = v["stride"] / FPS_NOMINAL
    protocol_name = args.preset or f"uniform_stride{args.stride}_nf10"

    videos = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(VIDEO_EXT) and "mask" not in f.lower():
                videos.append(os.path.join(root, f))
    videos.sort()
    if not videos:
        npys = [os.path.join(r, f) for r, _, fs in os.walk(args.input_dir) for f in fs if f.endswith(".npy")]
        sys.exit(f"No videos under {args.input_dir}. Found {len(npys)} .npy files instead; "
                 "the download may ship preprocessed arrays. Stop and check before continuing.")

    mapped, unmapped = {}, []
    for v in videos:
        m = map_path(os.path.relpath(v, args.input_dir))
        if m is None:
            unmapped.append(v)
        elif m in mapped:
            unmapped.append(f"{v}  (duplicate of {mapped[m]})")
        else:
            mapped[m] = v
    if unmapped:
        print("ERROR: could not map these videos to (dynamics, setting, run):", file=sys.stderr)
        for u in unmapped:
            print("  " + u, file=sys.stderr)
        sys.exit(2)

    if not args.dry_run:
        if os.path.exists(args.output_dir):
            sys.exit(f"Refusing to write into existing {args.output_dir}")
        os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "protocol.json"), "w") as f:
            json.dump({"name": protocol_name, "fps_nominal": FPS_NOMINAL, "dynamics": protocol}, f, indent=1)
    keys = sorted(mapped)
    jobs = [(mapped[k], os.path.join(args.output_dir, k[0], k[1], k[2], f"{k[2]}.npy"),
             protocol[k[0]]["stride"], protocol[k[0]]["nf"], args.dry_run) for k in keys]
    with Pool(args.workers) as pool:
        results = pool.map(process, jobs, chunksize=1)

    fields = ["dynamics", "setting", "run", "video", "fps", "frames", "stride", "nf", "dt", "n_samples",
              "status", "npy", "sha256"]
    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, job, r in zip(keys, jobs, results):
            w.writerow({"dynamics": k[0], "setting": k[1], "run": k[2], "video": job[0], "fps": r["fps"],
                        "frames": r["frames"], "stride": job[2], "nf": job[3],
                        "dt": f"{protocol[k[0]]['dt']:.6f}", "n_samples": r["n_samples"], "status": r["status"],
                        "npy": job[1] if r["status"] == "ok" else "", "sha256": r["sha256"]})

    counts = {}
    for k, r in zip(keys, results):
        counts.setdefault((k[0], k[1]), []).append(r)
    print(f"{len(keys)} videos, protocol {protocol_name}. Per setting:")
    print(f"  {'setting':<28}{'stride':>7}{'nf':>4}{'dt':>9}{'ok/total':>10}{'windows':>9}")
    for (d, s_), rs in sorted(counts.items()):
        p = protocol[d]
        n_ok = sum(r["status"] in ("ok", "dry_run") for r in rs)
        print(f"  {d + '/' + s_:<28}{p['stride']:>7}{p['nf']:>4}{p['dt']:>9.4f}{f'{n_ok}/{len(rs)}':>10}"
              f"{sum(r['n_samples'] for r in rs):>9}")
    fps_set = sorted({round(r["fps"], 3) for r in results if r["fps"]})
    print(f"fps values seen: {fps_set}  (dt = stride / 60 nominal, as in Garcia et al.)")
    for k, r in zip(keys, results):
        if r["status"] not in ("ok", "dry_run"):
            print(f"  NOT SAVED: {'/'.join(k)}: {r['status']} ({r['frames']} frames)")
    print(f"Report: {args.report}")
    expected = sum(len(v) for v in SETTINGS.values()) * 5
    if len(keys) != expected:
        print(f"NOTE: {len(keys)} videos mapped; expected {expected} (18 settings x 5).")


if __name__ == "__main__":
    main()
