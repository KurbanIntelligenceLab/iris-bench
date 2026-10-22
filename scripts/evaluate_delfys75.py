"""
Delfys75 parameter evaluation with model-specific mappings (added alongside, does not
modify, scripts/compare_baseline_unified.py). Run on the released Table 5 CSVs it
regenerates every value of paper Table 5; see REPRODUCE_DELFYS75.md.

Differences from compare_baseline_unified.py:
  * Mapping depends on the model that produced the CSV:
      baseline = main.py with src/models/PhysModels.py blocks (Garcia et al.)
      unified  = scripts/run_unified_delfys75.py (z'' + g1 z' + g0 z = 0)
  * Failed runs are EXCLUDED and reported as n_ok / n_total. A run is failed if
    its status column says so (new CSVs) or, for old main.py-format CSVs with no
    status column, if alpha, beta, max_z, min_z, z0, z1 are all exactly 0.
  * No parameter is taken from the folder name (the old script set the sliding
    block angle from "low/mid/high", which gave a trivial 0 error).
  * Mappings that are not physically meaningful are reported as such, without MAE.

Usage (Table 5, as in scripts/reproduce_delfys75.sh table5):
  python scripts/evaluate_delfys75.py \
      --run 1step=Results/delfys75_original/delfys75.csv:baseline \
      --run multi=Results/delfys75/delfys75.csv:baseline \
      [--params path/to/parameters.json] --out OUT_DIR
  --run also accepts the clips.csv files written by repro_d75/run_baseline.py / run_unified.py.

  --init_runs base_init5,base1,base_init20   per-clip comparison across inits
  --seed_group base1=base1,base1_s1,base1_s2  MAE mean +- std across seeds (repeatable)
  --write_gt_json FILE                       write GT in compare_baseline_unified.py format
"""

import argparse
import csv
import json
import math
import os
import statistics as st
import sys

G = 9.80665

# Ground truth used by the released comparison files (Delfys75 parameters.json, rounded).
# Used only where --params does not provide a value; the source is recorded per row.
# Note: Kaggle's parameters.json (v11) has NO g for dropped_ball (it gives r, h_0, focal)
# or free_fall (listed as bouncing_ball, with elasticity). g = 9.80665 is the released
# scripts' hard-coded choice, kept here so results stay comparable with the paper.
BUILTIN_GT = {
    ("dropped_ball", s): {"g": G} for s in ("large", "medium", "small")
}
BUILTIN_GT.update({("free_fall", s): {"g": G} for s in ("mousepad", "table", "tennis")})
BUILTIN_GT.update({
    ("led", "led_2s"): {"decay": 2.30},
    ("led", "led_5s"): {"decay": 0.92},
    ("led", "led_10s"): {"decay": 0.46},
    ("pendulum", "pendulum_45"): {"length": 0.45, "damping": -0.0592},
    ("pendulum", "pendulum_90"): {"length": 0.90, "damping": -0.0592},
    ("pendulum", "pendulum_150"): {"length": 1.50, "damping": -0.0592},
    ("torricelli", "large"): {"k": 0.0162},
    ("torricelli", "medium"): {"k": 0.0128},
    ("torricelli", "small"): {"k": 0.0096},
})
# Sliding block: Garcia et al. estimate only the total acceleration a = g (sin(theta) - mu cos(theta)).
SLIDING_ANGLE = {"low": 20.0, "mid": 25.0, "high": 30.0}
SLIDING_MU = 0.2076


def sliding_acc(angle_deg, mu):
    th = math.radians(angle_deg)
    return G * (math.sin(th) - mu * math.cos(th))


BUILTIN_GT.update({("sliding_block", s): {"acceleration": sliding_acc(a, SLIDING_MU)} for s, a in SLIDING_ANGLE.items()})

# Kaggle's free_fall folder holds bouncing-ball clips (parameters.json: "bouncing_ball",
# elasticity). They are not among Garcia et al.'s 75 Delfys75 videos, and g is not their
# ground truth, so they are reported separately.
EXTRA = {"free_fall": "extra: bouncing-ball clips, not in Delfys75; g is not their GT (elasticity)"}

# Rows printed in the paper (Table 4 and Table 5), for the short LaTeX tables.
TABLE4_ROWS = [("dropped_ball", "large", "g"), ("free_fall", "mousepad", "g"), ("led", "led_2s", "decay"),
               ("pendulum", "pendulum_45", "length"), ("pendulum", "pendulum_90", "length"),
               ("pendulum", "pendulum_150", "length"), ("sliding_block", "mid", "acceleration"),
               ("torricelli", "large", "k"), ("torricelli", "small", "k")]
TABLE5_ROWS = [("dropped_ball", "large", "g"), ("free_fall", "mousepad", "g"), ("free_fall", "table", "g"),
               ("free_fall", "tennis", "g"), ("led", "led_10s", "decay"), ("pendulum", "pendulum_45", "length"),
               ("torricelli", "large", "k")]


def map_params(model, dynamics, alpha, beta):
    """Return {param: (estimate or None, mapping_note)} for one clip.

    estimate None means "not estimable / not physically meaningful": no MAE is computed.
    """
    if model == "baseline":
        # Conventions read off src/models/PhysModels.py (identical to Garcia et al.).
        if dynamics in ("dropped_ball", "free_fall"):
            # y_hat = 2 y1 - y0 - alpha dt^2: alpha is an acceleration in latent units.
            return {"g": (alpha, "g=alpha (latent units; scale set only by the KL prior)")}
        if dynamics == "led":
            # y_hat = y1 - dt alpha y1: first-order decay, scale-invariant.
            return {"decay": (alpha, "decay=alpha")}
        if dynamics == "pendulum":
            # y_hat = 2y1 - y0 - dt(beta (y1-y0) + dt (g/|alpha|) sin y1): alpha is the length.
            return {"length": (abs(alpha), "L=|alpha|"),
                    "damping": (-beta, "damping=-beta (sign convention unverified, see notes)")}
        if dynamics == "sliding_block":
            # y_hat = 2y1 - y0 + 10 alpha dt^2: z'' = 10 alpha. beta never enters the model,
            # so mu is not estimable; the total acceleration is (Garcia et al. Sec. 4).
            return {"acceleration": (10 * abs(alpha), "a=10|alpha| (latent units; mu not estimable)")}
        if dynamics == "torricelli":
            # y_hat = |y1| - dt alpha sqrt|y1|: k=alpha in latent units.
            return {"k": (alpha, "k=alpha (latent units)")}
    elif model == "unified":
        # run_unified_delfys75.py fits z'' + g1 z' + g0 z = 0 to every system;
        # CSV stores alpha=g0 (gamma_0_0), beta=g1 (gamma_0_1).
        if dynamics == "pendulum":
            est = G / alpha if alpha > 0 else None
            return {"length": (est, "L=g/gamma0 (small-angle; None if gamma0<=0)"),
                    "damping": (-beta, "damping=-gamma1 (sign convention unverified, see notes)")}
        note = "not physically meaningful: damped-oscillator fit to a non-oscillator system"
        name = {"dropped_ball": "g", "free_fall": "g", "led": "decay",
                "sliding_block": "acceleration", "torricelli": "k"}[dynamics]
        return {name: (None, note)}
    raise ValueError(f"unknown model/dynamics: {model}/{dynamics}")


def load_gt(params_path):
    gt = {k: {p: (v, "builtin") for p, v in d.items()} for k, d in BUILTIN_GT.items()}
    if not params_path:
        return gt
    with open(params_path) as f:
        pj = json.load(f)
    def val(x):
        return x["mean"] if isinstance(x, dict) and "mean" in x else x

    for (dyn, setting), d in gt.items():
        s = (pj.get(dyn) or {}).get(setting) or {}
        for p in list(d):
            v = val(s.get(p))
            if p == "acceleration" and isinstance(val(s.get("angle")), (int, float)) \
                    and isinstance(val(s.get("friction")), (int, float)):
                v = sliding_acc(val(s["angle"]), val(s["friction"]))
            if isinstance(v, (int, float)):
                if abs(v - d[p][0]) > 1e-3 * abs(d[p][0]):
                    print(f"WARNING: GT {dyn}/{setting}/{p}: params file {v} != builtin {d[p][0]}; using params file")
                d[p] = (float(v), "params_json")
    return gt


def load_runs(path):
    """Read either a new clips.csv (with status) or an old main.py-format CSV."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    out = []
    if "status" in header:
        idx = {h: i for i, h in enumerate(header)}
        for r in body:
            ok = r[idx["status"]] == "ok"
            out.append({"dynamics": r[idx["dynamics"]], "setting": r[idx["setting"]], "run": r[idx["run"]],
                        "ok": ok, "error": r[idx["error"]] if "error" in idx else "",
                        "alpha": float(r[idx["alpha"]]) if ok else math.nan,
                        "beta": float(r[idx["beta"]]) if ok else math.nan})
    else:
        # Old format: path components, then alpha, beta, max_z, min_z, z0, z1.
        for r in body:
            if len(r) < 8:
                continue
            vals = [float(x) for x in r[-6:]]
            failed = all(v == 0.0 for v in vals)
            out.append({"dynamics": r[0].strip().lower(), "setting": r[1].strip().lower(),
                        "run": r[2].strip() if len(r) > 8 else "", "ok": not failed,
                        "error": "all-zero row (exception written as zeros)" if failed else "",
                        "alpha": vals[0], "beta": vals[1]})
    return out


def evaluate(name, model, runs, gt):
    per_clip, groups = [], {}
    for r in runs:
        key = (r["dynamics"], r["setting"])
        if key not in gt:
            continue
        for param, (gt_val, gt_src) in gt[key].items():
            g = groups.setdefault((key[0], key[1], param), {"errs": [], "ests": [], "n_total": 0,
                                                             "gt": gt_val, "gt_src": gt_src, "note": ""})
            g["n_total"] += 1
            est, note = (None, "run failed") if not r["ok"] else map_params(model, key[0], r["alpha"], r["beta"]).get(param, (None, "no mapping"))
            if r["ok"]:
                g["note"] = note
            err = abs(est - gt_val) if est is not None and math.isfinite(est) else None
            if err is not None:
                g["errs"].append(err)
                g["ests"].append(est)
            per_clip.append({"run_name": name, "model": model, "dynamics": key[0], "setting": key[1], "clip": r["run"],
                             "param": param, "status": "ok" if r["ok"] else "failed", "error_msg": r["error"],
                             "alpha": r["alpha"], "beta": r["beta"], "estimate": "" if est is None else est,
                             "gt": gt_val, "abs_err": "" if err is None else err, "mapping": note})
    summary = []
    for (dyn, setting, param), g in sorted(groups.items()):
        e = g["errs"]
        summary.append({"run_name": name, "model": model, "subset": "extra" if dyn in EXTRA else "delfys75",
                        "dynamics": dyn, "setting": setting, "param": param,
                        "gt": g["gt"], "gt_source": g["gt_src"],
                        "est_mean": st.mean(g["ests"]) if e else "", "mae": st.mean(e) if e else "",
                        "mae_std": st.stdev(e) if len(e) > 1 else ("" if not e else 0.0),
                        "n_ok": len(e), "n_total": g["n_total"], "mapping": g["note"] or "all runs failed"})
    return per_clip, summary


def fmt(v, nd=2):
    return "--" if v in ("", None) else f"{v:.{nd}f}"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_latex(path, summary, run_names, rows, caption):
    by = {(s["run_name"], s["dynamics"], s["setting"], s["param"]): s for s in summary}
    lines = [r"\begin{tabular}{lll r " + " ".join(["r"] * len(run_names)) + "}", r"\toprule",
             "Dynamics & Setting & Param & GT & " + " & ".join(run_names) + r" \\", r"\midrule"]
    for dyn, setting, param in rows:
        cells = []
        gt = ""
        for rn in run_names:
            s = by.get((rn, dyn, setting, param))
            if s is None:
                cells.append("--")
                continue
            gt = s["gt"]
            cells.append(f"{fmt(s['mae'])} ({s['n_ok']}/{s['n_total']})" if s["mae"] != "" else f"n/a ({s['n_ok']}/{s['n_total']})")
        mark = r"$^\ast$" if dyn in EXTRA else ""
        lines.append(f"{dyn.replace('_', ' ')}{mark} & {setting.replace('_', ' ')} & {param} & {fmt(gt, 3)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", f"% {caption}",
              "% Cells: MAE over successful clips (n_ok/n_total). n/a = failed or not physically mapped.",
              "% * " + EXTRA["free_fall"]]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def init_sensitivity(names, runs_by_name, out_dir):
    """Per clip alpha across inits: does the estimate track the initial value?"""
    keyed = {n: {(r["dynamics"], r["setting"], r["run"]): r for r in runs_by_name[n]} for n in names}
    clips = sorted(set().union(*[set(k) for k in keyed.values()]))
    rows = []
    for c in clips:
        row = {"dynamics": c[0], "setting": c[1], "clip": c[2]}
        vals = []
        for n in names:
            r = keyed[n].get(c)
            a = r["alpha"] if r and r["ok"] else math.nan
            row[f"alpha_{n}"] = a
            vals.append(a)
        fin = [v for v in vals if math.isfinite(v)]
        row["spread_max_minus_min"] = (max(fin) - min(fin)) if len(fin) == len(vals) else ""
        rows.append(row)
    write_csv(os.path.join(out_dir, "init_sensitivity_per_clip.csv"), rows)
    agg = {}
    for r in rows:
        if r["spread_max_minus_min"] != "":
            agg.setdefault((r["dynamics"], r["setting"]), []).append(r)
    lines = [f"Init sensitivity across runs {names} (alpha per clip; failed clips excluded)",
             f"{'dynamics':<14}{'setting':<14}" + "".join(f"{'mean_' + n:>22}" for n in names) + f"{'mean_spread':>14}{'n':>4}"]
    for (dyn, s), rs in sorted(agg.items()):
        means = [st.mean(r[f"alpha_{n}"] for r in rs) for n in names]
        lines.append(f"{dyn:<14}{s:<14}" + "".join(f"{m:>22.4f}" for m in means)
                     + f"{st.mean(r['spread_max_minus_min'] for r in rs):>14.4f}{len(rs):>4}")
    lines.append("If mean alpha moves with the init value, the parameter is not identified by the data.")
    with open(os.path.join(out_dir, "init_sensitivity_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def seed_summary(groups, summary, out_dir):
    """Per (dynamics, setting, param): mean and std over seeds of the per-seed MAE."""
    by = {(s["run_name"], s["dynamics"], s["setting"], s["param"]): s for s in summary}
    keys = sorted({(s["dynamics"], s["setting"], s["param"]) for s in summary}, key=lambda r: (r[0] in EXTRA, r))
    rows, lines = [], []
    for spec in groups:
        label, members = spec.split("=", 1)
        members = members.split(",")
        lines.append(f"Seed group {label}: {members}  (MAE mean +- std over seeds; n_ok = clips per seed)")
        for k in keys:
            vals, n_ok = [], []
            for m in members:
                s = by.get((m,) + k)
                if s is not None and s["mae"] != "":
                    vals.append(s["mae"])
                    n_ok.append(f"{s['n_ok']}/{s['n_total']}")
            if not any(by.get((m,) + k) for m in members):
                continue
            mean = st.mean(vals) if vals else ""
            std = st.stdev(vals) if len(vals) > 1 else ""
            rows.append({"group": label, "subset": "extra" if k[0] in EXTRA else "delfys75", "dynamics": k[0],
                         "setting": k[1], "param": k[2], "mae_mean": mean, "mae_std_over_seeds": std,
                         "n_seeds": len(vals), "n_ok_per_seed": " ".join(n_ok)})
            cell = f"{mean:.3f} +- {std:.3f}" if std != "" else (f"{mean:.3f}" if vals else "n/a")
            lines.append(f"  {k[0]:<14}{k[1]:<14}{k[2]:<13}{cell:>20}  seeds={len(vals)}  n_ok={' '.join(n_ok)}")
        lines.append("")
    write_csv(os.path.join(out_dir, "seed_summary.csv"), rows)
    with open(os.path.join(out_dir, "seed_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="name=path.csv:baseline|unified (repeatable)")
    ap.add_argument("--params", default=None, help="parameters.json (compare_baseline_unified.py format); else builtin GT")
    ap.add_argument("--out", required=False, help="output directory (must not already exist)")
    ap.add_argument("--init_runs", default=None, help="comma-separated run names for init sensitivity")
    ap.add_argument("--seed_group", action="append", default=[],
                    help="label=run1,run2,...: MAE mean +- std across these runs (seeds)")
    ap.add_argument("--write_gt_json", default=None, help="write GT json for compare_baseline_unified.py and exit")
    args = ap.parse_args()

    gt = load_gt(args.params)
    if args.write_gt_json:
        pj = {}
        for (dyn, s), d in gt.items():
            for p, (v, _) in d.items():
                pj.setdefault(dyn, {}).setdefault(s, {})[p] = {"mean": v}
        with open(args.write_gt_json, "w") as f:
            json.dump(pj, f, indent=1)
        print(f"Wrote {args.write_gt_json}")
        if not args.run:
            return
    if not args.out or not args.run:
        ap.error("--out and at least one --run are required")
    if os.path.exists(args.out):
        sys.exit(f"Refusing to overwrite existing output directory {args.out}")
    os.makedirs(args.out)

    all_clip, all_sum, names, runs_by_name = [], [], [], {}
    for spec in args.run:
        name, rest = spec.split("=", 1)
        path, model = rest.rsplit(":", 1)
        runs = load_runs(path)
        runs_by_name[name] = runs
        names.append(name)
        n_fail = sum(not r["ok"] for r in runs)
        print(f"{name}: {path} ({model}) - {len(runs)} clips, {n_fail} failed")
        c, s = evaluate(name, model, runs, gt)
        all_clip += c
        all_sum += s

    write_csv(os.path.join(args.out, "per_clip_errors.csv"), all_clip)
    write_csv(os.path.join(args.out, "summary.csv"), all_sum)
    write_latex(os.path.join(args.out, "table4_rows.tex"), all_sum, names, TABLE4_ROWS, "Rows of paper Table 4")
    write_latex(os.path.join(args.out, "table5_rows.tex"), all_sum, names, TABLE5_ROWS, "Rows of paper Table 5")
    all_rows = sorted({(s["dynamics"], s["setting"], s["param"]) for s in all_sum},
                      key=lambda r: (r[0] in EXTRA, r))
    write_latex(os.path.join(args.out, "all_rows.tex"), all_sum, names, all_rows, "All Delfys75 settings")

    by = {(s["run_name"], s["dynamics"], s["setting"], s["param"]): s for s in all_sum}
    lines = [f"{'dynamics':<14}{'setting':<14}{'param':<9}{'GT':>8}" + "".join(f"{n:>22}" for n in names)]
    extra_header_done = False
    for dyn, setting, param in all_rows:
        if dyn in EXTRA and not extra_header_done:
            lines.append(f"--- {EXTRA[dyn]} ---")
            extra_header_done = True
        gt_v = next(s["gt"] for s in all_sum if (s["dynamics"], s["setting"], s["param"]) == (dyn, setting, param))
        cells = []
        for n in names:
            s = by.get((n, dyn, setting, param))
            cells.append("--" if s is None else
                         f"{fmt(s['mae'], 3) if s['mae'] != '' else 'n/a'} ({s['n_ok']}/{s['n_total']})")
        lines.append(f"{dyn:<14}{setting:<14}{param:<9}{gt_v:>8.4f}" + "".join(f"{c:>22}" for c in cells))
    lines += ["", "Cells: MAE over successful clips (n_ok/n_total). n/a = all failed or no physical mapping.",
              "Mapping notes per row are in summary.csv. Pendulum damping sign convention is unverified."]
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if args.init_runs:
        init_sensitivity(args.init_runs.split(","), runs_by_name, args.out)
    if args.seed_group:
        seed_summary(args.seed_group, all_sum, args.out)


if __name__ == "__main__":
    main()
