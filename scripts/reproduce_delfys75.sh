#!/bin/bash
# Reproduce the Delfys75 results of the IRIS paper (Tables 4 and 5). See REPRODUCE_DELFYS75.md.
# Run from the repository root:
#
#   bash scripts/reproduce_delfys75.sh table5       # CPU, seconds: Table 5 from the released CSVs
#   bash scripts/reproduce_delfys75.sh data         # download Delfys75 (Kaggle v11); needs KAGGLE_API_TOKEN
#   bash scripts/reproduce_delfys75.sh preprocess   # videos -> .npy, stride 6 (Table 4 setting) + checksums
#   bash scripts/reproduce_delfys75.sh table4       # GPU, a few hours: re-run both Table 4 columns and score them
#
# Table 4 is re-run with the code version that produced it, included unchanged in
# repro_d75/table4_code/ (see the README there).
# Paths can be changed with environment variables (defaults in brackets):
#   D75_RAW [./delfys75_raw]  D75_NPY [./delfys75_s6]  D75_OUT [./Results/repro_d75_local]
#   (output folders are never overwritten)
set -euo pipefail

STEP="${1:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
RAW="${D75_RAW:-$REPO/delfys75_raw}"
NPY="${D75_NPY:-$REPO/delfys75_s6}"
TABLE4_CODE="$REPO/repro_d75/table4_code"
OUT="${D75_OUT:-$REPO/Results/repro_d75_local}"
DATASET="jaswar/physical-parameter-prediction"
VERSION=11
ZIP_SHA256="1d2320351105ceb0b90c8a926d815acb0fe4eb629ed859318b90e269853c7137"  # Kaggle v11, 2026-10-14
export WANDB_MODE=disabled MPLBACKEND=Agg

new_dir() {  # refuse to overwrite earlier results
    if [ -e "$1" ]; then echo "ERROR: $1 exists; move it away or set D75_OUT." >&2; exit 1; fi
    mkdir -p "$1"
}

gt_json() {  # ground truth in the format of scripts/compare_baseline_unified.py
    local params="$RAW/output_selected/parameters.json" out="$1"
    if [ -f "$params" ]; then
        python scripts/evaluate_delfys75.py --params "$params" --write_gt_json "$out"
    else
        python scripts/evaluate_delfys75.py --write_gt_json "$out"
    fi
}

case "$STEP" in
table5)
    # Table 5 = Results/delfys75/parameter_errors_comparison.txt:
    #   "1-step" from Results/delfys75_original/delfys75.csv, "Multi" from Results/delfys75/delfys75.csv
    #   (the Multi run used VLM-based equation-family selection, see Results/delfys75/vlm_results.csv).
    # Failed runs (all-zero rows) are excluded, as in the paper.
    D="$OUT/table5"; new_dir "$D"
    python scripts/evaluate_delfys75.py \
        --run 1step=Results/delfys75_original/delfys75.csv:baseline \
        --run multi=Results/delfys75/delfys75.csv:baseline \
        --out "$D/eval" > "$D/eval.txt"
    python - "$D/eval/summary.csv" <<'EOF' | tee "$D/checks.txt"
import csv, sys
S = {(r["run_name"], r["dynamics"], r["setting"], r["param"]): float(r["mae"])
     for r in csv.DictReader(open(sys.argv[1])) if r["mae"] != ""}
paper = [  # (dynamics, setting, param, 1-step, Multi) as printed in Table 5
    ("dropped_ball", "large", "g", 0.25, 0.18), ("free_fall", "mousepad", "g", 2.12, 1.65),
    ("free_fall", "table", "g", 1.06, 0.15), ("free_fall", "tennis", "g", 1.06, 0.53),
    ("led", "led_10s", "decay", 2.09, 7.46), ("pendulum", "pendulum_45", "length", 0.35, 0.50),
    ("torricelli", "large", "k", 6.27, 6.54)]
ok = 0
print(f"{'row':<30}{'paper 1-step':>13}{'regen.':>9}{'paper Multi':>13}{'regen.':>9}")
for d, s, p, a, b in paper:
    x, y = S[("1step", d, s, p)], S[("multi", d, s, p)]
    good = abs(x - a) <= 0.0051 and abs(y - b) <= 0.0051
    ok += good
    print(f"{d + '/' + s + '/' + p:<30}{a:>13.2f}{x:>9.2f}{b:>13.2f}{y:>9.2f}  {'OK' if good else 'DIFF'}")
print(f"{ok}/{len(paper)} Table 5 rows regenerated")
sys.exit(0 if ok == len(paper) else 1)
EOF
    ;;
data)
    : "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN (kaggle.com -> Settings -> API)}"
    new_dir "$RAW"
    HDR=$(mktemp); chmod 600 "$HDR"; trap 'rm -f "$HDR"' EXIT
    printf 'Authorization: Bearer %s\n' "$KAGGLE_API_TOKEN" > "$HDR"
    curl -fSL --retry 3 -H @"$HDR" \
        "https://www.kaggle.com/api/v1/datasets/download/$DATASET?datasetVersionNumber=$VERSION" -o "$RAW/delfys75_v$VERSION.zip"
    GOT=$(sha256sum "$RAW/delfys75_v$VERSION.zip" | cut -d' ' -f1)
    if [ "$GOT" != "$ZIP_SHA256" ]; then
        echo "WARNING: zip SHA-256 $GOT differs from the reference $ZIP_SHA256." >&2
        [ "${D75_SKIP_ZIP_CHECK:-0}" = "1" ] || { echo "Set D75_SKIP_ZIP_CHECK=1 to continue (Kaggle may re-package); the .npy manifest is still checked." >&2; exit 1; }
    else
        echo "zip SHA-256 matches the reference."
    fi
    python -I -c 'import sys, zipfile; z = zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None; z.extractall(sys.argv[2])' \
        "$RAW/delfys75_v$VERSION.zip" "$RAW"
    echo "Videos: $(find "$RAW" -name video.mp4 | wc -l) (expected 90)"
    ;;
preprocess)
    # Same frame processing as src/utils/video2npy.py, at the fixed stride 6 used for Table 4
    # (Delta t = 0.1 s at 60 fps, 10 frames per sample). Clips shorter than 60 frames
    # (dropped ball, free fall, sliding block) give no samples and are listed as too_short.
    D="$OUT/preprocess"; new_dir "$D"
    python -I repro_d75/video2npy_fixed.py --input_dir "$RAW" --output_dir "$NPY" --stride 6 \
        --report "$D/preprocess_s6_report.csv" --workers "${D75_WORKERS:-8}" | tee "$D/preprocess.log"
    python -I repro_d75/verify_npy.py --root "$NPY" --manifest "$D/delfys75_s6.sha256.tsv" \
        --report "$D/preprocess_s6_report.csv" | tee "$D/verify.log"
    echo "Compare $D/delfys75_s6.sha256.tsv with Results/repro_d75/preprocessing/delfys75_s6.sha256.tsv"
    ;;
table4)
    [ -d "$NPY" ] || { echo "Run the preprocess step first." >&2; exit 1; }
    [ -f "$TABLE4_CODE/main.py" ] || { echo "Missing $TABLE4_CODE (Table 4 code version)." >&2; exit 1; }
    D="$OUT/table4"; new_dir "$D"
    # Published setting: stride-6 data, --dt 0.05, one-step loss, seed 42.
    # The runners exit with 3 when they finish but some clips failed (expected here); other codes are crashes.
    for job in base corr; do
        set +e
        if [ "$job" = base ]; then
            python -u repro_d75/run_baseline.py --code_dir "$TABLE4_CODE" --data_root "$NPY" --dt 0.05 \
                --loss latent_loss --out_dir "$D/base" 2>&1 | tee "$D/base.log"
        else
            python -u repro_d75/run_unified.py --code_dir "$TABLE4_CODE" --data_root "$NPY" --dt 0.05 \
                --seed 42 --out_dir "$D/corr" 2>&1 | tee "$D/corr.log"
        fi
        rc=${PIPESTATUS[0]}
        set -e
        if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then echo "ERROR: $job runner crashed (exit $rc); see $D/$job.log" >&2; exit "$rc"; fi
    done
    gt_json "$D/gt.json"
    python scripts/compare_baseline_unified.py --baseline "$D/base/mainpy_format_ok_only.csv" \
        --unified "$D/corr/mainpy_format_ok_only.csv" --params "$D/gt.json" --out "$D/compare" > /dev/null
    echo "Re-run Table 4 (released scoring script): $D/compare/parameter_errors_comparison.txt"
    echo "Reference re-run: Results/repro_d75/table4_published_setting/parameter_errors_comparison.txt"
    grep -E 'length_m|led_2s|torricelli' "$D/compare/parameter_errors_comparison.txt" || true
    ;;
*)
    sed -n '2,14p' "$0"; exit 1 ;;
esac
