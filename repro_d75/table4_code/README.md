# Code version used for paper Table 4 (Delfys75)

The files in this folder are an unchanged copy of the code as of 2026-03-03. That is the
version that produced the Table 4 result files `Results/delfys75_baseline/` and
`Results/delfys75_improved/`.

| File | Role in Table 4 |
|---|---|
| `main.py`, `config.yaml`, `src/` | "Base." column |
| `scripts/run_unified_delfys75.py`, `config_unified.yaml`, `src/` | "Corr." column |

The code in the repository root has changed since then. For example, `main.py` now caps
each clip at 1500 samples and the pendulum block clamps its length parameter. This copy
lets Table 4 be re-run as closely as possible to how it was produced: same code, though GPU
nondeterminism remains. It is used by `bash scripts/reproduce_delfys75.sh table4`; see
[REPRODUCE_DELFYS75.md](../../REPRODUCE_DELFYS75.md).

Do not use this folder for new experiments; use the code in the repository root.
