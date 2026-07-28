"""Write one results/<run_id>.json per experiment - the paper's audit
trail. No number goes into the paper by hand; every table/figure is
regenerated from these files. See CLAUDE.md rule 7.

No model code, no plotting - this module only assembles and writes the
run record.

CAVEAT (2026-07-28) - the "all_hours" metrics block is mislabelled in
every run JSON written before this date by scripts/run_xgb_dev.py,
run_lstm_dev.py, run_cnn_lstm_dev.py, and run_residual_dev.py. It is
computed on the intersection of timestamps where the model AND
SmartPersistence AND Climatology AND ConvexCombination all produced a
prediction (see any of those scripts' "common_idx" construction).
Climatology has no prediction at (month, hour) cells that are never
daylight in training data - k_p is NaN by construction at night, see
src.models.climatology.Climatology.fit/predict - so that intersection
covers roughly 4247 of a year's 8694 hours, not a full 24-hour cycle.
The name says otherwise.

Do NOT rewrite the existing run JSONs to fix this: they are the audit
trail and are internally consistent (the block has always meant this
same quantity, whatever it is labelled). Runs written from 2026-07-28
onward use the key "common_hours" for this same quantity instead of
"all_hours" - if you are reading an older run JSON, "all_hours" IS
"common_hours" under the new name.

Neither "daylight" nor "common_hours"/"all_hours" answers the actual
night-inclusion question (how metrics behave when literal night hours
are included), because both are restricted to the four-way prediction
intersection above, which already excludes most true night rows via
Climatology's coverage gap. For that question, see
scripts/build_table4_protocol.py, which computes each evaluation
configuration from scratch using only the intersection it actually
needs (e.g. model + persistence only, with no daylight filter, to get
a metric that truly spans a 24-hour cycle) rather than reading it out
of run JSONs that were never designed to answer it. See also
paper/PROJECT_CHECKPOINT.md, Finding 2, for the closed-form prediction
RMSE_all = RMSE_day * sqrt(N_day / N_all) that motivated this script.
"""

import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REQUIRED_CONFIG_KEYS = ("model", "array", "horizon", "regime", "seed", "eval_split")
EVAL_SPLITS = ("val", "test")
REQUIRED_METRICS_KEYS = ("skill_vs_persistence", "skill_vs_convex", "convex_weight")
REQUIRED_TIMING_KEYS = (
    "fit_seconds",
    "predict_seconds",
    "n_train",
    "n_val",
    "n_test",
    # Count of timestamps dropped by src.eval.exclusions.exclusion_mask
    # (documented equipment outages, e.g. array17's 5-9 June 2015 - external
    # metadata, not a model-performance judgment). Required on every run so
    # a reader can see exactly how many hours were withheld and why,
    # without cross-referencing a separate log.
    "n_excluded_outage",
)


def make_run_id(config: dict) -> str:
    """Deterministic, human-readable run id from a config dict.

    Format: <model>_<array>_h<horizon>_<regime>_seed<seed>, e.g.
    xgboost_array11_h3_lagged_seed0. Requires config to contain
    'model', 'array', 'horizon', 'regime', 'seed', 'eval_split'.

    eval_split must be 'val' (development/tuning runs) or 'test' (the one
    final run per config, touched once at the end - CLAUDE.md research
    integrity rule). It is not part of the run id string itself, but is
    required in every config so aggregation scripts can filter on it and
    never accidentally mix a validation-split number with a test-split
    one. Raises ValueError if it is present but not one of those two
    values.
    """
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(f"config is missing required keys: {missing}")

    if config["eval_split"] not in EVAL_SPLITS:
        raise ValueError(
            f"config['eval_split'] must be one of {EVAL_SPLITS}, got "
            f"{config['eval_split']!r}"
        )

    return (
        f"{config['model']}_{config['array']}_h{config['horizon']}_"
        f"{config['regime']}_seed{config['seed']}"
    )


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def capture_environment() -> dict:
    """Snapshot the code and environment that is about to produce a result.

    git_dirty is True if the working tree has uncommitted changes. A
    commit hash recorded while the tree is dirty does NOT describe the
    code that actually ran - callers should treat such runs as
    unreproducible.
    """
    import pandas
    import pvlib
    import sklearn
    import statsmodels
    import torch
    import xgboost

    git_commit = _run_git("rev-parse", "HEAD")
    git_dirty = bool(_run_git("status", "--porcelain"))
    git_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_branch": git_branch,
        "python_version": platform.python_version(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
            "pvlib": pvlib.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "platform": platform.platform(),
        "gpu_name": gpu_name,
    }


def set_all_seeds(seed: int) -> dict:
    """Seed python random, numpy, and torch (CPU and CUDA). Returns a dict
    recording what was seeded, meant to be stored in config for the
    record.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = torch.cuda.is_available()
    if cuda_seeded:
        torch.cuda.manual_seed_all(seed)

    return {"seed": seed, "python": True, "numpy": True, "torch": True, "torch_cuda": cuda_seeded}


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain Python types and
    NaN/Inf floats to None, so json.dump never fails on a numpy type and
    never writes a non-standard NaN token.
    """
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        obj = int(obj)
    if isinstance(obj, (np.bool_,)):
        obj = bool(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _validate_metrics(metrics: dict) -> None:
    """Every run must record skill against BOTH reference forecasts, per
    scripts/compare_references.py: a run with only skill_vs_persistence
    cannot be used to reproduce that comparison later without re-running
    the whole grid (see that script's motivation, and CLAUDE.md rule 7).

    metrics is usually shaped as subset_name -> {metric_name: value}
    (e.g. "daylight"/"common_hours" in scripts/run_xgb_dev.py - older run
    JSONs use "all_hours" for the same "common_hours" quantity, see the
    module docstring CAVEAT), so each dict-valued entry is checked
    individually. If metrics has no dict-valued entries at all, it is
    itself the one metrics dict to check (this is what lets
    scripts/validate_runner.py's flat dummy metrics dict exercise this
    same check).
    """
    subsets = [v for v in metrics.values() if isinstance(v, dict)]
    targets = subsets if subsets else [metrics]
    for target in targets:
        missing = [k for k in REQUIRED_METRICS_KEYS if k not in target]
        if missing:
            raise KeyError(f"metrics is missing required keys: {missing}")


def write_run(
    config: dict,
    metrics: dict,
    timings: dict,
    extra: dict = None,
    results_dir: str = "results",
    overwrite: bool = False,
) -> Path:
    """Assemble and write results/<run_id>.json.

    Top-level keys: run_id, timestamp_utc, config, environment, metrics,
    timings, extra.

    timings must include fit_seconds, predict_seconds, n_train, n_val,
    n_test, n_excluded_outage (see src.eval.exclusions.exclusion_mask -
    applied identically to every model before metrics are computed).
    metrics (or each of its dict-valued entries, e.g.
    "daylight"/"common_hours" - see module docstring CAVEAT for what
    this block does and does not cover, and for older runs' "all_hours"
    key) must include skill_vs_persistence, skill_vs_convex,
    convex_weight - see _validate_metrics.

    Raises FileExistsError if results/<run_id>.json already exists and
    overwrite is False - a silently overwritten result is a lost
    experiment. Returns the path written.
    """
    missing = [k for k in REQUIRED_TIMING_KEYS if k not in timings]
    if missing:
        raise KeyError(f"timings is missing required keys: {missing}")

    _validate_metrics(metrics)

    run_id = make_run_id(config)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass overwrite=True to replace it "
            "(a silently overwritten result is a lost experiment)"
        )

    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": capture_environment(),
        "metrics": metrics,
        "timings": timings,
        "extra": extra if extra is not None else {},
    }
    record = _to_jsonable(record)

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)

    return out_path
