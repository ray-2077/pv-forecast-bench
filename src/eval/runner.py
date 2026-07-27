"""Write one results/<run_id>.json per experiment - the paper's audit
trail. No number goes into the paper by hand; every table/figure is
regenerated from these files. See CLAUDE.md rule 7.

No model code, no plotting - this module only assembles and writes the
run record.
"""

import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REQUIRED_CONFIG_KEYS = ("model", "array", "horizon", "regime", "seed")
REQUIRED_TIMING_KEYS = (
    "fit_seconds",
    "predict_seconds",
    "n_train",
    "n_val",
    "n_test",
)


def make_run_id(config: dict) -> str:
    """Deterministic, human-readable run id from a config dict.

    Format: <model>_<array>_h<horizon>_<regime>_seed<seed>, e.g.
    xgboost_array11_h3_lagged_seed0. Requires config to contain
    'model', 'array', 'horizon', 'regime', 'seed'.
    """
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise KeyError(f"config is missing required keys: {missing}")

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
    n_test.

    Raises FileExistsError if results/<run_id>.json already exists and
    overwrite is False - a silently overwritten result is a lost
    experiment. Returns the path written.
    """
    missing = [k for k in REQUIRED_TIMING_KEYS if k not in timings]
    if missing:
        raise KeyError(f"timings is missing required keys: {missing}")

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
