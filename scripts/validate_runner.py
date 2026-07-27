"""Validate src/eval/runner.py.

1. Writes a dummy run with fabricated metrics to results/_test_run.json.
2. Reads it back and prints it.
3. Asserts that writing the same run_id again raises FileExistsError.
4. Asserts git_dirty is a boolean and git_commit is a 40-char hex string.
5. Deletes results/_test_run.json afterwards so it is not committed.

These metrics are FABRICATED for the purpose of testing the writer only -
per CLAUDE.md, no fabricated number may ever appear in results/ for real,
which is why this file deletes itself at the end.

Usage:
    python scripts/validate_runner.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.runner import make_run_id, write_run

RESULTS_DIR = REPO_ROOT / "results"

DUMMY_CONFIG = {
    "model": "_test",
    "array": "run",
    "horizon": 0,
    "regime": "dummy",
    "seed": 0,
    "eval_split": "val",
}
TEST_RUN_ID = make_run_id(DUMMY_CONFIG)
TEST_RUN_PATH = RESULTS_DIR / f"{TEST_RUN_ID}.json"
DUMMY_METRICS = {
    "mae": 0.1,
    "rmse": 0.2,
    "skill_vs_persistence": 0.5,
    "skill_vs_convex": 0.3,
    "convex_weight": 0.7,
}
DUMMY_TIMINGS = {
    "fit_seconds": 1.0,
    "predict_seconds": 0.1,
    "n_train": 10,
    "n_val": 5,
    "n_test": 5,
    "n_excluded_outage": 0,
}


def _check(label, condition) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


def main() -> None:
    results = []

    # Clean up any leftover from a previous failed run before starting.
    if TEST_RUN_PATH.exists():
        TEST_RUN_PATH.unlink()

    print("=" * 60)
    print("write_run: first write")
    print("=" * 60)

    written_path = write_run(
        DUMMY_CONFIG, DUMMY_METRICS, DUMMY_TIMINGS, results_dir=str(RESULTS_DIR)
    )
    results.append(_check("write_run wrote results/_test_run.json", written_path == TEST_RUN_PATH))
    results.append(_check("file exists on disk", TEST_RUN_PATH.exists()))

    print("\n" + "=" * 60)
    print("read back")
    print("=" * 60)

    with open(TEST_RUN_PATH) as f:
        record = json.load(f)
    print(json.dumps(record, indent=2, sort_keys=True))

    print("\n" + "=" * 60)
    print("checks")
    print("=" * 60)

    try:
        write_run(DUMMY_CONFIG, DUMMY_METRICS, DUMMY_TIMINGS, results_dir=str(RESULTS_DIR))
        results.append(_check("re-writing same run_id raises FileExistsError", False))
    except FileExistsError:
        results.append(_check("re-writing same run_id raises FileExistsError", True))

    env = record["environment"]
    results.append(_check("git_dirty is a boolean", isinstance(env["git_dirty"], bool)))
    results.append(
        _check(
            "git_commit is a 40-char hex string",
            bool(re.fullmatch(r"[0-9a-f]{40}", env["git_commit"])),
        )
    )

    # Cleanup so the fabricated test result is never committed.
    TEST_RUN_PATH.unlink()
    results.append(_check("results/_test_run.json deleted", not TEST_RUN_PATH.exists()))

    print()
    if all(results):
        print("All runner checks PASSED.")
    else:
        print("Some runner checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
