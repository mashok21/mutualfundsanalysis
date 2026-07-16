import ast
import hashlib
import re
from pathlib import Path

import pandas as pd

from mutual_fund_ml.config import load_config, get_resolved_path


def _sha256_of_dir(dir_path: Path) -> dict:
    """Returns a mapping of filename -> SHA-256 hash for every file in a directory."""
    hashes = {}
    for f in sorted(dir_path.glob("*")):
        if f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            hashes[f.name] = h.hexdigest()
    return hashes


def test_raw_file_immutability():
    """Asserts that routine raw-data loading never mutates files under data/raw/."""
    config = load_config()
    raw_dir = get_resolved_path(config["paths"]["raw_data_dir"])

    if not raw_dir.exists() or not any(raw_dir.glob("*")):
        return  # No local proprietary raw data present (e.g. a clean clone) - nothing to guard.

    before = _sha256_of_dir(raw_dir)

    from mutual_fund_ml.data_loading import load_raw_data
    load_raw_data()

    after = _sha256_of_dir(raw_dir)

    assert before == after, "Raw data files under data/raw/ were modified by a routine load operation!"


def test_panel_key_uniqueness():
    """Asserts the Scheme Name + TS_Month End composite panel key has zero duplicate rows."""
    config = load_config()
    raw_dir = get_resolved_path(config["paths"]["raw_data_dir"])
    raw_path = raw_dir / config["paths"]["raw_filename"]

    if not raw_path.exists():
        return  # No local proprietary raw data present - nothing to guard.

    df_raw = pd.read_excel(raw_path, sheet_name="Rawdata")
    dup_count = int(df_raw.duplicated(subset=["Scheme Name", "TS_Month End"]).sum())
    assert dup_count == 0, f"Found {dup_count} duplicate Scheme Name + TS_Month End panel keys in raw data!"


def test_forecasting_holdout_never_fit():
    """
    Statically guards the frozen January-April 2026 forecasting holdout.

    Verifies, via source-code analysis of run_final_evaluation.py, that the
    holdout-derived variables (test_mask / df_test / X_test / y_test) are
    never passed into a `.fit(...)` call - i.e. the frozen holdout is only
    ever used for `.predict()`, never for training or retuning.

    This test performs static analysis only; it does not execute
    run_final_evaluation.py, which is intentionally never run outside a
    deliberate, locked final-evaluation event.
    """
    script_path = Path(__file__).resolve().parent.parent / "run_final_evaluation.py"
    assert script_path.exists(), "run_final_evaluation.py is missing - cannot verify the holdout guard!"

    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    holdout_identifiers = {"test_mask", "df_test", "X_test", "y_test"}

    violations = []
    for node in ast.walk(tree):
        is_fit_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        )
        if not is_fit_call:
            continue

        referenced_names = set()
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Name):
                    referenced_names.add(sub.id)

        if referenced_names & holdout_identifiers:
            violations.append(sorted(referenced_names & holdout_identifiers))

    assert not violations, (
        f"Holdout-derived variable(s) passed into a .fit() call in run_final_evaluation.py: {violations}. "
        "The frozen January-April 2026 holdout must never be used for training or retuning."
    )

    # Confirm the holdout window boundaries in config match what the script encodes.
    config = load_config()
    chron = config["parameters"]["chronological_split"]
    assert chron["val_end"] == 202512, "val_end boundary has shifted - holdout window definition changed!"
    assert chron["test_end"] == 202604, "test_end boundary has shifted - holdout window definition changed!"
    assert re.search(r"test_mask\s*=.*>=\s*202601", source), (
        "Expected holdout threshold (target_month >= 202601) not found in run_final_evaluation.py - "
        "the holdout split logic may have been changed."
    )
