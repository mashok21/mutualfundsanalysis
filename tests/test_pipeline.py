import os
from pathlib import Path
from mutual_fund_ml.config import load_config, get_project_root

def test_pipeline_config():
    root = get_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()

    config = load_config()
    assert "paths" in config
    assert "schema" in config
    assert "parameters" in config
