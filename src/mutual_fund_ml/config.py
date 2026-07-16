from pathlib import Path
import yaml
from typing import Any, Dict

def get_project_root() -> Path:
    """Finds and returns the project root directory."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "run_pipeline.py").exists():
            return parent
    # Fallback to current working directory if not found in parent path
    return Path.cwd()

def load_config() -> Dict[str, Any]:
    """Loads and returns the project YAML configuration."""
    root = get_project_root()
    config_path = root / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

def get_resolved_path(relative_path: str) -> Path:
    """Resolves a relative path to the absolute path starting from project root."""
    return get_project_root() / relative_path
