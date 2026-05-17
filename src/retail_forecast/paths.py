"""Repository root and default paths for data, config, and outputs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def resolve_project_path(rel: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    path = Path(rel)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def path_relative_to_project(path: Path | str) -> str:
    """Return a repo-relative POSIX path for portable JSON/logs."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()
