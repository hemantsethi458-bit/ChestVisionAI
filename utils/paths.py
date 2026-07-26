"""Project path resolution utilities."""

from pathlib import Path

from configs.config import PROJECT_ROOT


def get_project_root() -> Path:
    """Return the absolute project root directory."""
    return PROJECT_ROOT


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(relative_path: str) -> Path:
    """Resolve a project-relative path to an absolute path."""
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
