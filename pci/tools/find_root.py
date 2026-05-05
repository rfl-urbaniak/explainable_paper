from pathlib import Path


def find_repo_root(start: Path = Path.cwd()) -> Path:
    """Find the root of the current Git repository by looking for .git directory."""

    for path in [start] + list(start.parents):
        if (path / ".git").exists():
            return path

    raise RuntimeError("No .git directory found in this or any parent directories.")
