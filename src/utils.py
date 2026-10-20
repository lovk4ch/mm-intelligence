from pathlib import Path


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path.resolve()}")