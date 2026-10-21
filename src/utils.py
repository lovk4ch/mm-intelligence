import json

from pathlib import Path
from typing import Any


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    )

def load_json(path: Path) -> dict:
    require_file(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path.resolve()}")