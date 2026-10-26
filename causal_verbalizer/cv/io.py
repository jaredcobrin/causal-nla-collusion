"""Small shared helpers: config loading, hashing, JSON(L) files and the progress log."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
RUNS = ROOT / "runs"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "run.yaml").read_text())


def load_concepts() -> list[dict]:
    return yaml.safe_load((ROOT / "configs" / "concepts.yaml").read_text())


def h(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def seed_for(*parts) -> int:
    """Stable per-purpose seed, so each concept/stage has its own reproducible RNG."""
    return int(h("|".join(map(str, parts)), 8), 16)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class Log:
    """Timestamped progress log: printed and appended to <run>/progress.log."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "progress.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
