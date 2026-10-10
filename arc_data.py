"""
arc_data.py — load ARC tasks from the standard release layout.

Standard ARC on disk: a folder of per-task JSON files, each named <task_id>.json,
holding {"train": [{"input": grid, "output": grid}, ...],
        "test":  [{"input": grid, "output": grid}, ...]}.
Releases split tasks into training/ and evaluation/ subfolders.

Pure stdlib (json only) — no numpy, no pandas — because the eventual Kaggle run
is OFFLINE and compute-capped, and a loader with no heavy deps is one less thing
to break there. A grid is list[list[int]] throughout; we never convert to a
fancy type here, so what perception/heads receive is exactly the raw ARC grid.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

Grid = list              # list[list[int]]


@dataclass(frozen=True)
class Pair:
    """One input->output example. For a TEST pair the output may be None
    (hidden at competition time); for TRAIN it is always present."""
    input: Grid
    output: Optional[Grid]


@dataclass(frozen=True)
class Task:
    """One ARC task: a handful of train pairs (the rule to infer) and the test
    pair(s) to solve. `task_id` is the filename stem, for provenance."""
    task_id: str
    train: tuple          # tuple[Pair]
    test: tuple           # tuple[Pair]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)


def load_task(path: str | Path) -> Task:
    """Read one <task_id>.json into a Task."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    train = tuple(Pair(p["input"], p.get("output")) for p in raw.get("train", []))
    test = tuple(Pair(p["input"], p.get("output")) for p in raw.get("test", []))
    return Task(task_id=path.stem, train=train, test=test)


def load_split(folder: str | Path) -> dict:
    """Load every <task_id>.json in a folder (e.g. a training/ or evaluation/
    directory) into {task_id: Task}, sorted by id for stable ordering."""
    folder = Path(folder)
    tasks = {}
    for p in sorted(folder.glob("*.json")):
        tasks[p.stem] = load_task(p)
    return tasks


def load_dataset(root: str | Path) -> dict:
    """Load a full ARC release root that has training/ and evaluation/ subdirs.
    Returns {"training": {...}, "evaluation": {...}}. Missing subdirs are skipped
    (some custom layouts only have one)."""
    root = Path(root)
    out = {}
    for split in ("training", "evaluation"):
        sub = root / split
        if sub.is_dir():
            out[split] = load_split(sub)
    return out


# ── small helpers used everywhere downstream ────────────────────────────────
def dims(grid: Grid) -> tuple:
    return (len(grid), len(grid[0]) if grid else 0)


def grids_equal(a: Grid, b: Grid) -> bool:
    """Exact grid match — the ARC scoring criterion (a prediction is right only
    if it equals the truth cell-for-cell)."""
    return a == b
