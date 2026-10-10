"""
perception.py — ARC structural perception: a raw grid -> honestly-derived facts.

THE DISCIPLINE (the field law, applied to grids)
================================================
Perception reports what is THERE. It never guesses what transformation the task
wants — that is the heads' job (reasoning). Every fact here must pass one test:

    "Is this true of EVERY possible ARC grid, blind to nuance, without presuming
    the task's rule?"

So colour, position, connected regions, symmetry-present, palette, bounds all
qualify — they are measurements. "The background", "the important object", "the
odd one out" do NOT — they are judgments, and they live in the heads.

Two honest consequences, both deliberate:
  * OBJECTS are reported under BOTH 4- and 8-connectivity. Which grouping a task
    rewards varies by task, so committing to one on the field would be a hidden
    judgment. We report both; a head chooses.
  * "background colour" is NOT asserted here. We report each colour's count; a
    head may decide which (if any) is background.

This module is pure, dependency-free (stdlib only — matters for the offline
Kaggle run), and testable on its own. It does not import the engine; it feeds
structure to the Gaiya field/heads built elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

Grid = list          # list[list[int]] — rows of colour values 0..9
Cell = tuple         # (row, col)


# ──────────────────────────────────────────────────────────────────────────
# basic shape facts
# ──────────────────────────────────────────────────────────────────────────
def dimensions(grid: Grid) -> tuple:
    """(rows, cols). True of any non-empty grid."""
    return (len(grid), len(grid[0]) if grid else 0)


def palette(grid: Grid) -> dict:
    """{colour: count} over all cells. Pure tally — no colour is privileged.
    A head may read this to decide a background; perception does not."""
    counts: dict = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return counts


# ──────────────────────────────────────────────────────────────────────────
# objects — connected regions of one colour (BOTH connectivities, see above)
# ──────────────────────────────────────────────────────────────────────────
_NEIGHBOURS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBOURS_8 = _NEIGHBOURS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass(frozen=True)
class Object:
    """One connected region of a single colour. A measurement, not a judgment:
    it says 'these same-coloured cells touch', nothing about importance."""
    colour: int
    cells: tuple              # frozenset-like tuple of (r,c), sorted
    connectivity: int         # 4 or 8 — which neighbour rule produced it

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def bounds(self) -> tuple:
        """(min_r, min_c, max_r, max_c) — the object's bounding box."""
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return (min(rs), min(cs), max(rs), max(cs))

    @property
    def shape(self) -> tuple:
        """(height, width) of the bounding box."""
        mnr, mnc, mxr, mxc = self.bounds
        return (mxr - mnr + 1, mxc - mnc + 1)

    def normalized(self) -> frozenset:
        """Cells shifted so the bounding box starts at (0,0). Lets two objects
        be compared for SAME SHAPE regardless of where they sit. A fact about
        form, still not a judgment about meaning."""
        mnr, mnc, _, _ = self.bounds
        return frozenset((r - mnr, c - mnc) for r, c in self.cells)


def objects(grid: Grid, connectivity: int = 4,
            background: Any = None) -> list:
    """Connected-component objects of one colour each.

    connectivity : 4 (edge-touch) or 8 (edge+diagonal touch). BOTH are valid
                readings of a grid; callers usually compute both and let a
                head choose — see objects_both().
    background   : OPTIONAL. If given, cells of that colour are not grouped into
                objects (common when a head has DECIDED a background). Default
                None means every colour forms objects — the rule-blind default.
    """
    nbrs = _NEIGHBOURS_8 if connectivity == 8 else _NEIGHBOURS_4
    rows, cols = dimensions(grid)
    seen = [[False] * cols for _ in range(rows)]
    found: list = []
    for r in range(rows):
        for c in range(cols):
            if seen[r][c]:
                continue
            colour = grid[r][c]
            if background is not None and colour == background:
                seen[r][c] = True
                continue
            # flood fill this colour from (r,c)
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and not seen[nr][nc] and grid[nr][nc] == colour):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            found.append(Object(colour=colour,
                                cells=tuple(sorted(comp)),
                                connectivity=connectivity))
    return found


def objects_both(grid: Grid, background: Any = None) -> dict:
    """Both readings at once: {4: [...], 8: [...]}. The field hands a head both
    groupings and the head decides which the task rewards — never the field."""
    return {4: objects(grid, 4, background),
            8: objects(grid, 8, background)}


# ──────────────────────────────────────────────────────────────────────────
# symmetry — present or not (a fact); meaning is a head's call
# ──────────────────────────────────────────────────────────────────────────
def _transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def symmetries(grid: Grid) -> dict:
    """Which exact symmetries the grid HAS. Each is a yes/no fact checked by
    comparison — not an assertion that the task is 'about' that symmetry.

        horizontal : grid equals its left-right mirror
        vertical   : grid equals its top-bottom mirror
        rot180     : grid equals itself rotated 180°
        diagonal   : grid equals its transpose (only if square)
        anti_diag  : grid equals its anti-transpose (only if square)
    """
    rows, cols = dimensions(grid)
    out = {}
    out["horizontal"] = all(grid[r] == grid[r][::-1] for r in range(rows))
    out["vertical"] = grid == grid[::-1]
    out["rot180"] = grid == [row[::-1] for row in grid[::-1]]
    if rows == cols:
        t = _transpose(grid)
        out["diagonal"] = grid == t
        anti = [list(row) for row in zip(*[row[::-1] for row in grid[::-1]])]
        out["anti_diag"] = grid == anti
    else:
        out["diagonal"] = False
        out["anti_diag"] = False
    return out


# ──────────────────────────────────────────────────────────────────────────
# a single rolled-up perception record for a grid
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Perception:
    """Everything perception honestly derives from ONE grid. Facts only.
    Heads read this; they are not bound by it and add no judgment to it."""
    grid: Grid
    dims: tuple
    palette: dict
    objects4: list
    objects8: list
    symmetry: dict

    def summary(self) -> dict:
        """A compact, inspectable view — for tests, traces, and the paper."""
        return {
            "dims": self.dims,
            "palette": self.palette,
            "n_objects_4": len(self.objects4),
            "n_objects_8": len(self.objects8),
            "symmetry": {k: v for k, v in self.symmetry.items() if v},
        }


def perceive(grid: Grid, background: Any = None) -> Perception:
    """Run the whole rule-blind perception pass over one grid."""
    both = objects_both(grid, background)
    return Perception(
        grid=grid,
        dims=dimensions(grid),
        palette=palette(grid),
        objects4=both[4],
        objects8=both[8],
        symmetry=symmetries(grid),
    )
