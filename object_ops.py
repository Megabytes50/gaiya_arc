"""
object_ops.py — object-level transformations, where most ARC rules actually live.

Built on perception.py (objects = connected same-colour regions). These are the
primitives that let the search express rules about THINGS, not just whole-grid
geometry: keep the biggest object, recolour objects, fill holes, gravity, etc.

TWO FLAVOURS (same split as dsl.py)
===================================
  * PARAMETER-FREE  grid -> grid     (keep_largest_object, drop_background, ...)
    Safe, cheap, move the solve number immediately. Built first.
  * FITTED          parameter LEARNED from the pairs (per-object colour map, a
    move vector). Built second, because "move the object" needs "move it WHERE",
    and the where is read off the examples, never guessed.

Discipline carried from perception: these operate on a grid and a chosen
background (the search's guess, verified by the pairs). Perception stays
rule-blind; the JUDGEMENT ("this is background", "keep the biggest") lives here,
in the transformation layer — exactly where judgement belongs.

Pure stdlib. Every function returns a NEW grid.
"""
from __future__ import annotations
from typing import Optional
from collections import Counter
import perception as P

Grid = list


def _dims(g: Grid) -> tuple:
    return (len(g), len(g[0]) if g else 0)

def _bg(g: Grid, background: Optional[int]) -> int:
    if background is not None:
        return background
    return Counter(v for row in g for v in row).most_common(1)[0][0]

def _blank(rows: int, cols: int, fill: int) -> Grid:
    return [[fill] * cols for _ in range(rows)]


# ──────────────────────────────────────────────────────────────────────────
# parameter-free object primitives
# ──────────────────────────────────────────────────────────────────────────
def keep_largest_object(g: Grid, background: Optional[int] = None,
                        connectivity: int = 8) -> Grid:
    """Blank everything except the single largest non-background object. A common
    ARC rule: 'the answer is the biggest thing.'"""
    bg = _bg(g, background)
    objs = P.objects(g, connectivity, background=bg)
    if not objs:
        return [row[:] for row in g]
    biggest = max(objs, key=lambda o: o.size)
    rows, cols = _dims(g)
    out = _blank(rows, cols, bg)
    for r, c in biggest.cells:
        out[r][c] = g[r][c]
    return out


def keep_smallest_object(g: Grid, background: Optional[int] = None,
                         connectivity: int = 8) -> Grid:
    """Blank everything except the single smallest non-background object."""
    bg = _bg(g, background)
    objs = P.objects(g, connectivity, background=bg)
    if not objs:
        return [row[:] for row in g]
    smallest = min(objs, key=lambda o: o.size)
    rows, cols = _dims(g)
    out = _blank(rows, cols, bg)
    for r, c in smallest.cells:
        out[r][c] = g[r][c]
    return out


def crop_to_largest_object(g: Grid, background: Optional[int] = None,
                           connectivity: int = 8) -> Grid:
    """Crop to the bounding box of the largest object (answer IS the biggest
    thing, tightly cropped). Very common."""
    bg = _bg(g, background)
    objs = P.objects(g, connectivity, background=bg)
    if not objs:
        return [row[:] for row in g]
    o = max(objs, key=lambda x: x.size)
    mnr, mnc, mxr, mxc = o.bounds
    return [g[r][mnc:mxc + 1] for r in range(mnr, mxr + 1)]


def most_common_object_shape(g: Grid, background: Optional[int] = None,
                             connectivity: int = 8) -> Grid:
    """Crop to one object whose SHAPE is the most common among all objects — the
    'odd one out' family, inverted (keep the typical one). Returns the typical
    object's patch. If shapes are all unique, falls back to largest."""
    bg = _bg(g, background)
    objs = P.objects(g, connectivity, background=bg)
    if not objs:
        return [row[:] for row in g]
    shape_counts = Counter(o.normalized() for o in objs)
    top_shape, _ = shape_counts.most_common(1)[0]
    typical = [o for o in objs if o.normalized() == top_shape]
    o = typical[0]
    mnr, mnc, mxr, mxc = o.bounds
    return [g[r][mnc:mxc + 1] for r in range(mnr, mxr + 1)]


def odd_one_out_shape(g: Grid, background: Optional[int] = None,
                      connectivity: int = 8) -> Grid:
    """Crop to the object whose shape is UNIQUE among objects that mostly share a
    shape — the classic 'find the different one' rule. If no unique one, largest."""
    bg = _bg(g, background)
    objs = P.objects(g, connectivity, background=bg)
    if not objs:
        return [row[:] for row in g]
    shape_counts = Counter(o.normalized() for o in objs)
    uniques = [o for o in objs if shape_counts[o.normalized()] == 1]
    o = uniques[0] if uniques else max(objs, key=lambda x: x.size)
    mnr, mnc, mxr, mxc = o.bounds
    return [g[r][mnc:mxc + 1] for r in range(mnr, mxr + 1)]


# ──────────────────────────────────────────────────────────────────────────
# gravity — pull all non-background cells in one direction
# ──────────────────────────────────────────────────────────────────────────
def gravity_down(g: Grid, background: Optional[int] = None) -> Grid:
    """Let every non-background cell fall to the bottom of its column (order
    preserved). 'Things drop.' A recurring ARC dynamic."""
    bg = _bg(g, background)
    rows, cols = _dims(g)
    out = _blank(rows, cols, bg)
    for c in range(cols):
        col_vals = [g[r][c] for r in range(rows) if g[r][c] != bg]
        for i, v in enumerate(reversed(col_vals)):
            out[rows - 1 - i][c] = v
    return out

def gravity_up(g: Grid, background: Optional[int] = None) -> Grid:
    bg = _bg(g, background)
    rows, cols = _dims(g)
    out = _blank(rows, cols, bg)
    for c in range(cols):
        col_vals = [g[r][c] for r in range(rows) if g[r][c] != bg]
        for i, v in enumerate(col_vals):
            out[i][c] = v
    return out

def gravity_left(g: Grid, background: Optional[int] = None) -> Grid:
    bg = _bg(g, background)
    rows, cols = _dims(g)
    out = _blank(rows, cols, bg)
    for r in range(rows):
        row_vals = [v for v in g[r] if v != bg]
        for i, v in enumerate(row_vals):
            out[r][i] = v
    return out

def gravity_right(g: Grid, background: Optional[int] = None) -> Grid:
    bg = _bg(g, background)
    rows, cols = _dims(g)
    out = _blank(rows, cols, bg)
    for r in range(rows):
        row_vals = [v for v in g[r] if v != bg]
        for i, v in enumerate(reversed(row_vals)):
            out[r][cols - 1 - i] = v
    return out


# ──────────────────────────────────────────────────────────────────────────
# the object-primitive registry (parameter-free) — added to the search
# ──────────────────────────────────────────────────────────────────────────
OBJECT_PARAM_FREE: dict = {
    "keep_largest_object": keep_largest_object,
    "keep_smallest_object": keep_smallest_object,
    "crop_to_largest_object": crop_to_largest_object,
    "most_common_object_shape": most_common_object_shape,
    "odd_one_out_shape": odd_one_out_shape,
    "gravity_down": gravity_down,
    "gravity_up": gravity_up,
    "gravity_left": gravity_left,
    "gravity_right": gravity_right,
}
