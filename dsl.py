"""
dsl.py — the transformation vocabulary the inner-shell search composes from.

WHAT A DSL IS, PLAINLY
======================
A DSL ("domain-specific language") here is just a LIBRARY OF SMALL, EXACT GRID
TRANSFORMATIONS. Each primitive is a function: grid -> grid. Nothing fancy, no
learning inside a primitive — each one does one definite, reversible-to-reason-
about thing (rotate, flip, recolour, tile, crop). The POWER is in COMPOSING them:
"flip, then recolour" solves tasks no single primitive solves. The inner-shell
search (search.py) strings primitives together to fit a task's train pairs.

WHY THESE PRIMITIVES
====================
They are chosen because they capture the rule-types that actually recur across
ARC: symmetry/orientation, colour relabeling, self-tiling (the 007bbfb7 fractal),
and cropping to content. We start with the high-frequency ones and grow the set
as real tasks show us what's missing — exactly the "discover what's missing by
using it" discipline, applied to primitives.

Two kinds of primitive:
  * PARAMETER-FREE  g -> g          (rotate90, flip_h, transpose, ...)
  * PARAMETERIZED   derived from the train pairs, then applied
                    (recolour needs a colour map; tile/scale need a factor).
    These are built by a FITTER that looks at (input, output) and proposes the
    parameter — so the search doesn't guess blindly.

Pure stdlib. A grid is list[list[int]]. Every function returns a NEW grid; none
mutates its input (the X-gate instinct: a transform never alters what it was
given — it produces a fresh field).
"""
from __future__ import annotations
from typing import Callable, Optional
from collections import Counter

Grid = list


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def dims(g: Grid) -> tuple:
    return (len(g), len(g[0]) if g else 0)


def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def most_common_colour(g: Grid) -> int:
    """The modal colour — a candidate background. NOTE: this is the search's
    guess, not a perception fact (perception stays rule-blind); a fitter may use
    it to propose a background, and the train pairs confirm or reject it."""
    c = Counter(v for row in g for v in row)
    return c.most_common(1)[0][0]


# ──────────────────────────────────────────────────────────────────────────
# parameter-free geometric primitives
# ──────────────────────────────────────────────────────────────────────────
def identity(g: Grid) -> Grid:
    return copy_grid(g)

def rotate90(g: Grid) -> Grid:
    """90° clockwise."""
    return [list(row) for row in zip(*g[::-1])]

def rotate180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]

def rotate270(g: Grid) -> Grid:
    """90° counter-clockwise."""
    return [list(row) for row in zip(*g)][::-1]

def flip_h(g: Grid) -> Grid:
    """Mirror left-right."""
    return [row[::-1] for row in g]

def flip_v(g: Grid) -> Grid:
    """Mirror top-bottom."""
    return g[::-1]

def transpose(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)]


# ──────────────────────────────────────────────────────────────────────────
# cropping to content (background-aware)
# ──────────────────────────────────────────────────────────────────────────
def crop_to_content(g: Grid, background: Optional[int] = None) -> Grid:
    """Crop to the bounding box of all non-background cells. If the grid is all
    background, return it unchanged. background defaults to the modal colour."""
    if background is None:
        background = most_common_colour(g)
    rows, cols = dims(g)
    cells = [(r, c) for r in range(rows) for c in range(cols)
            if g[r][c] != background]
    if not cells:
        return copy_grid(g)
    mnr = min(r for r, _ in cells); mxr = max(r for r, _ in cells)
    mnc = min(c for _, c in cells); mxc = max(c for _, c in cells)
    return [g[r][mnc:mxc + 1] for r in range(mnr, mxr + 1)]


# ──────────────────────────────────────────────────────────────────────────
# scaling and tiling
# ──────────────────────────────────────────────────────────────────────────
def scale_up(g: Grid, fr: int, fc: int) -> Grid:
    """Blow up each cell into an fr×fc block of itself. (A 2x scale turns one
    cell into a 2×2 patch of the same colour.)"""
    out = []
    for row in g:
        big_row = []
        for v in row:
            big_row.extend([v] * fc)
        for _ in range(fr):
            out.append(big_row[:])
    return out

def tile(g: Grid, tr: int, tc: int) -> Grid:
    """Repeat the whole grid tr times down and tc times across."""
    out = []
    for _ in range(tr):
        for row in g:
            out.append(row * tc)
    return out

def self_fractal(g: Grid, background: Optional[int] = None) -> Grid:
    """The 007bbfb7 rule: for an r×c grid, build an (r*r)×(c*c) grid where each
    NON-background cell is replaced by a copy of the whole grid, and each
    background cell by an all-background block. 'Tile the grid by itself, gated
    by the grid.' A composed idea expressed as one primitive because it recurs."""
    if background is None:
        background = most_common_colour(g)
    rows, cols = dims(g)
    big = [[background] * (cols * cols) for _ in range(rows * rows)]
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != background:
                for rr in range(rows):
                    for cc in range(cols):
                        big[r * rows + rr][c * cols + cc] = g[rr][cc]
    return big


# ──────────────────────────────────────────────────────────────────────────
# recolour (parameterized — needs a colour map fitted from a train pair)
# ──────────────────────────────────────────────────────────────────────────
def apply_colour_map(g: Grid, cmap: dict) -> Grid:
    """Relabel colours by cmap; colours absent from cmap pass through unchanged."""
    return [[cmap.get(v, v) for v in row] for row in g]

def fit_colour_map(inp: Grid, out: Grid) -> Optional[dict]:
    """Try to LEARN a colour->colour relabeling that turns inp into out. Works
    only if inp and out have identical shape and the mapping is CONSISTENT
    (every occurrence of colour X in inp maps to the same colour in out). Returns
    the map, or None if no consistent relabeling exists. This is how the search
    gets a recolour parameter from the examples instead of guessing it."""
    if dims(inp) != dims(out):
        return None
    cmap: dict = {}
    rows, cols = dims(inp)
    for r in range(rows):
        for c in range(cols):
            a, b = inp[r][c], out[r][c]
            if a in cmap:
                if cmap[a] != b:
                    return None            # inconsistent -> not a recolour rule
            else:
                cmap[a] = b
    return cmap


# ──────────────────────────────────────────────────────────────────────────
# the primitive registry — what the search draws from
# ──────────────────────────────────────────────────────────────────────────
# Parameter-free primitives: name -> fn(grid)->grid
PARAM_FREE: dict = {
    "identity": identity,
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "crop_to_content": crop_to_content,
    "self_fractal": self_fractal,
}
