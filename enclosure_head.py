"""
enclosure_head.py — the first REAL domain head (Sonia's observation, 2026-10-12).

PUZZLE TYPE: enclosure colour-matching.
  A blue border spans edge-to-edge, enclosing colored "legend" clusters. Grey
  "target" clusters sit outside. Black = empty. The output is the input with each
  grey cluster recoloured to the legend colour it matches. The MATCHING KEY is a
  cluster's INTERIOR-HOLE COUNT: the number of black cells fully enclosed by that
  cluster's own cells (closed = 0 holes; open = >=1; the count is the size of the
  enclosed space). A grey cluster takes the colour of the legend cluster with the
  matching hole signature.

ARCHITECTURE (the point Sonia stressed):
  A HEAD = one invariant-detector + a GROWING LIST OF CONSTRUCTIONS. The head
  detects the type; then it tries its constructions in order (Least Effort:
  simplest first) and keeps the first that reproduces ALL train pairs. Adding a
  future construction for this puzzle type = appending to CONSTRUCTIONS, no
  rewrite. This is how one head accumulates ways-of-solving for its type.

Constructions implemented now (more can be appended):
  C1  direct hole-count match   — grey cluster -> legend colour with SAME holes.
                                   (covers the closed/open 2-cluster case AND the
                                    distinct-count case as one rule.)
  C2  ordinal-multiplier match  — the n x (n+1) / ordinal pattern Sonia observed
                                   for 3+ clusters; tried only if C1 fails.

The head reports a four-outcome verdict: SATISFIED (a construction reproduced all
train pairs -> output built), NOT_APPLICABLE (type guard fails), UNMET (type holds
but no current construction reproduces the pairs -> escalate / needs a new
construction). UNMET is the honest signal that this type needs another construction.

Pure stdlib, built on perception.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from collections import Counter
import perception as P

Grid = list
BLUE, GREY, BLACK = 1, 5, 0        # ARC conventions: 1=blue, 5=grey, 0=black


def _dims(g): return (len(g), len(g[0]) if g else 0)


# ──────────────────────────────────────────────────────────────────────────
# interior-hole count: black cells fully enclosed by a cluster's cells
# ──────────────────────────────────────────────────────────────────────────
def _interior_holes(grid: Grid, cells: set) -> int:
    """Count BLACK cells enclosed by this cluster: black cells inside the
    cluster's bounding box that cannot reach the box border through black cells
    without crossing a cluster cell. Returns the number of such enclosed black
    cells (the 'space' size). closed cluster -> 0."""
    if not cells:
        return 0
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    mnr, mxr, mnc, mxc = min(rs), max(rs), min(cs), max(cs)
    H, W = mxr - mnr + 1, mxc - mnc + 1
    # local box; mark cluster cells as walls
    wall = [[False] * W for _ in range(H)]
    for (r, c) in cells:
        wall[r - mnr][c - mnc] = True
    # flood black from the box border; anything black NOT reached is enclosed
    outside = [[False] * W for _ in range(H)]
    stack = []
    for r in range(H):
        for c in (0, W - 1):
            if not wall[r][c] and not outside[r][c]:
                outside[r][c] = True; stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if not wall[r][c] and not outside[r][c]:
                outside[r][c] = True; stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and not wall[nr][nc] and not outside[nr][nc]:
                outside[nr][nc] = True; stack.append((nr, nc))
    enclosed = 0
    for r in range(H):
        for c in range(W):
            if not wall[r][c] and not outside[r][c]:
                enclosed += 1
    return enclosed


# ──────────────────────────────────────────────────────────────────────────
# scene parse: border, inside legend clusters, outside grey clusters
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Cluster:
    colour: int
    cells: tuple
    holes: int


def _blue_border_box(grid: Grid):
    """Find a blue BORDER that encloses a region: the bounding box of the blue
    cells, required to be a genuine frame (blue present on all four sides of that
    box). Returns (mnr, mnc, mxr, mxc) of the enclosed interior, or None.
    This is a STRUCTURAL test — not 'blue exists somewhere'."""
    rows, cols = _dims(grid)
    blue = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == BLUE]
    if len(blue) < 8:                      # a real border is not a few stray cells
        return None
    rs = [r for r, _ in blue]; cs = [c for _, c in blue]
    mnr, mxr, mnc, mxc = min(rs), max(rs), min(cs), max(cs)
    if (mxr - mnr) < 2 or (mxc - mnc) < 2:
        return None                        # too small to enclose anything
    # require blue present along each of the four sides of its bounding box
    top = any(grid[mnr][c] == BLUE for c in range(mnc, mxc + 1))
    bot = any(grid[mxr][c] == BLUE for c in range(mnc, mxc + 1))
    left = any(grid[r][mnc] == BLUE for r in range(mnr, mxr + 1))
    right = any(grid[r][mxc] == BLUE for r in range(mnr, mxr + 1))
    if not (top and bot and left and right):
        return None
    return (mnr, mnc, mxr, mxc)


def _inside(box, r, c):
    mnr, mnc, mxr, mxc = box
    return mnr < r < mxr and mnc < c < mxc


def _parse_scene(grid: Grid):
    """Split into legend (colored, INSIDE the blue border) and target (grey,
    OUTSIDE) clusters, each with its interior-hole count. Returns
    (legend, grey) or None if the STRUCTURE isn't a real enclosure scene:
      - black must be the dominant/background colour;
      - a genuine blue border box must exist (see _blue_border_box);
      - colored legend clusters sit inside it; grey clusters sit outside."""
    pal = Counter(v for row in grid for v in row)
    if pal.get(BLACK, 0) == 0 or pal.most_common(1)[0][0] != BLACK:
        return None                        # black must be the background
    if BLUE not in pal or GREY not in pal:
        return None
    box = _blue_border_box(grid)
    if box is None:
        return None
    objs = P.objects(grid, connectivity=8, background=BLACK)
    legend, grey = [], []
    for o in objs:
        if o.colour == BLUE:
            continue
        cells = set(o.cells)
        # a cluster is 'inside' if the majority of its cells are inside the box
        n_in = sum(1 for (r, c) in cells if _inside(box, r, c))
        inside = n_in > len(cells) / 2
        h = _interior_holes(grid, cells)
        cl = Cluster(colour=o.colour, cells=o.cells, holes=h)
        if o.colour == GREY and not inside:
            grey.append(cl)
        elif o.colour != GREY and inside:
            legend.append(cl)
    if not legend or not grey:
        return None
    return legend, grey


# ──────────────────────────────────────────────────────────────────────────
# CONSTRUCTIONS (the growing list — append future ones here)
# ──────────────────────────────────────────────────────────────────────────
def _paint(grid: Grid, cells, colour) -> Grid:
    out = [row[:] for row in grid]
    for (r, c) in cells:
        out[r][c] = colour
    return out


def c1_direct_hole_match(grid: Grid) -> Optional[Grid]:
    """C1: each grey cluster -> the legend colour whose INTERIOR-HOLE COUNT equals
    the grey cluster's. Covers closed/open (0 vs >=1) and distinct-count cases."""
    parsed = _parse_scene(grid)
    if parsed is None:
        return None
    legend, grey = parsed
    by_holes = {}
    for cl in legend:
        by_holes.setdefault(cl.holes, cl.colour)
    out = grid
    painted = 0
    for g in grey:
        if g.holes in by_holes:
            out = _paint(out, g.cells, by_holes[g.holes])
            painted += 1
    return out if painted == len(grey) else None   # must colour EVERY grey cluster


def c2_ordinal_multiplier(grid: Grid) -> Optional[Grid]:
    """C2: the ordinal pattern Sonia observed — legend clusters sorted by holes
    get an ordinal multiplier (1,2,3, cycling), the product keys the grey match.
    Tried only if C1 fails. (Kept faithful to the observation; the train pairs
    decide if it's real via the head's verify step.)"""
    parsed = _parse_scene(grid)
    if parsed is None:
        return None
    legend, grey = parsed
    legend_sorted = sorted(legend, key=lambda c: c.holes)
    # product key: holes * ordinal (1-based), ordinal cycles 1,2,3,1,2,3...
    key_to_colour = {}
    for i, cl in enumerate(legend_sorted):
        ordinal = (i % 3) + 1
        key_to_colour[cl.holes * ordinal] = cl.colour
    out = grid
    painted = 0
    for g in grey:
        # a grey cluster's key is its own holes (ordinal 1 for targets)
        if g.holes in key_to_colour:
            out = _paint(out, g.cells, key_to_colour[g.holes])
            painted += 1
    return out if painted == len(grey) else None


# the GROWING construction list — append future constructions for this type here.
CONSTRUCTIONS: list = [
    ("direct_hole_match", c1_direct_hole_match),
    ("ordinal_multiplier", c2_ordinal_multiplier),
]


# ──────────────────────────────────────────────────────────────────────────
# the head: detect the type, then try constructions in order (Least Effort)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class HeadVerdict:
    outcome: str                 # SATISFIED | NOT_APPLICABLE | UNMET
    construction: Optional[str]  # which construction reproduced the pairs
    build: Optional[Callable]    # construction fn to apply to the test input


def applies(task) -> bool:
    """Type guard: does the enclosure scene parse on the first train input?"""
    return _parse_scene(task.train[0].input) is not None


def run(task) -> HeadVerdict:
    """Detect the type; if it holds, try each construction in order and keep the
    first that reproduces EVERY train pair. SATISFIED -> build the test output.
    UNMET -> type holds but no construction fits (needs a new construction)."""
    if not applies(task):
        return HeadVerdict("NOT_APPLICABLE", None, None)
    for name, fn in CONSTRUCTIONS:
        ok = True
        for p in task.train:
            try:
                if fn(p.input) != p.output:
                    ok = False; break
            except Exception:
                ok = False; break
        if ok:
            return HeadVerdict("SATISFIED", name, fn)
    return HeadVerdict("UNMET", None, None)


def solve(task) -> Optional[Grid]:
    """Full head solve: verdict -> apply the fitting construction to the test."""
    v = run(task)
    if v.outcome == "SATISFIED":
        try:
            return v.build(task.test[0].input)
        except Exception:
            return None
    return None
