"""
fill_ops.py — region-filling, symmetry-completion, and colour primitives.

High-value ARC rule-types that geometry and object-selection alone can't express:
  * FILL ENCLOSED REGIONS — colour the holes inside closed shapes.
  * SYMMETRY COMPLETION    — a grid is a broken symmetric pattern; restore it.
  * COLOUR SWAP            — exchange two colours.
  * BORDER                 — frame the grid.

Same discipline: pure stdlib, new grid each time, background is the search's
guess verified by the pairs. Built on perception for connectivity flood-fill.
"""
from __future__ import annotations
from typing import Optional
from collections import Counter

Grid = list

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _dims(g): return (len(g), len(g[0]) if g else 0)
def _bg(g, b): return b if b is not None else Counter(v for row in g for v in row).most_common(1)[0][0]
def _copy(g): return [row[:] for row in g]


# ──────────────────────────────────────────────────────────────────────────
# fill enclosed regions
# ──────────────────────────────────────────────────────────────────────────
def fill_enclosed(g: Grid, fill_colour: Optional[int] = None,
                  background: Optional[int] = None) -> Grid:
    """Find background cells that are ENCLOSED (cannot reach the grid border via
    4-connected background), and recolour them. The classic 'fill the inside of
    the shape' rule. If fill_colour is None, uses the single non-background
    colour present (common case: one shape colour)."""
    bg = _bg(g, background)
    rows, cols = _dims(g)
    # flood from the border through background: anything reached is OUTSIDE.
    outside = [[False] * cols for _ in range(rows)]
    stack = []
    for r in range(rows):
        for c in (0, cols - 1):
            if g[r][c] == bg and not outside[r][c]:
                outside[r][c] = True; stack.append((r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if g[r][c] == bg and not outside[r][c]:
                outside[r][c] = True; stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] == bg and not outside[nr][nc]:
                outside[nr][nc] = True; stack.append((nr, nc))
    # enclosed = background cells never reached from the border
    if fill_colour is None:
        non_bg = [v for row in g for v in row if v != bg]
        if not non_bg:
            return _copy(g)
        fill_colour = Counter(non_bg).most_common(1)[0][0]
    out = _copy(g)
    for r in range(rows):
        for c in range(cols):
            if g[r][c] == bg and not outside[r][c]:
                out[r][c] = fill_colour
    return out


# ──────────────────────────────────────────────────────────────────────────
# symmetry completion — restore a broken symmetric pattern
# ──────────────────────────────────────────────────────────────────────────
def _complete_with(g: Grid, mirror, bg: int) -> Grid:
    """Overlay g with its mirror image: wherever g is background but the mirror
    has a colour, take the mirror's colour. Restores a pattern broken by holes."""
    rows, cols = _dims(g)
    m = mirror(g)
    out = _copy(g)
    for r in range(rows):
        for c in range(cols):
            if out[r][c] == bg and m[r][c] != bg:
                out[r][c] = m[r][c]
    return out

def complete_h_symmetry(g: Grid, background: Optional[int] = None) -> Grid:
    """Restore left-right symmetry by filling holes from the mirror."""
    bg = _bg(g, background)
    return _complete_with(g, lambda x: [row[::-1] for row in x], bg)

def complete_v_symmetry(g: Grid, background: Optional[int] = None) -> Grid:
    """Restore top-bottom symmetry."""
    bg = _bg(g, background)
    return _complete_with(g, lambda x: x[::-1], bg)

def complete_rot180_symmetry(g: Grid, background: Optional[int] = None) -> Grid:
    """Restore 180° rotational symmetry."""
    bg = _bg(g, background)
    return _complete_with(g, lambda x: [row[::-1] for row in x[::-1]], bg)

def complete_all_symmetry(g: Grid, background: Optional[int] = None) -> Grid:
    """Restore symmetry under H, V, and rot180 together (the strongest completion;
    fills a hole from whichever mirror supplies a colour)."""
    bg = _bg(g, background)
    g = complete_h_symmetry(g, bg)
    g = complete_v_symmetry(g, bg)
    g = complete_rot180_symmetry(g, bg)
    return g


# ──────────────────────────────────────────────────────────────────────────
# border / frame
# ──────────────────────────────────────────────────────────────────────────
def add_border(g: Grid, colour: Optional[int] = None) -> Grid:
    """Wrap the grid in a one-cell frame. Colour defaults to the least-common
    colour present (often the 'marker' colour)."""
    rows, cols = _dims(g)
    if colour is None:
        cnt = Counter(v for row in g for v in row)
        colour = cnt.most_common()[-1][0]
    out = [[colour] * (cols + 2)]
    for row in g:
        out.append([colour] + row[:] + [colour])
    out.append([colour] * (cols + 2))
    return out


# ──────────────────────────────────────────────────────────────────────────
# fitted: colour swap / per-colour recolour learned from the pairs
# ──────────────────────────────────────────────────────────────────────────
def swap_two_colours(g: Grid, a: int, b: int) -> Grid:
    """Exchange colours a and b everywhere."""
    out = []
    for row in g:
        out.append([b if v == a else (a if v == b else v) for v in row])
    return out


def fill_enclosed_with(g: Grid, fill_colour: int, background: int) -> Grid:
    """fill_enclosed with an EXPLICIT fill colour and background (for the fitted
    version: the search learns the colour from the pairs, not a guess)."""
    return fill_enclosed(g, fill_colour=fill_colour, background=background)


def complete_periodic(g: Grid, ph: int, pw: int, background: Optional[int] = None) -> Grid:
    """Treat the grid as a PERIODIC pattern with period (ph, pw): the true colour
    at (r,c) is the majority non-background colour seen at all cells sharing
    (r mod ph, c mod pw). Fill every background hole with that periodic colour.
    This is the 'repeating pattern with holes punched in it' rule — different from
    mirror symmetry, and very common in ARC."""
    bg = _bg(g, background)
    rows, cols = _dims(g)
    # for each residue class, the consensus colour (ignore background holes)
    consensus: dict = {}
    votes: dict = {}
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != bg:
                key = (r % ph, c % pw)
                votes.setdefault(key, Counter())[g[r][c]] += 1
    for key, cnt in votes.items():
        consensus[key] = cnt.most_common(1)[0][0]
    out = _copy(g)
    for r in range(rows):
        for c in range(cols):
            if out[r][c] == bg:
                key = (r % ph, c % pw)
                if key in consensus:
                    out[r][c] = consensus[key]
    return out


# registry of parameter-free fill/symmetry primitives
FILL_PARAM_FREE: dict = {
    "fill_enclosed": fill_enclosed,
    "complete_h_symmetry": complete_h_symmetry,
    "complete_v_symmetry": complete_v_symmetry,
    "complete_rot180_symmetry": complete_rot180_symmetry,
    "complete_all_symmetry": complete_all_symmetry,
    "add_border": add_border,
}
