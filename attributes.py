"""
attributes.py — LAYER-2b: the ATTRIBUTE vocabulary + attribute recombination
(Sonia's design, 2026-10-16).

WHY THIS LAYER EXISTS
=====================
Our invariants (relational_ops.py) are MONOLITHIC: each seals a PERCEPTION (find
the lattice, find the enclosed region, find the period) and a CONSTRUCTION (fill,
stamp, project) into one unit that fires only on its own narrow family. Running
whole-invariant recombination showed WHY that stalls: no task has two whole
invariants co-firing, so there is nothing to compose. (Measured 2026-10-16:
0 tasks with >=2 operator invariants held.)

The reusable unit is FINER than an invariant. A new puzzle is understood by
recombining perceptual ATTRIBUTES that each already occur in the library, even
when no whole invariant fires. Example (06df4c85): a box-grid LATTICE (attribute
A, also in region_count_descale / grey_tiling) + same-axis same-colour
FILL-TO-CLOSE between matching pairs (attribute B, = connect_pairs/span_fill
lifted to lattice cells). Neither whole invariant fires; both ATTRIBUTES are
ours. "Context can be recombined" — Sonia.

THE VOCABULARY (grounded in 5 verified solves, 2026-10-16)
==========================================================
Each attribute is a typed, parameterised perceptual/constructive primitive.
Params are FITTED per task (learned), never hard-coded — the free params are the
recombination surface. The ones the five solves exercised:

  PARTITION
    lattice_partition(bar_colour)        -> cell grid (bars = full-length lines)
    enclosure(bg)                        -> cells enclosed by non-bg (border flood)
  AXIS  (the key generalisation: row | col | diag-down | diag-up)
    axis_residue(axis, period)           -> residue class of each cell
    axis_pairs(axis)                     -> matched same-colour marks on an axis
  MARK / FILL
    fill_between(colour)                 -> close the gap between matched marks
    fill_region(colour)                  -> paint a detected region
    project_template(shape, dir, step)   -> stamp recoloured copies along a dir
  TRANSFORM
    recolour(cmap)                       -> remap colours
    grow(out_h, out_w)                   -> extend canvas, continue periodicity

A CONSTRUCTION is now a COMPOSITION of attributes with fitted params. The
recombination gate is: which attributes are DETECTABLE in this grid (cheap
structural tests) — only those participate, and the fit searches param
combinations. The honesty gate is unchanged: a composition ships only if it
reproduces EVERY train pair byte-exact.

HONEST SCOPE
============
This is the hand-grounded vocabulary from 5 solves. It is NOT claimed complete.
The point phase-1 proves: the SAME small attribute set, recombined with fitted
params, reproduces 5 tasks that 20 whole invariants could not — and generalises
(every solve verified on the held-out TEST pair, not just train).

Pure stdlib.
"""
from __future__ import annotations
from collections import deque, defaultdict, Counter
from typing import Optional

Grid = list
BG = 0


def _dims(g): return (len(g), len(g[0]) if g else 0)


# ──────────────────────────────────────────────────────────────────────────
# AXIS: the free parameter that recurs everywhere. A residue key over one of
# four axes. Generalising periodic_completion's row/col axis to the diagonals is
# what unlocked 05269061 for free — the attribute is axis-agnostic.
# ──────────────────────────────────────────────────────────────────────────
AXES = ("row", "col", "diag_down", "diag_up")


def axis_key(axis: str, r: int, c: int) -> int:
    if axis == "row":       return r
    if axis == "col":       return c
    if axis == "diag_down": return r - c      # constant on ↘ diagonals
    if axis == "diag_up":   return r + c      # constant on ↗ (anti) diagonals
    raise ValueError(axis)


# ──────────────────────────────────────────────────────────────────────────
# PARTITION attributes
# ──────────────────────────────────────────────────────────────────────────
def lattice(grid, bar_colour=None):
    """Full-length single-colour rows/cols partition the grid into cells.
    Returns (bar_colour, row_bands, col_bands) or None if no lattice."""
    H, W = _dims(grid)
    bc = bar_colour
    if bc is None:
        for r in range(H):
            if len(set(grid[r])) == 1 and grid[r][0] != BG:
                bc = grid[r][0]; break
        if bc is None:
            for c in range(W):
                col = {grid[r][c] for r in range(H)}
                if len(col) == 1 and next(iter(col)) != BG:
                    bc = next(iter(col)); break
    if bc is None:
        return None
    brow = {r for r in range(H) if all(grid[r][c] == bc for c in range(W))}
    bcol = {c for c in range(W) if all(grid[r][c] == bc for r in range(H))}
    if not brow and not bcol:
        return None

    def bands(n, bars):
        out = []; s = None
        for i in range(n):
            if i in bars:
                if s is not None: out.append((s, i - 1)); s = None
            else:
                if s is None: s = i
        if s is not None: out.append((s, n - 1))
        return out
    return bc, bands(H, brow), bands(W, bcol)


def enclosure(grid, bg=BG):
    """Cells of colour `bg` not reachable from the border (4-connected) = enclosed.
    Returns a set of (r,c)."""
    H, W = _dims(grid)
    seen = [[False] * W for _ in range(H)]
    q = deque()
    for r in range(H):
        for c in (0, W - 1):
            if grid[r][c] == bg and not seen[r][c]:
                seen[r][c] = True; q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if grid[r][c] == bg and not seen[r][c]:
                seen[r][c] = True; q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == bg and not seen[nr][nc]:
                seen[nr][nc] = True; q.append((nr, nc))
    return {(r, c) for r in range(H) for c in range(W)
            if grid[r][c] == bg and not seen[r][c]}


def components(grid, bg=BG, diagonal=True):
    """Connected same-colour components. Returns list of (colour, [cells])."""
    H, W = _dims(grid)
    seen = [[False] * W for _ in range(H)]
    nbrs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if diagonal else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    out = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not seen[r][c]:
                col = grid[r][c]; q = deque([(r, c)]); seen[r][c] = True; cells = [(r, c)]
                while q:
                    a, b = q.popleft()
                    for dr, dc in nbrs:
                        na, nb = a + dr, b + dc
                        if 0 <= na < H and 0 <= nb < W and not seen[na][nb] and grid[na][nb] == col:
                            seen[na][nb] = True; q.append((na, nb)); cells.append((na, nb))
                out.append((col, cells))
    return out


# ──────────────────────────────────────────────────────────────────────────
# AXIS-RESIDUE attribute: fit (axis, period) so colour is a function of the
# residue class, consistent over all GIVEN (non-bg) cells; then fill every cell.
# Generalises periodic_completion to any of the four axes. (05269061)
# ──────────────────────────────────────────────────────────────────────────
def fit_axis_residue(grid, bg=BG, max_period=6):
    H, W = _dims(grid)
    given = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]
    if not given:
        return None
    for axis in AXES:
        for P in range(2, max_period + 1):
            cls = {}; ok = True
            for r, c, v in given:
                k = axis_key(axis, r, c) % P
                if k in cls and cls[k] != v:
                    ok = False; break
                cls[k] = v
            if ok and len(cls) == P:
                return {"axis": axis, "period": P, "classes": cls}
    return None


def apply_axis_residue(grid, params, out_h=None, out_w=None):
    H, W = _dims(grid)
    OH, OW = out_h or H, out_w or W
    axis, P, cls = params["axis"], params["period"], params["classes"]
    return [[cls[axis_key(axis, r, c) % P] for c in range(OW)] for r in range(OH)]


# ──────────────────────────────────────────────────────────────────────────
# AXIS-PAIRS + FILL-BETWEEN on lattice cells: where two cells on the same
# cell-row/col carry the same colour, fill the cells between. (06df4c85)
# ──────────────────────────────────────────────────────────────────────────
def lattice_cell_colours(grid, latt_info):
    bc, rb, cb = latt_info
    cell = {}
    for bi, (r0, r1) in enumerate(rb):
        for bj, (c0, c1) in enumerate(cb):
            cols = [grid[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                    if grid[r][c] not in (BG, bc)]
            cell[(bi, bj)] = cols[0] if cols else None
    return cell


def fill_between_on_lattice(grid, latt_info):
    bc, rb, cb = latt_info
    cell = lattice_cell_colours(grid, latt_info)
    out = [row[:] for row in grid]

    def paint(bi, bj, col):
        r0, r1 = rb[bi]; c0, c1 = cb[bj]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] != bc:
                    out[r][c] = col
    nR, nC = len(rb), len(cb)
    for bi in range(nR):
        byc = defaultdict(list)
        for bj in range(nC):
            if cell[(bi, bj)] is not None:
                byc[cell[(bi, bj)]].append(bj)
        for col, bjs in byc.items():
            if len(bjs) >= 2:
                for bj in range(min(bjs), max(bjs) + 1):
                    paint(bi, bj, col)
    for bj in range(nC):
        byr = defaultdict(list)
        for bi in range(nR):
            if cell[(bi, bj)] is not None:
                byr[cell[(bi, bj)]].append(bi)
        for col, bis in byr.items():
            if len(bis) >= 2:
                for bi in range(min(bis), max(bis) + 1):
                    paint(bi, bj, col)
    return out


# ──────────────────────────────────────────────────────────────────────────
# FILL-REGION: paint a detected region (e.g. enclosure) one fitted colour.
# ──────────────────────────────────────────────────────────────────────────
def fill_region(grid, region, colour):
    out = [row[:] for row in grid]
    for r, c in region:
        out[r][c] = colour
    return out


# ──────────────────────────────────────────────────────────────────────────
# PROJECT-TEMPLATE: the largest component is a template; each single-colour
# marker projects recoloured copies of the template in the marker's direction,
# stepped by the template size, to the grid edge. (045e512c)
# ──────────────────────────────────────────────────────────────────────────
def project_template(grid, bg=BG):
    H, W = _dims(grid)
    cs = sorted(components(grid, bg, diagonal=True), key=lambda x: -len(x[1]))
    if not cs:
        return None
    tcol, tcells = cs[0]
    markers = cs[1:]
    tr0 = min(r for r, c in tcells); tc0 = min(c for r, c in tcells)
    tr1 = max(r for r, c in tcells); tc1 = max(c for r, c in tcells)
    th, tw = tr1 - tr0 + 1, tc1 - tc0 + 1
    shape = [(r - tr0, c - tc0) for r, c in tcells]
    tcr, tcc = (tr0 + tr1) / 2, (tc0 + tc1) / 2
    out = [row[:] for row in grid]
    for mcol, mcells in markers:
        mr = sum(r for r, c in mcells) / len(mcells)
        mc = sum(c for r, c in mcells) / len(mcells)
        dr = (mr > tcr) - (mr < tcr); dc = (mc > tcc) - (mc < tcc)
        if dr == 0 and dc == 0:
            continue
        step_r, step_c = dr * (th + 1), dc * (tw + 1)
        k = 1
        while k <= 60:
            br, bc = tr0 + step_r * k, tc0 + step_c * k
            any_on = False
            for sr, sc in shape:
                rr, cc = br + sr, bc + sc
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr][cc] = mcol; any_on = True
            if not any_on:
                break
            k += 1
    return out


# ──────────────────────────────────────────────────────────────────────────
# TRANSFORM attributes
# ──────────────────────────────────────────────────────────────────────────
def recolour(grid, cmap):
    return [[cmap.get(v, v) for v in row] for row in grid]


# ──────────────────────────────────────────────────────────────────────────
# GEOMETRIC attributes (pure, parameter-free building blocks). These are the
# cheap transforms the composer can chain — exhausting the geometry of what we
# have before any hand-authoring. Each is grid -> grid.
# ──────────────────────────────────────────────────────────────────────────
def flip_h(grid):   return [row[::-1] for row in grid]
def flip_v(grid):   return grid[::-1]
def transpose(grid):
    H, W = _dims(grid)
    return [[grid[r][c] for r in range(H)] for c in range(W)]
def rot90(grid):    return [list(row) for row in zip(*grid[::-1])]
def rot180(grid):   return [row[::-1] for row in grid[::-1]]
def rot270(grid):   return [list(row) for row in zip(*grid)][::-1]


def crop_to_content(grid, bg=BG):
    """Smallest bounding box containing all non-bg cells."""
    H, W = _dims(grid)
    cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    if not cells:
        return grid
    r0 = min(r for r, c in cells); r1 = max(r for r, c in cells)
    c0 = min(c for r, c in cells); c1 = max(c for r, c in cells)
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]


def tile(grid, ny, nx):
    """Tile the grid ny times vertically, nx times horizontally."""
    H, W = _dims(grid)
    return [[grid[r % H][c % W] for c in range(W * nx)] for r in range(H * ny)]


def symmetrize(grid, bg=BG):
    """Fill bg cells using whichever of the four reflective symmetries the
    non-bg cells already (partially) satisfy — the classic 'repair the symmetry'
    attribute. Returns the completed grid (best-effort; may leave bg if no
    symmetry explains a cell)."""
    H, W = _dims(grid)
    out = [row[:] for row in grid]
    maps = [lambda r, c: (r, W - 1 - c),
            lambda r, c: (H - 1 - r, c),
            lambda r, c: (H - 1 - r, W - 1 - c),
            lambda r, c: (c, r) if H == W else None]
    for r in range(H):
        for c in range(W):
            if out[r][c] != bg:
                continue
            for m in maps:
                p = m(r, c)
                if p is None:
                    continue
                rr, cc = p
                if 0 <= rr < H and 0 <= cc < W and grid[rr][cc] != bg:
                    out[r][c] = grid[rr][cc]; break
    return out


# ──────────────────────────────────────────────────────────────────────────
# LATTICE CROSS-FILL (e760a62e): connect same-axis same-colour marker pairs on
# the lattice (as fill_between), AND where a row-fill and a column-fill of
# different colours cross, the crossing cell takes a fitted THIRD colour (the
# cross marker). Generalises fill_between with the cross_project crossing rule,
# lifted to lattice-cell granularity. cross_colour is fitted (out-minus-in), not
# hard-coded; with cross_colour=None and no crossings it reduces to fill_between.
# ──────────────────────────────────────────────────────────────────────────
def lattice_cross_fill(grid, latt_info, cross_colour=None):
    from collections import defaultdict
    bc, rb, cb = latt_info
    cell = lattice_cell_colours(grid, latt_info)
    nR, nC = len(rb), len(cb)
    rowfill = defaultdict(set); colfill = defaultdict(set)
    for bi in range(nR):
        byc = defaultdict(list)
        for bj in range(nC):
            if cell[(bi, bj)] is not None:
                byc[cell[(bi, bj)]].append(bj)
        for col, bjs in byc.items():
            if len(bjs) >= 2:
                for bj in range(min(bjs), max(bjs) + 1):
                    rowfill[(bi, bj)].add(col)
    for bj in range(nC):
        byr = defaultdict(list)
        for bi in range(nR):
            if cell[(bi, bj)] is not None:
                byr[cell[(bi, bj)]].append(bi)
        for col, bis in byr.items():
            if len(bis) >= 2:
                for bi in range(min(bis), max(bis) + 1):
                    colfill[(bi, bj)].add(col)
    out = [row[:] for row in grid]

    def paint(bi, bj, colr):
        r0, r1 = rb[bi]; c0, c1 = cb[bj]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] != bc:
                    out[r][c] = colr
    for bi in range(nR):
        for bj in range(nC):
            rf = rowfill.get((bi, bj), set()); cf = colfill.get((bi, bj), set())
            allc = rf | cf
            if not allc:
                continue
            if rf and cf and len(allc) >= 2 and cross_colour is not None:
                paint(bi, bj, cross_colour)
            else:
                paint(bi, bj, next(iter(allc)))
    return out


# ──────────────────────────────────────────────────────────────────────────
# OBJECT FAMILY (per-object perception + action). Objects = connected non-bg
# components; connectivity (4- vs 8-) is a PARAMETER — 8-connectivity merges a
# diagonal chain into one blob, 4-connectivity keeps the pieces apart, and which
# is right is task-dependent, so callers try both. Grounded in two verified
# solves (90347967 anchor-pivot rotation; 8ee62060 slot reflection).
# ──────────────────────────────────────────────────────────────────────────
def objects(grid, bg=BG, diagonal=True):
    from collections import deque
    H, W = _dims(grid)
    seen = [[False] * W for _ in range(H)]
    nbrs = ([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if diagonal else [(-1, 0), (1, 0), (0, -1), (0, 1)])
    objs = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not seen[r][c]:
                q = deque([(r, c)]); seen[r][c] = True; cells = [(r, c)]
                while q:
                    a, b = q.popleft()
                    for dr, dc in nbrs:
                        na, nb = a + dr, b + dc
                        if 0 <= na < H and 0 <= nb < W and not seen[na][nb] and grid[na][nb] != bg:
                            seen[na][nb] = True; q.append((na, nb)); cells.append((na, nb))
                objs.append(cells)
    return objs


def anchor_pivot_rotate(grid, anchor_colour, bg=BG):
    """Rotate the whole non-bg figure 180 degrees about the single cell of
    `anchor_colour`, which stays fixed. (90347967: grey is the pivot.)"""
    H, W = _dims(grid)
    anch = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == anchor_colour]
    if len(anch) != 1:
        return None
    gr, gc = anch[0]
    out = [[bg] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                nr, nc = 2 * gr - r, 2 * gc - c
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr][nc] = grid[r][c]
                else:
                    return None   # rotation would fall off-grid: rule doesn't hold
    return out


def object_slot_reflect(grid, axis, bg=BG, diagonal=False):
    """Each object keeps its internal pattern upright but its POSITION SLOT is
    reflected across the grid on `axis` ('lr' or 'ud'). (8ee62060.)"""
    H, W = _dims(grid)
    out = [[bg] * W for _ in range(H)]
    for cells in objects(grid, bg, diagonal):
        r0 = min(r for r, c in cells); r1 = max(r for r, c in cells)
        c0 = min(c for r, c in cells); c1 = max(c for r, c in cells)
        for r, c in cells:
            if axis == "lr":
                nr, nc = r, (W - 1 - c1) + (c - c0)
            else:
                nr, nc = (H - 1 - r1) + (r - r0), c
            out[nr][nc] = grid[r][c]
    return out
