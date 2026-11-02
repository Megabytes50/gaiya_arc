"""
auto_compose.py — THE SPARK TEST (Sonia, 2026-10-16).

Question: with ONLY the typed attribute primitives (attributes.py) and NO
hand-written per-family recipe, can a blind search COMPOSE + FIT PARAMS to
re-derive the five solves — and maybe reach tasks we never hand-coded?

If yes → the attribute vocabulary genuinely composes; enriching it is worthwhile.
If no  → the hand recipes were doing the real work; stop and rethink.

MECHANISM
=========
Each attribute is wrapped as a STEP with:
  .name
  .propose(task) -> list of PARAMETERISED ops. Each op is fitted from the TRAIN
     pairs (so params are learned, not hard-coded) and is a pure grid->grid fn
     (or grid->grid for the whole task via closure). An attribute that needs no
     fitting proposes one op; one with choices (axis, period, colour) proposes a
     SMALL set already narrowed by what it detects in the grids.

The searcher composes ops up to MAX_DEPTH, left to right, and accepts a chain the
moment it reproduces EVERY train pair byte-exact (the honesty gate). Ops are
GATED: a step proposes nothing when its attribute isn't present, so the branching
factor stays tiny — the Least-Effort prune, now at attribute granularity.

Terminal ops (produce the whole output: residue-fill, lattice-fill, enclosure-
fill, project) can end a chain alone. Transform ops (recolour, grow) only matter
composed. This is deliberately a SMALL beam — we want a spark, not a DSL.

Pure stdlib.
"""
from __future__ import annotations
from typing import Callable, Optional
import attributes as A

Grid = list
BG = 0
MAX_DEPTH = 2


def _dims(g): return (len(g), len(g[0]) if g else 0)


# ──────────────────────────────────────────────────────────────────────────
# Each proposer returns a list of (op_label, op_fn) where op_fn: Grid -> Grid|None.
# Ops are fitted from the whole task (closures capture fitted params). Proposers
# gate on detectability so the search space stays small.
# ──────────────────────────────────────────────────────────────────────────
def propose_axis_residue(task):
    if A.fit_axis_residue(task.train[0].input) is None:
        return []
    def op(g):
        p = A.fit_axis_residue(g)
        return A.apply_axis_residue(g, p) if p else None
    return [("axis_residue", op)]


def propose_lattice_fill(task):
    if A.lattice(task.train[0].input) is None:
        return []
    def op(g):
        info = A.lattice(g)
        return A.fill_between_on_lattice(g, info) if info else None
    return [("lattice_fill_between", op)]


def propose_lattice_cross_fill(task):
    # lattice present + a single fitted cross colour (appears in every output,
    # in no input) = crossing-marker rule on the lattice. Fits the colour; falls
    # back to plain fill (cross=None) when no such colour, so it subsumes
    # lattice_fill_between on tasks with crossings.
    if A.lattice(task.train[0].input) is None:
        return []
    cross = None
    for p in task.train:
        ic = {v for row in p.input for v in row}
        oc = {v for row in p.output for v in row}
        ex = oc - ic
        if len(ex) == 1:
            c = next(iter(ex))
            if cross is None:
                cross = c
            elif cross != c:
                return []
        elif len(ex) > 1:
            return []
    if cross is None:
        return []   # no crossing colour: plain lattice_fill_between already covers it
    def op(g):
        info = A.lattice(g)
        return A.lattice_cross_fill(g, info, cross) if info else None
    return [("lattice_cross_fill", op)]


def propose_enclosure_fill(task):
    if any(_dims(p.input) != _dims(p.output) for p in task.train):
        return []
    if not any(A.enclosure(p.input) for p in task.train):
        return []
    fill = None
    for p in task.train:
        for (r, c) in A.enclosure(p.input):
            v = p.output[r][c]
            if v != BG:
                if fill is None: fill = v
                elif fill != v: return []
    if fill is None:
        return []
    def op(g):
        return A.fill_region(g, A.enclosure(g), fill)
    return [("enclosure_fill", op)]


def propose_project(task):
    if len(A.components(task.train[0].input)) < 2:
        return []
    def op(g):
        return A.project_template(g)
    return [("project_template", op)]


def propose_grow_periodic(task):
    # only when output size differs; fit size rule + vertical period
    p0 = task.train[0]
    if _dims(p0.input) == _dims(p0.output):
        return []
    consts = set(); facs = set()
    for p in task.train:
        a, b = _dims(p.input); c, d = _dims(p.output)
        consts.add((c, d))
        if a and b and c % a == 0 and d % b == 0:
            facs.add((c // a, d // b))
    def period_rows(rows):
        n = len(rows)
        for P in range(1, n + 1):
            if all(rows[r] == rows[r % P] for r in range(n)):
                return P
        return n
    def op(g):
        ih, iw = _dims(g)
        if len(consts) == 1: oh, ow = next(iter(consts))
        elif len(facs) == 1: fr, fc = next(iter(facs)); oh, ow = ih*fr, iw*fc
        else: return None
        P = period_rows(g)
        return [[g[r % P][c % iw] for c in range(ow)] for r in range(oh)]
    return [("grow_periodic", op)]


def propose_recolour(task):
    # a transform step: fit ONE consistent colour map across train, proposed as a
    # composable op. Two fit modes:
    #   (a) CELLWISE (same-shape pairs): map each input cell to its output cell.
    #   (b) PALETTE (size-change pairs): map the input's single non-bg colour to
    #       the output's single non-bg colour — so recolour can compose AFTER a
    #       grow step that changes size. This is what lets grow ∘ recolour form.
    same_shape = all(_dims(p.input) == _dims(p.output) for p in task.train)
    cmap = {}
    if same_shape:
        for p in task.train:
            for ri, ro in zip(p.input, p.output):
                for a, b in zip(ri, ro):
                    if a in cmap and cmap[a] != b: return []
                    cmap[a] = b
    else:
        for p in task.train:
            incols = {v for row in p.input for v in row if v != BG}
            outcols = {v for row in p.output for v in row if v != BG}
            if len(incols) != 1 or len(outcols) != 1:
                return []
            a, b = next(iter(incols)), next(iter(outcols))
            if a in cmap and cmap[a] != b: return []
            cmap[a] = b
    if not cmap or all(k == v for k, v in cmap.items()):
        return []
    # ANTI-OVERFIT GATE: a real palette-swap remaps colours that RECUR across the
    # train inputs. If the non-identity source colours differ from pair to pair
    # (e.g. 2->4 in pair 0, 3->6 in pair 1), the "map" is a per-pair relabel of a
    # RELATIONAL rule (key-cell recolours shape) we have NOT named — it will not
    # cover the test's new colours. Refuse unless every remapped source colour is
    # present in EVERY train input. (This is the aabf363d lesson: train-fit
    # necessary, not sufficient — abstain rather than ship a false positive.)
    remapped = {k for k, v in cmap.items() if k != v and k != BG}
    for p in task.train:
        incols = {v for row in p.input for v in row}
        if not remapped <= incols:
            return []
    def op(g):
        return A.recolour(g, cmap)
    return [("recolour", op)]


# ── GEOMETRIC proposers: parameter-free transforms, always available as chain
# links (the composer decides if they help). Kept cheap — they only matter in
# composition, so they add almost nothing to the branching factor at depth 1.
def propose_geometric(task):
    ops = [
        ("flip_h", A.flip_h), ("flip_v", A.flip_v), ("transpose", A.transpose),
        ("rot90", A.rot90), ("rot180", A.rot180), ("rot270", A.rot270),
    ]
    return ops


def propose_crop(task):
    # only when at least one output is strictly smaller than its input
    if not any(_dims(p.output) < _dims(p.input) for p in task.train):
        return []
    return [("crop_to_content", lambda g: A.crop_to_content(g))]


def propose_symmetrize(task):
    # only same-shape with bg present to fill
    if any(_dims(p.input) != _dims(p.output) for p in task.train):
        return []
    return [("symmetrize", lambda g: A.symmetrize(g))]


def propose_tile(task):
    # fit an integer tiling factor from train output/input dim ratios
    facs = set()
    for p in task.train:
        a, b = _dims(p.input); c, d = _dims(p.output)
        if a and b and c % a == 0 and d % b == 0 and (c // a, d // b) != (1, 1):
            facs.add((c // a, d // b))
    if len(facs) != 1:
        return []
    ny, nx = next(iter(facs))
    return [(f"tile({ny}x{nx})", lambda g: A.tile(g, ny, nx))]


def propose_anchor_pivot(task):
    # same-shape; fit the anchor colour = a colour that occurs exactly once in
    # every train input and whose cell is unchanged in the output.
    if any(_dims(p.input) != _dims(p.output) for p in task.train):
        return []
    from collections import Counter
    cand = None
    p0 = task.train[0]
    cnt = Counter(v for row in p0.input for v in row)
    singles = [v for v, k in cnt.items() if k == 1 and v != BG]
    ops = []
    for anchor in singles:
        def op(g, a=anchor):
            return A.anchor_pivot_rotate(g, a)
        ops.append((f"anchor_pivot({anchor})", op))
    return ops


def propose_object_slot_reflect(task):
    if any(_dims(p.input) != _dims(p.output) for p in task.train):
        return []
    ops = []
    for axis in ("lr", "ud"):
        for diag in (False, True):
            def op(g, ax=axis, dg=diag):
                return A.object_slot_reflect(g, ax, diagonal=dg)
            tag = f"object_slot_reflect({axis}{'/8' if diag else ''})"
            ops.append((tag, op))
    return ops


PROPOSERS = [
    propose_axis_residue,
    propose_lattice_cross_fill,
    propose_lattice_fill,
    propose_anchor_pivot,
    propose_object_slot_reflect,
    propose_enclosure_fill,
    propose_project,
    propose_grow_periodic,
    propose_recolour,
    propose_crop,
    propose_symmetrize,
    propose_tile,
    propose_geometric,
]


def _reproduces(chain, task) -> bool:
    for p in task.train:
        g = p.input
        for _, op in chain:
            g = op(g)
            if g is None:
                return False
        if g != p.output:
            return False
    return True


def _run(chain, grid):
    g = grid
    for _, op in chain:
        g = op(g)
        if g is None:
            return None
    return g


def auto_solve(task, max_depth=MAX_DEPTH):
    """Blind compositional search. Returns (output_for_test, label, considered)
    where label is the composed attribute chain, or (None, None, considered)."""
    # propose all ops once (params fitted from train); gate keeps this small
    ops = []
    for prop in PROPOSERS:
        ops.extend(prop(task))
    considered = 0
    # depth-1 then depth-2 (least effort first)
    for depth in range(1, max_depth + 1):
        from itertools import product
        for chain in product(ops, repeat=depth):
            considered += 1
            if _reproduces(list(chain), task):
                out = _run(list(chain), task.test[0].input)
                label = " ∘ ".join(name for name, _ in chain)
                return out, label, considered
    return None, None, considered
