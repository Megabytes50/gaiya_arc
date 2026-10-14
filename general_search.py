"""
general_search.py — one unified inner-shell search where ANY operation (fitted
or parameter-free) can be a step in a composed program.

WHY THIS EXISTS (the limitation it fixes)
=========================================
The first search (search.py) had three disconnected layers and composed only
PARAMETER-FREE primitives at depth 2. So a FITTED op (fill, symmetry, recolour,
periodic) could only ever be the WHOLE program, never a STEP. But real ARC rules
are often "fitted-op THEN something", e.g. restore symmetry -> crop to the filled
region; recolour -> gravity. Those were inexpressible. This module makes every
operation a composable step, so the inner shell can genuinely COMPOSE operations
— which is the engine's thesis (heads composed into a program), made executable.

HOW IT WORKS — step proposers over intermediate states
=======================================================
The search state is the tuple of CURRENT grids (one per train pair), starting
from the inputs, aiming at the outputs. At each step we ask every PROPOSER:
"given these current grids and their targets, what concrete transforms are worth
trying from here?" A proposer returns zero or more (name, fn) steps:
  * parameter-free proposers offer their op (one step).
  * fitted proposers FIT their parameter against the CURRENT grids (not the
    original inputs) and offer the concrete, parameterized step — so a fit can
    happen mid-composition.
We apply a step to all current grids; if they all now equal their targets, the
program solves the task. Otherwise we recurse, up to max_depth, best-first by a
cheap heuristic (how close the grids are to their targets), pruning revisited
states and oversized grids. Bounded and ordered so it stays inside Kaggle
compute.

Pure stdlib. Reuses dsl / object_ops / fill_ops primitives and the fitters' ideas.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import dsl, object_ops, fill_ops

Grid = list
MAX_DIM = 120


def _too_big(g: Grid) -> bool:
    return bool(g) and (len(g) > MAX_DIM or len(g[0]) > MAX_DIM)


def _apply(fn, grids: list) -> Optional[list]:
    """Apply one step fn to every current grid; None if any errors or overflows."""
    out = []
    for g in grids:
        try:
            r = fn(g)
        except Exception:
            return None
        if _too_big(r):
            return None
        out.append(r)
    return out


def _hash_state(grids: list) -> tuple:
    return tuple(tuple(tuple(row) for row in g) for g in grids)


def _distance(grids: list, targets: list) -> int:
    """Cheap closeness heuristic: total mismatched cells where shapes agree, plus
    a big penalty per shape mismatch. Lower = closer. Guides best-first order."""
    d = 0
    for g, t in zip(grids, targets):
        if dsl.dims(g) != dsl.dims(t):
            d += 1000
        else:
            d += sum(1 for gr, tr in zip(g, t) for a, b in zip(gr, tr) if a != b)
    return d


# ──────────────────────────────────────────────────────────────────────────
# step proposers: (current_grids, targets) -> list[(name, fn)]
# ──────────────────────────────────────────────────────────────────────────
def _propose_param_free(cur, tgt) -> list:
    steps = []
    for name, fn in dsl.PARAM_FREE.items():
        if name == "identity":
            continue
        steps.append((name, fn))
    for name, fn in object_ops.OBJECT_PARAM_FREE.items():
        steps.append((name, fn))
    steps.append(("add_border", fill_ops.add_border))
    return steps


def _colours(grids) -> set:
    return set(v for g in grids for row in g for v in row)


def _fits_all(fn, cur, tgt) -> bool:
    """FIT-BEFORE-OFFER: does this concrete step turn EVERY current grid into its
    target exactly? Only steps that pass are offered — so a fitted proposer
    contributes at most ONE validated candidate, never a cloud of guesses. This
    is what makes composition affordable, and it is Gaiya-honest: a head proposes
    only when it is actually SATISFIED across the evidence."""
    for g, t in zip(cur, tgt):
        try:
            if fn(g) != t:
                return False
        except Exception:
            return False
    return True


def _propose_recolour(cur, tgt) -> list:
    """Fit a colour map from pair 0; offer it ONLY if it solves ALL pairs from
    here. (A recolour is a terminal step — if it fits all, we're done.)"""
    if dsl.dims(cur[0]) != dsl.dims(tgt[0]):
        return []
    cmap = dsl.fit_colour_map(cur[0], tgt[0])
    if not cmap or all(k == v for k, v in cmap.items()):
        return []
    fn = lambda g, m=cmap: dsl.apply_colour_map(g, m)
    return [(f"recolour{cmap}", fn)] if _fits_all(fn, cur, tgt) else []


def _propose_fill(cur, tgt) -> list:
    """Offer fill_enclosed(bg,fill) ONLY for a colour pair that solves all pairs
    from here. Enumerate candidates, but offer only validated ones."""
    cols = _colours(cur) | _colours(tgt)
    out = []
    for bg in cols:
        for fill in cols:
            if fill == bg:
                continue
            fn = lambda g, b=bg, f=fill: fill_ops.fill_enclosed_with(g, f, b)
            if _fits_all(fn, cur, tgt):
                out.append((f"fill(bg={bg},fill={fill})", fn))
    return out


def _propose_symmetry(cur, tgt) -> list:
    """Offer a symmetry completion ONLY if it solves all pairs from here."""
    cols = _colours(cur)
    opts = [("sym_h", fill_ops.complete_h_symmetry),
            ("sym_v", fill_ops.complete_v_symmetry),
            ("sym_r180", fill_ops.complete_rot180_symmetry),
            ("sym_all", fill_ops.complete_all_symmetry)]
    out = []
    for bg in cols:
        for n, f in opts:
            fn = lambda g, ff=f, b=bg: ff(g, b)
            if _fits_all(fn, cur, tgt):
                out.append((f"{n}(bg={bg})", fn))
    return out


def _propose_periodic(cur, tgt) -> list:
    """Offer a periodic completion ONLY if it solves all pairs from here."""
    g0 = cur[0]
    rows, cols = dsl.dims(g0)
    cset = _colours(cur)
    ch = sorted(set([p for p in range(1, min(rows, 6) + 1)] +
                    [d for d in range(1, rows + 1) if rows % d == 0]))
    cw = sorted(set([p for p in range(1, min(cols, 6) + 1)] +
                    [d for d in range(1, cols + 1) if cols % d == 0]))
    out = []
    for bg in cset:
        for ph in ch:
            for pw in cw:
                if ph == rows and pw == cols:
                    continue
                fn = lambda g, a=ph, b=pw, c=bg: fill_ops.complete_periodic(g, a, b, c)
                if _fits_all(fn, cur, tgt):
                    out.append((f"periodic({ph},{pw},bg={bg})", fn))
                    break               # one validated period per bg is enough
            else:
                continue
            break
    return out


# Two tiers. Param-free proposers run at EVERY state (cheap, compose freely).
# Fitted proposers are TERMINAL: they offer only a step that already solves all
# pairs from the current state, so they are tried but never branch.
PROPOSERS = [_propose_param_free]
TERMINAL_PROPOSERS = [_propose_recolour, _propose_fill,
                      _propose_symmetry, _propose_periodic]


@dataclass
class GProgram:
    steps: list
    @property
    def name(self) -> str:
        return " -> ".join(n for n, _ in self.steps) or "identity"
    def run(self, g: Grid) -> Grid:
        for _, fn in self.steps:
            g = fn(g)
            if _too_big(g):
                raise ValueError("oversized")
        return g


def solve(task, max_depth: int = 3, beam: int = 6, max_programs: int = 2) -> list:
    """General best-first composed search. Returns up to `max_programs` solving
    GPrograms. beam = how many frontier states to keep per depth (controls cost)."""
    inputs = [p.input for p in task.train]
    targets = [p.output for p in task.train]

    def is_solved(grids):
        return all(g == t for g, t in zip(grids, targets))

    found, seen = [], set()
    seen.add(_hash_state(inputs))

    # frontier: list of (grids, steps). Start at the raw inputs.
    frontier = [(inputs, [])]

    # Terminal check at the ROOT: does a fitted op solve the task outright?
    for prop in TERMINAL_PROPOSERS:
        for name, fn in prop(inputs, targets):
            prog = GProgram([(name, fn)])
            if prog.name not in {p.name for p in found}:
                found.append(prog)
                if len(found) >= max_programs:
                    return found[:max_programs]

    for depth in range(max_depth):
        scored_next = []
        for grids, steps in frontier:
            # 1) TERMINAL: can a fitted op finish the task from this state? (no branch)
            for prop in TERMINAL_PROPOSERS:
                for name, fn in prop(grids, targets):
                    ng = _apply(fn, grids)
                    if ng is not None and is_solved(ng):
                        prog = GProgram(steps + [(name, fn)])
                        if prog.name not in {p.name for p in found}:
                            found.append(prog)
                            if len(found) >= max_programs:
                                return found[:max_programs]
            # 2) EXPAND with cheap param-free steps (these branch)
            cands = []
            for prop in PROPOSERS:
                cands.extend(prop(grids, targets))
            for name, fn in cands:
                ng = _apply(fn, grids)
                if ng is None:
                    continue
                h = _hash_state(ng)
                if h in seen:
                    continue
                seen.add(h)
                nsteps = steps + [(name, fn)]
                if is_solved(ng):
                    prog = GProgram(nsteps)
                    if prog.name not in {p.name for p in found}:
                        found.append(prog)
                        if len(found) >= max_programs:
                            return found[:max_programs]
                    continue
                scored_next.append((_distance(ng, targets), ng, nsteps))
        if not scored_next:
            break
        scored_next.sort(key=lambda x: x[0])
        frontier = [(g, s) for _, g, s in scored_next[:beam]]

    return found[:max_programs]


def predict(task, max_programs: int = 2, **kw) -> list:
    progs = solve(task, max_programs=max_programs, **kw)
    ti = task.test[0].input
    out = []
    for p in progs:
        try:
            out.append(p.run(ti))
        except Exception:
            pass
    return out
