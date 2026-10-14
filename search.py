"""
search.py — the inner-shell search: find a PROGRAM (a sequence of DSL primitives)
that turns every train input into its exact output, then apply it to the test.

THE IDEA, PLAINLY
=================
A task hands us a few (input -> output) train pairs and asks for the hidden rule.
A "program" here is a short pipeline of primitives, e.g. [flip_h, recolour{1:2}].
A program SOLVES the task iff, run on every train input, it reproduces that
pair's output EXACTLY (ARC scores exact match only). The search's whole job is to
find such programs cheaply, and to return TWO of them (pass@2: ARC lets us submit
two guesses per test input).

WHY THE LAYERING (cheapest first — this is what keeps it inside Kaggle compute)
==============================================================================
  Layer 0  DEPTH-1 param-free : try each parameter-free primitive alone. ~10 tries.
  Layer 1  FITTED             : primitives whose parameter is LEARNED from the
                                pairs (recolour's colour map; tile/scale's factor
                                from the input->output size ratio). We never guess
                                a parameter — we read it off pair 1 and verify it
                                holds on all pairs. Still cheap.
  Layer 2  DEPTH-2            : compose two primitives. Expensive (NxN), so it is
                                ENTERED ONLY when layers 0-1 found nothing, and
                                pruned (skip identity-first, skip no-op chains).

THE GAIYA WRAPPER (the hybrid — the paper's thesis)
===================================================
Each candidate is a HEAD proposing a hypothesis; a program that fits all train
pairs is SATISFIED, one that fits none is UNMET. Layers 0-1 are the fast outer
reflex; entering layer 2 is the SEAM escalating an UNMET task to deeper inner-
shell deliberation. A program that solves a task can be CRYSTALLIZED by AIDA into
a reusable head (future-you: that is the self-extension story). This module is
the kernel that machinery wraps; it is kept standalone and testable so the engine
story is provable, not just asserted.

Pure stdlib. A "program" is a list of (name, fn) steps; applying it is folding the
grid through each fn. Every step returns a fresh grid (never mutates input).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import dsl
import object_ops
import fill_ops

Grid = list

# ARC outputs never exceed 30x30. We allow a little headroom for legitimate
# intermediate grids (a 30x30 tiled 3x = 90), then hard-stop: anything past this
# is a runaway transform, not a real ARC rule. Keeps the search memory-safe.
MAX_DIM = 120


class _TooBig(Exception):
    """Raised inside Program.run when an intermediate grid blows past MAX_DIM.
    Caught by solves()/predict() as an ordinary failure — the program is simply
    not the rule."""
    pass


# ──────────────────────────────────────────────────────────────────────────
# a program: an ordered pipeline of named steps, grid -> grid
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Program:
    """A found rule: named steps applied in order. `name` is human-readable for
    the trace and the paper ('flip_h -> recolour'); `run` applies it to a grid."""
    steps: list                     # list[(step_name, fn: grid->grid)]

    @property
    def name(self) -> str:
        return " -> ".join(n for n, _ in self.steps)

    def run(self, g: Grid) -> Grid:
        for _, fn in self.steps:
            g = fn(g)
            # SIZE GUARD: ARC grids are at most 30x30. A step that produces
            # something vastly larger (e.g. scale_up stacked on scale_up, or
            # self_fractal on a big grid) is wrong by definition AND a memory
            # bomb. Abandon the program the instant it exceeds a sane cap, so a
            # runaway transform can never allocate gigabytes (this is also what
            # keeps the solver inside Kaggle's memory limit).
            if g and (len(g) > MAX_DIM or len(g[0]) > MAX_DIM):
                raise _TooBig()
        return g

    def solves(self, task) -> bool:
        """True iff this program reproduces EVERY train pair's output exactly."""
        for p in task.train:
            try:
                if self.run(p.input) != p.output:
                    return False
            except Exception:
                return False          # a step that errors cannot be the rule
        return True


# ──────────────────────────────────────────────────────────────────────────
# fitted primitives — parameter LEARNED from the pairs, not guessed
# ──────────────────────────────────────────────────────────────────────────
def _fit_recolour(task) -> Optional[Program]:
    """If a single consistent colour map explains all pairs, that is the rule."""
    cmap = dsl.fit_colour_map(task.train[0].input, task.train[0].output)
    if cmap is None:
        return None
    prog = Program([(f"recolour{cmap}", lambda g, m=cmap: dsl.apply_colour_map(g, m))])
    return prog if prog.solves(task) else None


def _fit_scale(task) -> Optional[Program]:
    """If every output is the input scaled up by the same integer (fr, fc), scale
    is the rule. Factor is READ from the size ratio of pair 0, then verified."""
    i, o = task.train[0].input, task.train[0].output
    ir, ic = dsl.dims(i); orr, oc = dsl.dims(o)
    if ir == 0 or ic == 0 or orr % ir or oc % ic:
        return None
    fr, fc = orr // ir, oc // ic
    if (fr, fc) == (1, 1):
        return None
    prog = Program([(f"scale_up({fr},{fc})", lambda g, a=fr, b=fc: dsl.scale_up(g, a, b))])
    return prog if prog.solves(task) else None


def _fit_tile(task) -> Optional[Program]:
    """If every output is the input repeated (tr, tc) times, tiling is the rule."""
    i, o = task.train[0].input, task.train[0].output
    ir, ic = dsl.dims(i); orr, oc = dsl.dims(o)
    if ir == 0 or ic == 0 or orr % ir or oc % ic:
        return None
    tr, tc = orr // ir, oc // ic
    if (tr, tc) == (1, 1):
        return None
    prog = Program([(f"tile({tr},{tc})", lambda g, a=tr, b=tc: dsl.tile(g, a, b))])
    return prog if prog.solves(task) else None


def _fit_colour_swap(task) -> Optional[Program]:
    """If every output is the input with exactly two colours exchanged, swap is
    the rule. The pair (a,b) is READ from where input and output differ on pair 0,
    then verified on all pairs."""
    i, o = task.train[0].input, task.train[0].output
    if dsl.dims(i) != dsl.dims(o):
        return None
    diffs = set()
    for ra, rb in zip(i, o):
        for a, b in zip(ra, rb):
            if a != b:
                diffs.add((a, b))
    # a clean swap shows exactly two directed differences: a->b and b->a
    if len(diffs) != 2:
        return None
    (a1, b1), (a2, b2) = list(diffs)
    if not (a1 == b2 and b1 == a2):
        return None
    prog = Program([(f"swap({a1},{b1})",
                     lambda g, x=a1, y=b1: fill_ops.swap_two_colours(g, x, y))])
    return prog if prog.solves(task) else None


def _fit_fill_enclosed(task) -> Optional[Program]:
    """Fill enclosed regions, with the fill colour and background LEARNED: try the
    plausible (background, fill) colour pairs drawn from pair 0, keep one that
    reproduces all pairs. No hard-coded colour guess."""
    i, o = task.train[0].input, task.train[0].output
    if dsl.dims(i) != dsl.dims(o):
        return None
    colours = set(v for row in i for v in row) | set(v for row in o for v in row)
    for bg in colours:
        for fill in colours:
            if fill == bg:
                continue
            prog = Program([(f"fill_enclosed(bg={bg},fill={fill})",
                             lambda g, b=bg, f=fill: fill_ops.fill_enclosed_with(g, f, b))])
            if prog.solves(task):
                return prog
    return None


def _fit_symmetry_complete(task) -> Optional[Program]:
    """Restore symmetry by filling holes from a mirror — but LEARN which symmetry
    (H, V, rot180, or all) reproduces the pairs, and the background. Tries each."""
    i, o = task.train[0].input, task.train[0].output
    if dsl.dims(i) != dsl.dims(o):
        return None
    colours = set(v for row in i for v in row)
    options = [
        ("complete_h", fill_ops.complete_h_symmetry),
        ("complete_v", fill_ops.complete_v_symmetry),
        ("complete_rot180", fill_ops.complete_rot180_symmetry),
        ("complete_all", fill_ops.complete_all_symmetry),
    ]
    for bg in colours:
        for name, fn in options:
            prog = Program([(f"{name}(bg={bg})", lambda g, f=fn, b=bg: f(g, b))])
            if prog.solves(task):
                return prog
    return None


def _fit_periodic(task) -> Optional[Program]:
    """Complete a periodic pattern — LEARN the period (ph, pw) and background from
    the pairs. Tries small periods (ARC patterns are small) and the divisors of
    the grid size, keeps one that reproduces all pairs."""
    i, o = task.train[0].input, task.train[0].output
    if dsl.dims(i) != dsl.dims(o):
        return None
    rows, cols = dsl.dims(i)
    colours = set(v for row in i for v in row)
    # candidate periods: every small value and every divisor of the dimension
    cand_h = sorted(set([p for p in range(1, min(rows, 10) + 1)] +
                        [d for d in range(1, rows + 1) if rows % d == 0]))
    cand_w = sorted(set([p for p in range(1, min(cols, 10) + 1)] +
                        [d for d in range(1, cols + 1) if cols % d == 0]))
    for bg in colours:
        for ph in cand_h:
            for pw in cand_w:
                if ph == rows and pw == cols:
                    continue                 # whole grid = no real period
                prog = Program([(f"periodic({ph},{pw},bg={bg})",
                                 lambda g, a=ph, b=pw, c=bg: fill_ops.complete_periodic(g, a, b, c))])
                if prog.solves(task):
                    return prog
    return None


FITTERS = [_fit_recolour, _fit_scale, _fit_tile, _fit_colour_swap,
           _fit_fill_enclosed, _fit_symmetry_complete, _fit_periodic]


# ──────────────────────────────────────────────────────────────────────────
# the search
# ──────────────────────────────────────────────────────────────────────────
# the full parameter-free primitive pool: grid-level + object-level.
# Fill/symmetry/periodic are now FITTED (in FITTERS), not parameter-free — a
# parameter-free fill can only match tasks whose parameter happens to match its
# bake, which is why it added nothing. See INSIGHTS.md (rigid-vs-fitted lesson).
ALL_PARAM_FREE: dict = {**dsl.PARAM_FREE,
                        **object_ops.OBJECT_PARAM_FREE}


def _depth1_programs() -> list:
    """One Program per parameter-free primitive (grid-level AND object-level)."""
    return [Program([(name, fn)]) for name, fn in ALL_PARAM_FREE.items()]


def solve(task, max_programs: int = 2, allow_depth2: bool = True) -> list:
    """Return up to `max_programs` DISTINCT programs that solve the task's train
    pairs, cheapest layer first. Empty list = the search found nothing.

    The returned programs are the pass@2 candidates: each, applied to the test
    input, is one of our two guesses.
    """
    found: list = []
    seen_names: set = set()

    def consider(prog: Optional[Program]):
        if prog is None:
            return
        if prog.name in seen_names:
            return
        if prog.solves(task):
            found.append(prog)
            seen_names.add(prog.name)

    # ── Layer 0: depth-1 parameter-free ──────────────────────────────────
    for prog in _depth1_programs():
        consider(prog)
        if len(found) >= max_programs:
            return found[:max_programs]

    # ── Layer 1: fitted primitives (learned parameters) ──────────────────
    for fitter in FITTERS:
        consider(fitter(task))
        if len(found) >= max_programs:
            return found[:max_programs]

    # ── Layer 2: depth-2 composition (SEAM escalation; pruned) ───────────
    if allow_depth2 and len(found) < max_programs:
        pf = list(ALL_PARAM_FREE.items())
        for n1, f1 in pf:
            if n1 == "identity":
                continue                      # prune: identity-first is wasteful
            for n2, f2 in pf:
                if n2 == "identity":
                    continue
                prog = Program([(n1, f1), (n2, f2)])
                consider(prog)
                if len(found) >= max_programs:
                    return found[:max_programs]

    return found[:max_programs]


def solve_instrumented(task, max_programs: int = 2, allow_depth2: bool = True):
    """Like solve(), but also counts STATES EXPLORED — the number of candidate
    programs actually tested against the train pairs. This is the honest compute
    proxy for the Least-Effort experiment: a DSL searcher pays this cost in full
    on every task; Gaiya pays it only on the residual it hasn't already learned.
    Returns (programs, states_explored)."""
    found: list = []
    seen_names: set = set()
    states = {"n": 0}

    def consider(prog: Optional[Program]):
        if prog is None:
            return
        if prog.name in seen_names:
            return
        states["n"] += 1                       # one state = one program tested
        if prog.solves(task):
            found.append(prog)
            seen_names.add(prog.name)

    for prog in _depth1_programs():
        consider(prog)
        if len(found) >= max_programs:
            return found[:max_programs], states["n"]
    for fitter in FITTERS:
        consider(fitter(task))
        if len(found) >= max_programs:
            return found[:max_programs], states["n"]
    if allow_depth2 and len(found) < max_programs:
        pf = list(ALL_PARAM_FREE.items())
        for n1, f1 in pf:
            if n1 == "identity":
                continue
            for n2, f2 in pf:
                if n2 == "identity":
                    continue
                consider(Program([(n1, f1), (n2, f2)]))
                if len(found) >= max_programs:
                    return found[:max_programs], states["n"]
    return found[:max_programs], states["n"]


def predict(task, max_programs: int = 2) -> list:
    """The pass@2 output: up to two predicted output grids for the FIRST test
    input, one per solving program. If the search found nothing, returns []."""
    progs = solve(task, max_programs=max_programs)
    test_in = task.test[0].input
    out = []
    for prog in progs:
        try:
            out.append(prog.run(test_in))
        except Exception:
            pass
    return out
