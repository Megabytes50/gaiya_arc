"""
invariants.py — the CORRECTED architecture (Sonia's rebuild, 2026-10-12).

THE CORRECTION
==============
The earlier build put puzzle-SOLVING in the heads (if structure X, apply program
P). That is backwards and un-Gaiya. The right architecture:

  * HEADS detect INVARIANTS, they do NOT solve. An invariant = a RELATION THAT
    HOLDS ACROSS ALL TRAIN PAIRS (object count conserved, colour X always -> Y,
    output is the symmetric completion, output is the largest object's box...).
    A head reports SATISFIED (its invariant holds on every pair), NOT_APPLICABLE
    (nothing to check), or UNMET (holds on some pairs, not all). Heads accumulate
    invariants — and invariants RECUR across hundreds of ARC tasks even though
    solution programs never recur. Memorising invariants is what humans do; it is
    NOT the answer-memorisation ARC punishes.

  * The GESTALT is the SET of invariants found to hold for this task.

  * The TERMINUS solves: the held-invariant SET JOINTLY SELECTS a CONSTRUCTION
    and PARAMETERISES it; the terminus executes that to build the output grid.
    The invariants are the REASON; the construction is the mechanism; both are
    inspectable (traceability).

  * AIDA accumulates which invariant-sets select which constructions — ways of
    seeing, not answer-programs. These recur, so accumulation is real here.

This file is the thin END-TO-END vertical slice: a few real invariant-heads +
the constructions their sets select + the terminus, measured on AGI-1, to prove
the corrected architecture solves before scaling the invariant library.

Pure stdlib. Built on perception.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from collections import Counter
import perception as P

Grid = list


def _dims(g): return (len(g), len(g[0]) if g else 0)
def _palette(g): return Counter(v for row in g for v in row)
def _bg(g):
    return _palette(g).most_common(1)[0][0]


# ──────────────────────────────────────────────────────────────────────────
# INVARIANT DETECTION — each checks a relation ACROSS ALL TRAIN PAIRS
# ──────────────────────────────────────────────────────────────────────────
# A detector returns: None (NOT_APPLICABLE / doesn't hold) or a dict of
# PARAMETERS the invariant carries (e.g. the colour map, the axis). Holding =
# the same parameterisation is consistent across every train pair.

def inv_same_shape(train) -> Optional[dict]:
    """INVARIANT: output has the same dimensions as input, every pair."""
    if all(_dims(p.input) == _dims(p.output) for p in train):
        return {}
    return None


def inv_colour_map(train) -> Optional[dict]:
    """INVARIANT: a single consistent colour->colour map explains every pair
    (same shape required). Carries the map as its parameter."""
    cmap = {}
    for p in train:
        if _dims(p.input) != _dims(p.output):
            return None
        for ri, ro in zip(p.input, p.output):
            for a, b in zip(ri, ro):
                if a in cmap:
                    if cmap[a] != b:
                        return None
                else:
                    cmap[a] = b
    if all(k == v for k, v in cmap.items()):
        return None                       # identity isn't an interesting invariant
    return {"cmap": cmap}


def inv_symmetry_completion(train) -> Optional[dict]:
    """INVARIANT: the output is the input with a broken symmetry restored (holes
    filled from the mirror), same shape, same symmetry axis across all pairs."""
    import fill_ops as F
    axes = [("h", F.complete_h_symmetry), ("v", F.complete_v_symmetry),
            ("r180", F.complete_rot180_symmetry), ("all", F.complete_all_symmetry)]
    for name, fn in axes:
        ok = True
        for p in train:
            if _dims(p.input) != _dims(p.output):
                ok = False; break
            if fn(p.input, _bg(p.input)) != p.output:
                ok = False; break
        if ok:
            return {"axis": name}
    return None


def inv_largest_object_box(train) -> Optional[dict]:
    """INVARIANT: the output is the tight bounding-box crop of the largest
    non-background object, every pair."""
    import object_ops as O
    for conn in (8, 4):
        if all(O.crop_to_largest_object(p.input, _bg(p.input), conn) == p.output
               for p in train):
            return {"connectivity": conn}
    return None


def inv_scale(train) -> Optional[dict]:
    """INVARIANT: output is the input scaled up by a constant (fr, fc), every
    pair (same factor across pairs)."""
    import dsl
    i0, o0 = train[0].input, train[0].output
    ir, ic = _dims(i0); orr, oc = _dims(o0)
    if ir == 0 or ic == 0 or orr % ir or oc % ic:
        return None
    fr, fc = orr // ir, oc // ic
    if (fr, fc) == (1, 1):
        return None
    if all(dsl.scale_up(p.input, fr, fc) == p.output for p in train):
        return {"fr": fr, "fc": fc}
    return None


# the invariant-head registry: name -> detector
INVARIANT_HEADS = {
    "same_shape": inv_same_shape,
    "colour_map": inv_colour_map,
    "symmetry_completion": inv_symmetry_completion,
    "largest_object_box": inv_largest_object_box,
    "scale": inv_scale,
}


def detect_invariants(task) -> dict:
    """Run every invariant-head over the task's train pairs. Returns
    {invariant_name: parameters} for every invariant that HOLDS. This set is the
    gestalt — the structured reading the terminus solves from."""
    held = {}
    for name, detector in INVARIANT_HEADS.items():
        params = detector(task.train)
        if params is not None:
            held[name] = params
    return held


# ──────────────────────────────────────────────────────────────────────────
# CONSTRUCTIONS — how an invariant-set builds the output from a test input
# ──────────────────────────────────────────────────────────────────────────
# Each construction takes (test_input, held_invariants) -> output grid.

def _c_colour_map(g, held):
    import dsl
    return dsl.apply_colour_map(g, held["colour_map"]["cmap"])

def _c_symmetry(g, held):
    import fill_ops as F
    axis = held["symmetry_completion"]["axis"]
    fn = {"h": F.complete_h_symmetry, "v": F.complete_v_symmetry,
          "r180": F.complete_rot180_symmetry, "all": F.complete_all_symmetry}[axis]
    return fn(g, _bg(g))

def _c_largest_box(g, held):
    import object_ops as O
    return O.crop_to_largest_object(g, _bg(g), held["largest_object_box"]["connectivity"])

def _c_scale(g, held):
    import dsl
    return dsl.scale_up(g, held["scale"]["fr"], held["scale"]["fc"])


# ──────────────────────────────────────────────────────────────────────────
# TERMINUS — held invariant-SET JOINTLY selects + parameterises a construction
# ──────────────────────────────────────────────────────────────────────────
# The selection map: a frozenset of invariant names (the ones that MATTER for the
# construction) -> the construction to run. Order matters: more specific
# (larger / more constraining) invariant-sets are tried first. AIDA will LEARN
# and extend this map from feedback; this seed is the hand-authored starting set.
SELECTION = [
    # (required invariants present, construction)
    (frozenset({"symmetry_completion"}), _c_symmetry),
    (frozenset({"largest_object_box"}), _c_largest_box),
    (frozenset({"scale"}), _c_scale),
    (frozenset({"colour_map", "same_shape"}), _c_colour_map),
    (frozenset({"colour_map"}), _c_colour_map),
]


def terminus(test_input, held, selection=SELECTION):
    """The held invariant SET jointly selects ONE construction and runs it.
    Tries selection rules whose required invariants are all present, most
    specific first. Returns (output_grid, construction_reason) or (None, None)."""
    held_names = set(held.keys())
    # most specific (largest required set) first
    for required, construction in sorted(selection, key=lambda x: -len(x[0])):
        if required <= held_names:
            try:
                return construction(test_input, held), tuple(sorted(required))
            except Exception:
                continue
    return None, None


# ──────────────────────────────────────────────────────────────────────────
# the full corrected solve: detect invariants -> terminus constructs
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class InvResult:
    held: dict                 # invariants that hold {name: params}
    output: Any                # constructed output grid (or None)
    reason: Any                # which invariant-set selected the construction


def solve(task):
    """Corrected pipeline: heads detect invariants across train pairs; the
    terminus jointly selects a construction from the held set and builds the
    output for the test input."""
    held = detect_invariants(task)
    out, reason = terminus(task.test[0].input, held)
    return InvResult(held=held, output=out, reason=reason)
