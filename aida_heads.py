"""
aida_heads.py — the AIDA accumulation loop: Gaiya's Least-Effort reasoning put on
ARC. This is the module that makes the paper's thesis DEMONSTRABLE rather than
asserted.

THE LEAST-EFFORT LOOP
=====================
Gaiya never re-reasons what it already knows, nor reasons about what is clearly
inapplicable. Concretely, per task:

  1. CONSULT (cheap): run every stored HEAD's GUARD against the task's perception
     features. A guard that doesn't match returns NOT_APPLICABLE — that head is
     skipped at ~zero cost. A matching guard proposes its RESOLVER (a program);
     we test it on the train pairs. If one solves the task -> DONE, with NO
     search. Cost = a handful of guard checks + program tests.

  2. ESCALATE ONLY THE RESIDUAL: if no stored head solves it, the task escalates
     to the inner-shell SEARCH (search.solve_instrumented) — the expensive path.
     This is the cost a pure DSL searcher pays on EVERY task; Gaiya pays it only
     when its accumulated knowledge is insufficient.

  3. CRYSTALLIZE (AIDA): when the search solves a task, AIDA induces heads from
     the solve:
        - a PRECISE head: the exact program, guarded by the exact perception
          signature (always kept — no harm, Sonia's ruling).
        - a GENERALIZED head: the same program, guarded by a SUBSET of perception
          features (the ones shared across the task's train pairs), validated on
          OTHER training tasks and shipped only if it clears a THRESHOLD without
          false-firing. The generalization is trusted because it SURVIVES held-out
          tasks, not because it is elegant — the anti-overfit gate.

CONTRAST (the paper's two figures)
==================================
  * Accumulation curve: states-explored per task falls for Gaiya as heads
    accumulate; stays flat for the DSL baseline (no store). DSL cannot produce a
    falling curve — it has no memory. The core proof of Least Effort.
  * Pruning: fraction of the hypothesis space removed by guards before search.

Pure stdlib. Guards read perception.py facts only (no new machinery).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import perception as P
import search as S

Grid = list


# ──────────────────────────────────────────────────────────────────────────
# perception FEATURES a guard reads (all from perception.py — no new facts)
# ──────────────────────────────────────────────────────────────────────────
def task_signature(task) -> dict:
    """A compact, rule-blind feature signature of a task, computed from the FIRST
    train pair's input/output via perception. These are the signals a guard
    matches on. All are facts perception already derives — field-law honest."""
    i, o = task.train[0].input, task.train[0].output
    pi = P.perceive(i)
    di, do = P.dimensions(i), P.dimensions(o)
    sig = {
        "in_dims": di,
        "out_dims": do,
        "same_shape": di == do,
        "size_ratio": (do[0] / di[0] if di[0] else 0,
                       do[1] / di[1] if di[1] else 0),
        "n_colours_in": len(pi.palette),
        "out_smaller": (do[0] * do[1]) < (di[0] * di[1]),
        "out_larger": (do[0] * do[1]) > (di[0] * di[1]),
        "has_symmetry_in": any(pi.symmetry.values()),
        "n_objects_4": len(pi.objects4),
        "n_objects_8": len(pi.objects8),
    }
    return sig


def _match(sig: dict, guard_feats: dict) -> bool:
    """A guard matches a task iff every feature it CARES about agrees. A guard
    that cares about fewer features is more GENERAL (fires on more tasks)."""
    for k, v in guard_feats.items():
        if sig.get(k) != v:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────
# a head: guard (perception features it requires) + resolver (a program)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Head:
    """Accumulated knowledge: 'when a task's perception matches guard_feats, try
    this program.' kind is 'precise' or 'generalized' (provenance for the paper).
    program is a search.Program."""
    guard_feats: dict
    program: "S.Program"
    kind: str
    origin_task: str = ""

    def applies(self, sig: dict) -> bool:
        return _match(sig, self.guard_feats)


class HeadStore:
    """Gaiya's growing knowledge. Empty at start; accumulates crystallized heads.
    Persists across tasks within a run — that persistence is what produces the
    falling compute curve."""
    def __init__(self):
        self.heads: list[Head] = []

    def add(self, head: Head):
        # avoid exact duplicates (same program + same guard)
        for h in self.heads:
            if h.program.name == head.program.name and h.guard_feats == head.guard_feats:
                return
        self.heads.append(head)

    def applicable(self, sig: dict) -> list:
        """The Least-Effort filter: only heads whose guard matches. Everything
        else is NOT_APPLICABLE and costs nothing."""
        return [h for h in self.heads if h.applies(sig)]


# ──────────────────────────────────────────────────────────────────────────
# crystallization: turn a solved task into heads (precise + generalized)
# ──────────────────────────────────────────────────────────────────────────
# the feature subset a GENERALIZED guard keeps — the structural signals most
# likely to be the REASON a transform applies, not incidental details.
_GENERAL_KEYS = ["same_shape", "out_larger", "out_smaller",
                 "has_symmetry_in", "size_ratio"]


def crystallize(task, program, store: HeadStore, validate_pool: list,
                threshold: float = 1.0) -> dict:
    """Make heads from a solve. Returns a small record of what was shipped.
    validate_pool: other tasks (with known outputs) to test a generalized guard
    against. threshold: fraction of its FIRINGS on the pool that must be CORRECT
    for the generalized head to ship (1.0 = must never false-fire)."""
    sig = task_signature(task)
    shipped = {"precise": False, "generalized": False}

    # PRECISE head: exact signature -> this program. Always kept.
    store.add(Head(guard_feats=dict(sig), program=program,
                   kind="precise", origin_task=task.task_id))
    shipped["precise"] = True

    # GENERALIZED head: keep only the general keys as the guard.
    gen_feats = {k: sig[k] for k in _GENERAL_KEYS if k in sig}
    # validate: among pool tasks whose signature MATCHES this generalized guard,
    # how often does running `program` reproduce ALL their train pairs?
    fires = 0
    correct = 0
    for vt in validate_pool:
        if vt.task_id == task.task_id:
            continue
        try:
            vsig = task_signature(vt)
        except Exception:
            continue
        if not _match(vsig, gen_feats):
            continue
        fires += 1
        if program.solves(vt):
            correct += 1
    # ship the generalized head only if it fired on the pool and never (or rarely,
    # per threshold) got it wrong. If it never fired, we can't vouch for it -> skip.
    if fires >= 1 and (correct / fires) >= threshold:
        store.add(Head(guard_feats=gen_feats, program=program,
                       kind="generalized", origin_task=task.task_id))
        shipped["generalized"] = True
        shipped["gen_fires"] = fires
        shipped["gen_correct"] = correct
    return shipped


# ──────────────────────────────────────────────────────────────────────────
# the accumulating solver — the Least-Effort loop, instrumented
# ──────────────────────────────────────────────────────────────────────────
def solve_with_accumulation(task, store: HeadStore, validate_pool: list,
                            max_programs: int = 2):
    """Consult heads first (cheap); escalate to search only on the residual;
    crystallize a fresh solve. Returns (programs, states_explored, route) where
    route is 'head' (solved from memory, ~0 search) or 'search' (had to reason).
    """
    sig = task_signature(task)

    # 1) CONSULT: try applicable heads. Each test is one 'state'.
    states = 0
    hits = []
    for h in store.applicable(sig):
        states += 1
        if h.program.solves(task):
            hits.append(h.program)
            if len(hits) >= max_programs:
                break
    if hits:
        return hits[:max_programs], states, "head"

    # 2) ESCALATE: the residual goes to the inner-shell search.
    progs, search_states = S.solve_instrumented(task, max_programs=max_programs)
    states += search_states

    # 3) CRYSTALLIZE: a fresh solve becomes knowledge for future tasks.
    if progs:
        crystallize(task, progs[0], store, validate_pool)

    return progs, states, "search"
