"""
near_miss.py — EVIDENCE-DRIVEN enrichment targeting (Sonia, 2026-10-16).

The full-corpus run showed the ceiling is now VOCABULARY, not search depth. This
finds WHERE to grow it: tasks where some attribute chain ALMOST reproduces the
train pairs (>= threshold cell agreement) but not byte-exact. A near-miss names a
missing attribute from the transfer target itself — so the next perception Sonia
names is the highest-value one, not a guess.

For each task it finds the chain with the best MEAN per-pair cell agreement across
train (same proposers/gate as auto_compose, but scored by agreement not pass/fail),
and reports tasks in [threshold, 1.0). It also prints cheap structural tags
(lattice / multi-object / symmetry / size-change) so clusters of near-misses point
at a specific attribute family.

HONEST: a near-miss is a HINT, not a solve. We never count these. Pure stdlib.
Run next to attributes.py + auto_compose.py + arc_data.py:
    python near_miss.py            (defaults: AGI-2, threshold 0.90, depth 2)
    python near_miss.py --agi1 PATH --agi2 PATH --threshold 0.85
"""
from __future__ import annotations
import os, sys, glob, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arc_data as D
import auto_compose as AC
import attributes as A

BG = 0


def _dims(g): return (len(g), len(g[0]) if g else 0)


def changed_agreement(inp, got, out):
    """Agreement measured ONLY on cells that DIFFER between input and output — the
    cells a real rule must change. This kills the trivial 'output≈input' inflation
    that makes identity chains look like near-misses. Returns 0 if dims mismatch or
    nothing changes (a pure-identity task is not a near-miss for us)."""
    if _dims(got) != _dims(out) or _dims(inp) != _dims(out):
        return 0.0
    H, W = _dims(out)
    changed = [(r, c) for r in range(H) for c in range(W) if inp[r][c] != out[r][c]]
    if not changed:
        return 0.0
    right = sum(1 for r, c in changed if got[r][c] == out[r][c])
    # also penalise cells the chain changed that it SHOULDN'T have
    wrong_extra = sum(1 for r in range(H) for c in range(W)
                      if inp[r][c] == out[r][c] and got[r][c] != out[r][c])
    return (right - wrong_extra) / len(changed)


def best_chain_agreement(task, max_depth=2):
    """Best mean-train-agreement over all gated chains (same ops as auto_compose).
    Returns (mean_agreement, label) or (0, None)."""
    ops = []
    for prop in AC.PROPOSERS:
        try:
            ops.extend(prop(task))
        except Exception:
            pass
    best = (0.0, None)
    from itertools import product
    for depth in range(1, max_depth + 1):
        for chain in product(ops, repeat=depth):
            agrees = []
            ok = True
            is_identity = True
            for p in task.train:
                g = p.input
                for _, fn in chain:
                    try:
                        g = fn(g)
                    except Exception:
                        g = None
                    if g is None:
                        ok = False; break
                if not ok:
                    break
                if g != p.input:
                    is_identity = False
                agrees.append(changed_agreement(p.input, g, p.output))
            if not ok or not agrees or is_identity:
                continue  # an identity-equivalent chain is not a near-miss
            m = sum(agrees) / len(agrees)
            if m > best[0]:
                best = (m, " ∘ ".join(n for n, _ in chain))
    return best


def tags(task):
    g = task.train[0].input
    t = []
    if A.lattice(g) is not None: t.append("lattice")
    if len(A.components(g)) >= 4: t.append("multi_object")
    if A.enclosure(g): t.append("enclosure")
    if g == [r[::-1] for r in g] or g == g[::-1]: t.append("symmetry")
    if any(_dims(p.input) != _dims(p.output) for p in task.train): t.append("size_change")
    return t


def run(name, folder, threshold, depth):
    tasks = D.load_split(folder)
    near = []
    for tid, task in tasks.items():
        # skip ones we already solve byte-exact
        out, label, _ = AC.auto_solve(task, max_depth=depth)
        if label is not None:
            continue
        m, chain = best_chain_agreement(task, depth)
        if threshold <= m < 1.0:
            near.append((m, tid, chain, tags(task)))
    near.sort(reverse=True)
    print(f"\n{'='*64}\n[{name}] near-misses >= {threshold:.0%} (unsolved): {len(near)}\n{'='*64}")
    tagc = Counter()
    for m, tid, chain, tg in near:
        print(f"  {m:5.1%}  {tid}  via [{chain}]   tags={tg}")
        for x in tg: tagc[x] += 1
    print(f"\n  tag frequency among near-misses: {dict(tagc.most_common())}")
    print("  (a cluster of one tag = a specific missing attribute to name next)")
    return near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agi1", default=None)
    ap.add_argument("--agi2", default=None)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    # reuse auto_compose's locator via run_autocompose_full if present
    try:
        import run_autocompose_full as RF
        a1, a2 = RF.find_arc_dirs(args.agi1, args.agi2)
    except Exception:
        a1, a2 = args.agi1, args.agi2
    print(f"AGI-1: {a1}\nAGI-2: {a2}\nthreshold {args.threshold}  depth {args.depth}")
    for nm, fold in (("AGI-1", a1), ("AGI-2", a2)):
        if fold:
            run(nm, fold, args.threshold, args.depth)


if __name__ == "__main__":
    main()
