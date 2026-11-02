"""
learn_vs_handfit.py — THE DECISIVE TEST (Sonia, 2026-10-16).

Question: does TRAINING a head to pick an invariant's PARAMETER beat HAND-FITTING
it? This settles whether "train every invariant head on the whole corpus" (the
load-bearing half of the 100-invariant plan) is real, or whether the symbolic
hand-fit is already optimal and learning adds nothing.

SUBJECT: anchor_pivot's parameter = which colour is the fixed 180-rotation pivot.

STEP 0 (gating): how many tasks does anchor_pivot's STRUCTURE even hold on? If the
domain is tiny (< ~8), you cannot train a head on it — which is itself the answer:
most invariants have tiny domains, so "train every head" has nothing to train on.

If the domain is big enough, STEP 1-3 compare hand-fit vs a learned picker on a
held-out split. Pure stdlib for the domain scan (no torch needed); the learned
picker is a tiny logistic model in numpy if the domain warrants it.

HONEST: this test is built to be ABLE TO FAIL. A tie on easy cases where both
pick the obvious anchor is NOT a win for learning. Learning wins only if it
recovers tasks hand-fit gets wrong, on held-out.

Run on the full corpora:
    python learn_vs_handfit.py
"""
from __future__ import annotations
import os, sys, glob, json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arc_data as D
BG = 0


def _dims(g): return (len(g), len(g[0]) if g else 0)


def anchor_rotate(grid, anchor, bg=BG):
    H, W = _dims(grid)
    anch = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == anchor]
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
                    return None
    return out


def true_anchor(task):
    """The colour that, used as pivot, reproduces EVERY train pair byte-exact.
    Returns the colour, or None if no single colour works (structure doesn't hold)."""
    cols = set()
    for p in task.train:
        for row in p.input:
            cols.update(row)
    cols.discard(BG)
    good = []
    for a in cols:
        if all(anchor_rotate(p.input, a) == p.output for p in task.train):
            good.append(a)
    if len(good) == 1:
        return good[0]
    if len(good) > 1:
        return "AMBIG"   # several colours work on train — the interesting case
    return None


def find_dirs():
    try:
        import run_autocompose_full as RF
        return RF.find_arc_dirs(None, None)
    except Exception:
        home = os.path.expanduser("~")
        a1 = glob.glob(home + "/Downloads/ARC-AGI-master/data/training")
        a2 = glob.glob(home + "/Downloads/ARC-AGI-2-main/data/training")
        return (a1[0] if a1 else None), (a2[0] if a2 else None)


def hand_fit_anchor(task):
    """The symbolic rule: anchor = a colour occurring exactly once in the first
    input and unchanged in the first output. Returns a colour or None."""
    p = task.train[0]
    cnt = Counter(v for row in p.input for v in row)
    singles = [v for v, k in cnt.items() if k == 1 and v != BG]
    # keep only those whose single cell is unchanged in the output
    kept = []
    H, W = _dims(p.input)
    pos = {v: (r, c) for r in range(H) for c in range(W)
           for v in [p.input[r][c]] if v in singles}
    for v in singles:
        r, c = pos[v]
        if p.output[r][c] == v:
            kept.append(v)
    return kept[0] if len(kept) == 1 else (kept if kept else None)


def main():
    a1, a2 = find_dirs()
    print(f"AGI-1: {a1}\nAGI-2: {a2}")
    domain = []   # (task_id, true_anchor)
    for nm, fold in (("AGI-1", a1), ("AGI-2", a2)):
        if not fold:
            continue
        tasks = D.load_split(fold)
        for tid, task in tasks.items():
            ta = true_anchor(task)
            if ta is not None and ta != "AMBIG":
                domain.append((nm, tid, ta))
    print(f"\nanchor_pivot STRUCTURE holds on {len(domain)} tasks total:")
    for nm, tid, ta in domain:
        print(f"  {nm}  {tid}  true_anchor={ta}")
    if len(domain) < 8:
        print(f"\nVERDICT: domain too small ({len(domain)}) to train a head on.")
        print("For this invariant, hand-fit is the only honest option. If most")
        print("invariants look like this, 'train every head' has little to train on;")
        print("the lever is NAMING more invariants + fitting params symbolically.")
        return
    # domain is big enough: compare hand-fit accuracy vs a trivial learned baseline
    # (here just report hand-fit accuracy; a learned picker is worth building only
    # if hand-fit actually MISSES some — we measure that first).
    import run_autocompose_full  # noqa
    correct = 0; misses = []
    # rebuild tasks index for lookup
    idx = {}
    for nm, fold in (("AGI-1", a1), ("AGI-2", a2)):
        if fold:
            idx.update(D.load_split(fold))
    for nm, tid, ta in domain:
        hf = hand_fit_anchor(idx[tid])
        if hf == ta:
            correct += 1
        else:
            misses.append((tid, ta, hf))
    print(f"\nHAND-FIT picks the right anchor on {correct}/{len(domain)} domain tasks.")
    print(f"Hand-fit MISSES ({len(misses)}): {misses}")
    if not misses:
        print("\nVERDICT: hand-fit is already perfect on this invariant's domain.")
        print("Learning cannot beat a rule that is already 100%. Keep symbolic.")
    else:
        print("\nThese misses are where a learned picker COULD win. Worth building")
        print("a learned head ONLY if it recovers these on held-out. Next step.")


if __name__ == "__main__":
    main()
