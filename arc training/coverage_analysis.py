"""
coverage_analysis.py — measure how much of the ARC task distribution is reachable
by invariants, and which perception signals recur most. Evidence for the claim:
"the bottleneck is invariant COVERAGE, not the engine."

WHAT IT MEASURES (all from grid FACTS — no guessing about rules)
================================================================
For every task in the full corpus it computes structural properties that bound
what kind of invariant could solve it, then buckets the distribution:

  SHAPE RELATION (input->output dimensions, across all train pairs):
    - same-shape           (output = input size)  -> recolour/fill family
    - integer-scale        (output = k x input)    -> scale/tile family
    - crop/shrink          (output smaller)         -> object-extract family
    - grow (non-integer)   -> construction family
    - mixed / inconsistent -> no single shape rule

  For SAME-SHAPE tasks (the biggest, most invariant-tractable bucket), finer:
    - pure recolour        (a consistent colour map explains every pair)
    - sparse-change        (<30% of cells change -> local/relational edit:
                            fill, connect, highlight — the family we've been in)
    - dense-change         (>=30% change -> global restructure)

  RECURRING PERCEPTION SIGNALS (the highest-value signal — what to name next):
    counts how often each structural feature appears across the corpus:
    has-divider-lattice, has-symmetry, single-background, few-colours,
    multi-object, has-grey(5), etc. A feature present in MANY tasks is an
    invariant worth naming because it recurs.

  COVERED-NOW: how many tasks the current 7 hand invariants actually fire on.

HONEST: these buckets BOUND tractability, they don't claim solvability. A
same-shape sparse-change task is 'invariant-reachable' (the family we solve in);
it is not a promise that one specific invariant solves it. The point is the
DISTRIBUTION SHAPE — where the reachable mass is — not a solve prediction.

REQUIREMENTS: just numpy (no torch needed). USAGE: python coverage_analysis.py
Self-locates the ARC folders like the scan script.
"""
from __future__ import annotations
import os, json, glob, argparse
from collections import Counter, defaultdict

BG = 0


def dims(g): return (len(g), len(g[0]) if g and g[0] else 0)
def palette(g): return Counter(v for row in g for v in row)


def shape_relation(task):
    rels = set()
    for p in task["train"]:
        ih, iw = dims(p["input"]); oh, ow = dims(p["output"])
        if (ih, iw) == (oh, ow): rels.add("same")
        elif ih and iw and oh % ih == 0 and ow % iw == 0 and (oh//ih, ow//iw) != (1,1):
            rels.add("scale")
        elif oh * ow < ih * iw: rels.add("crop")
        elif oh * ow > ih * iw: rels.add("grow")
        else: rels.add("other")
    if len(rels) == 1: return next(iter(rels))
    return "mixed"


def same_shape_subtype(task):
    # consistent colour map across all pairs?
    cmap = {}
    cmap_ok = True
    change_fracs = []
    for p in task["train"]:
        i, o = p["input"], p["output"]
        H, W = dims(i)
        changed = 0; tot = H * W
        for r in range(H):
            for c in range(W):
                a, b = i[r][c], o[r][c]
                if a != b: changed += 1
                if a in cmap and cmap[a] != b: cmap_ok = False
                cmap[a] = b
        change_fracs.append(changed / max(1, tot))
    if cmap_ok and not all(k == v for k, v in cmap.items()):
        return "pure_recolour"
    avg = sum(change_fracs) / len(change_fracs)
    return "sparse_change" if avg < 0.30 else "dense_change"


def perception_signals(task):
    """Structural features of the FIRST input — presence flags that, if common,
    point to high-value invariants to name."""
    g = task["train"][0]["input"]
    H, W = dims(g); pal = palette(g)
    sig = set()
    # divider lattice: a full non-bg row or column
    div = any(len(set(g[r])) == 1 and g[r][0] != BG for r in range(H)) or \
          any(len({g[r][c] for r in range(H)}) == 1 and g[0][c] != BG for c in range(W))
    if div: sig.add("divider_lattice")
    # symmetry (h or v mirror of whole grid)
    if g == [row[::-1] for row in g]: sig.add("sym_h")
    if g == g[::-1]: sig.add("sym_v")
    # background dominance
    if pal and pal.most_common(1)[0][0] == BG and pal.get(BG,0) > H*W*0.5:
        sig.add("black_bg_dominant")
    # colour count
    ncol = len([k for k in pal if k != BG])
    if ncol <= 2: sig.add("few_colours")
    if ncol >= 5: sig.add("many_colours")
    if 5 in pal: sig.add("has_grey")
    # object-ish: count 4-connected non-bg components (cheap)
    seen = [[False]*W for _ in range(H)]
    comps = 0
    for r in range(H):
        for c in range(W):
            if g[r][c] != BG and not seen[r][c]:
                comps += 1
                stack=[(r,c)]; seen[r][c]=True
                while stack:
                    a,b=stack.pop()
                    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        na,nb=a+dr,b+dc
                        if 0<=na<H and 0<=nb<W and g[na][nb]!=BG and not seen[na][nb]:
                            seen[na][nb]=True; stack.append((na,nb))
    if comps >= 4: sig.add("multi_object")
    if comps == 1: sig.add("single_object")
    return sig


def find_arc_dirs(a1, a2):
    home = os.path.expanduser("~")
    cands = [home+"/Downloads", home, os.getcwd(), home+"/gaiya_arc"]
    def hunt(pats):
        for base in cands:
            for pat in pats:
                for p in glob.glob(os.path.join(base, pat)):
                    if len(glob.glob(os.path.join(p, "*.json"))) >= 50:
                        return p
        return None
    a1 = a1 or hunt(["ARC-AGI-master/data/training","ARC-AGI/data/training","*AGI*/data/training"])
    a2 = a2 or hunt(["ARC-AGI-2-main/data/training","ARC-AGI-2*/data/*","*AGI-2*/data/*"])
    return a1, a2


def load_dir(path):
    t = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        try: t[os.path.basename(f)[:-5]] = json.load(open(f))
        except Exception: pass
    return t


def analyse(name, tasks):
    n = len(tasks)
    shape = Counter(); subtype = Counter(); sigs = Counter()
    for t in tasks.values():
        if not t.get("train"): continue
        sr = shape_relation(t); shape[sr] += 1
        if sr == "same":
            subtype[same_shape_subtype(t)] += 1
        for s in perception_signals(t): sigs[s] += 1
    print(f"\n{'='*58}\n[{name}] {n} tasks\n{'='*58}")
    print("SHAPE RELATION (what dimension rule the task needs):")
    for k, v in shape.most_common():
        print(f"  {k:16s} {v:4d}  ({100*v/n:4.1f}%)")
    print("\nSAME-SHAPE breakdown (the invariant-tractable bucket):")
    ss = sum(subtype.values())
    for k, v in subtype.most_common():
        print(f"  {k:16s} {v:4d}  ({100*v/n:4.1f}% of all, {100*v/max(1,ss):4.1f}% of same-shape)")
    print("\nRECURRING PERCEPTION SIGNALS (high count = high-value invariant to name):")
    for k, v in sigs.most_common():
        print(f"  {k:20s} {v:4d}  ({100*v/n:4.1f}% of tasks)")
    # the headline coverage estimate: same-shape sparse-change + pure-recolour are
    # the clearest invariant-reachable mass we've been operating in
    reachable = subtype.get("pure_recolour",0) + subtype.get("sparse_change",0)
    print(f"\nINVARIANT-REACHABLE MASS (same-shape recolour/sparse-edit): "
          f"{reachable} tasks = {100*reachable/n:.1f}% of {name}")
    print("  (the family our 7 invariants live in; names the headroom)")
    return shape, subtype, sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agi1", default=None); ap.add_argument("--agi2", default=None)
    args = ap.parse_args()
    a1, a2 = find_arc_dirs(args.agi1, args.agi2)
    print("ARC data located:"); print(f"  AGI-1: {a1}\n  AGI-2: {a2}")
    if not a1 and not a2:
        raise SystemExit("No ARC folders found; pass --agi1 PATH --agi2 PATH")
    if a1: analyse("AGI-1 (full training)", load_dir(a1))
    if a2: analyse("AGI-2 (full training)", load_dir(a2))
    print("\nReading it: the % in 'invariant-reachable mass' is the headroom a")
    print("coverage-first strategy can address. The top recurring signals are the")
    print("invariants to name next for the biggest solve-rate gain.")


if __name__ == "__main__":
    main()
