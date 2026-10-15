"""
recurrence_scan.py — which single invariant recurs across the most ARC tasks?

Tests Sonia's hypothesis directly: is there a consistent invariant repeated
across the corpus that one named construction can lock everywhere it appears?
A battery of candidate deep-relations is run against every task's TRAIN pairs;
a candidate "fires" on a task only if its construction reproduces EVERY train
pair byte-exact (the engine's honesty gate). Candidates are ranked by fire
count. The top rows are the highest-yield invariants to build properly.

HONEST: firing on all train pairs is strong evidence but not proof the test
pair solves; the scanner also reports, for the top candidates, how many of
their fires ALSO reproduce the held-out test pair (where its output is in the
file). Fire-count ranks the hypothesis; test-confirm guards against train-fit.

numpy-free, pure stdlib. Self-locates the ARC folders.
USAGE: python recurrence_scan.py
"""
from __future__ import annotations
import os, json, glob, argparse
from collections import Counter, defaultdict

BG = 0


def dims(g): return (len(g), len(g[0]) if g and g[0] else 0)
def bg_of(g): return Counter(v for row in g for v in row).most_common(1)[0][0]
def copy(g): return [row[:] for row in g]


# ──────────────────────────────────────────────────────────────────────────
# candidate deep-relation CONSTRUCTIONS. Each: grid -> grid (or None if N/A).
# Kept same-shape unless the relation inherently changes size. A candidate that
# throws or returns None simply doesn't fire.
# ──────────────────────────────────────────────────────────────────────────
def _components(grid, bg, conn8=True):
    H, W = dims(grid); seen=[[False]*W for _ in range(H)]; comps=[]
    ds=[(-1,0),(1,0),(0,-1),(0,1)]+([(-1,-1),(-1,1),(1,-1),(1,1)] if conn8 else [])
    for r in range(H):
        for c in range(W):
            if grid[r][c]!=bg and not seen[r][c]:
                st=[(r,c)]; seen[r][c]=True; cells=[]
                while st:
                    a,b=st.pop(); cells.append((a,b))
                    for dr,dc in ds:
                        na,nb=a+dr,b+dc
                        if 0<=na<H and 0<=nb<W and grid[na][nb]!=bg and not seen[na][nb]:
                            seen[na][nb]=True; st.append((na,nb))
                comps.append(cells)
    return comps

def c_identity(g): return copy(g)

def c_denoise(g):
    """Remove single-cell specks (non-bg cells with no non-bg 4-neighbour)."""
    H,W=dims(g); bg=bg_of(g); out=copy(g); changed=False
    for r in range(H):
        for c in range(W):
            if g[r][c]!=bg:
                if not any(0<=r+dr<H and 0<=c+dc<W and g[r+dr][c+dc]!=bg
                           for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))):
                    out[r][c]=bg; changed=True
    return out if changed else None

def c_keep_largest(g):
    bg=bg_of(g); comps=_components(g,bg)
    if len(comps)<2: return None
    big=max(comps,key=len); H,W=dims(g); out=[[bg]*W for _ in range(H)]
    for (r,c) in big: out[r][c]=g[r][c]
    return out

def c_keep_smallest(g):
    bg=bg_of(g); comps=_components(g,bg)
    if len(comps)<2: return None
    small=min(comps,key=len); H,W=dims(g); out=[[bg]*W for _ in range(H)]
    for (r,c) in small: out[r][c]=g[r][c]
    return out

def _gravity(g, dr, dc):
    H,W=dims(g); bg=bg_of(g); out=[[bg]*W for _ in range(H)]
    if dr: # vertical
        for c in range(W):
            col=[g[r][c] for r in range(H) if g[r][c]!=bg]
            if dr>0:
                for i,v in enumerate(reversed(col)): out[H-1-i][c]=v
            else:
                for i,v in enumerate(col): out[i][c]=v
    else:
        for r in range(H):
            row=[v for v in g[r] if v!=bg]
            if dc>0:
                for i,v in enumerate(reversed(row)): out[r][W-1-i]=v
            else:
                for i,v in enumerate(row): out[r][i]=v
    return out
def c_gravity_down(g): return _gravity(g,1,0)
def c_gravity_up(g): return _gravity(g,-1,0)
def c_gravity_left(g): return _gravity(g,0,-1)
def c_gravity_right(g): return _gravity(g,0,1)

def c_rot180(g): return [row[::-1] for row in g[::-1]]
def c_flip_h(g): return [row[::-1] for row in g]
def c_flip_v(g): return g[::-1]
def c_transpose(g): return [list(r) for r in zip(*g)]

def c_border(g):
    H,W=dims(g); bg=bg_of(g)
    cols=Counter(v for row in g for v in row if v!=bg)
    if not cols: return None
    col=cols.most_common(1)[0][0]; out=copy(g)
    for c in range(W): out[0][c]=col; out[H-1][c]=col
    for r in range(H): out[r][0]=col; out[r][W-1]=col
    return out

def c_fill_enclosed(g):
    """Fill background regions not connected to the border with a fill colour =
    the single non-bg colour (if unambiguous)."""
    H,W=dims(g); bg=bg_of(g)
    nonbg=set(v for row in g for v in row if v!=bg)
    if len(nonbg)!=1: return None
    fill=next(iter(nonbg))
    outside=[[False]*W for _ in range(H)]; st=[]
    for r in range(H):
        for c in (0,W-1):
            if g[r][c]==bg and not outside[r][c]: outside[r][c]=True; st.append((r,c))
    for c in range(W):
        for r in (0,H-1):
            if g[r][c]==bg and not outside[r][c]: outside[r][c]=True; st.append((r,c))
    while st:
        r,c=st.pop()
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and g[nr][nc]==bg and not outside[nr][nc]:
                outside[nr][nc]=True; st.append((nr,nc))
    out=copy(g); changed=False
    for r in range(H):
        for c in range(W):
            if g[r][c]==bg and not outside[r][c]: out[r][c]=fill; changed=True
    return out if changed else None

def _complete_sym(g, kind):
    H,W=dims(g); bg=bg_of(g); out=copy(g)
    def partners(r,c):
        if kind=='h': return [(r,W-1-c)]
        if kind=='v': return [(H-1-r,c)]
        if kind=='r180': return [(H-1-r,W-1-c)]
        if kind=='both': return [(r,W-1-c),(H-1-r,c),(H-1-r,W-1-c)]
    for r in range(H):
        for c in range(W):
            if out[r][c]==bg:
                for pr,pc in partners(r,c):
                    if 0<=pr<H and 0<=pc<W and g[pr][pc]!=bg: out[r][c]=g[pr][pc]; break
    return out
def c_sym_h(g): return _complete_sym(g,'h')
def c_sym_v(g): return _complete_sym(g,'v')
def c_sym_both(g): return _complete_sym(g,'both')

def c_recolour_by_rank(g):
    """Recolour each object by its SIZE RANK into a fixed palette order. Fires only
    if that produces a consistent remap (heuristic candidate)."""
    bg=bg_of(g); comps=_components(g,bg)
    if len(comps)<2: return None
    order=sorted(range(len(comps)), key=lambda i:-len(comps[i]))
    H,W=dims(g); out=copy(g)
    palette=[1,2,3,4,5,6,7,8,9]
    for rank,idx in enumerate(order):
        col=palette[rank] if rank<len(palette) else palette[-1]
        for (r,c) in comps[idx]: out[r][c]=col
    return out

def c_connect_pairs(g):
    bg=bg_of(g); H,W=dims(g); out=copy(g)
    pos=defaultdict(list)
    for r in range(H):
        for c in range(W):
            if g[r][c]!=bg: pos[g[r][c]].append((r,c))
    for col,cells in pos.items():
        br=defaultdict(list); bc=defaultdict(list)
        for r,c in cells: br[r].append(c); bc[c].append(r)
        for r,cs in br.items():
            if len(cs)>=2:
                for c in range(min(cs),max(cs)+1): out[r][c]=col
        for c,rs in bc.items():
            if len(rs)>=2:
                for r in range(min(rs),max(rs)+1): out[r][c]=col
    return out


def c_periodic_completion(g):
    """The grid is a periodic tiling with BLACK (0) holes; find the smallest
    period whose residue classes are internally consistent over the non-hole
    cells and cover every cell, then fill the holes from the period. Holes are
    black, NOT the most-common colour (dense patterns are mostly a real colour).
    Guards: period >= 1 (no zero-period hang), skip grids over MAXDIM to bound
    cost on the full corpus."""
    HOLE = 0
    MAXDIM = 40
    H, W = dims(g)
    if H == 0 or W == 0 or H > MAXDIM or W > MAXDIM:
        return None
    for ph in range(1, H + 1):
        for pw in range(1, W + 1):
            if ph == H and pw == W:
                continue
            cls = {}; ok = True
            for r in range(H):
                for c in range(W):
                    v = g[r][c]
                    if v != HOLE:
                        k = (r % ph, c % pw)
                        if k in cls and cls[k] != v:
                            ok = False; break
                        cls[k] = v
                if not ok:
                    break
            if not ok:
                continue
            need = {(r % ph, c % pw) for r in range(H) for c in range(W)}
            if not need <= set(cls):
                continue
            return [[cls.get((r % ph, c % pw), g[r][c]) for c in range(W)]
                    for r in range(H)]
    return None


def _complete_sym_black(g, kind):
    """Symmetry completion with BLACK holes (not modal background)."""
    H, W = dims(g); HOLE = 0; out = copy(g)
    def partners(r, c):
        if kind == 'h': return [(r, W-1-c)]
        if kind == 'v': return [(H-1-r, c)]
        if kind == 'r180': return [(H-1-r, W-1-c)]
        if kind == 'both': return [(r, W-1-c), (H-1-r, c), (H-1-r, W-1-c)]
    changed = False
    for r in range(H):
        for c in range(W):
            if out[r][c] == HOLE:
                for pr, pc in partners(r, c):
                    if 0 <= pr < H and 0 <= pc < W and g[pr][pc] != HOLE:
                        out[r][c] = g[pr][pc]; changed = True; break
    return out if changed else None
def c_symblack_h(g): return _complete_sym_black(g, 'h')
def c_symblack_v(g): return _complete_sym_black(g, 'v')
def c_symblack_both(g): return _complete_sym_black(g, 'both')


CANDIDATES = {
    "identity": c_identity, "denoise": c_denoise,
    "keep_largest": c_keep_largest, "keep_smallest": c_keep_smallest,
    "gravity_down": c_gravity_down, "gravity_up": c_gravity_up,
    "gravity_left": c_gravity_left, "gravity_right": c_gravity_right,
    "rot180": c_rot180, "flip_h": c_flip_h, "flip_v": c_flip_v, "transpose": c_transpose,
    "add_border": c_border, "fill_enclosed": c_fill_enclosed,
    "sym_h": c_sym_h, "sym_v": c_sym_v, "sym_both": c_sym_both,
    "recolour_by_rank": c_recolour_by_rank, "connect_pairs": c_connect_pairs,
    # pattern-completion candidates with BLACK holes (the fix that surfaced
    # periodic_completion — modal-bg hid it before):
    "periodic_completion": c_periodic_completion,
    "symblack_h": c_symblack_h, "symblack_v": c_symblack_v, "symblack_both": c_symblack_both,
}


def fires_train(fn, task):
    for p in task["train"]:
        try:
            got = fn(p["input"])
        except Exception:
            return False
        if got is None or got != p["output"]:
            return False
    return True

def confirms_test(fn, task):
    tp = task.get("test", [])
    if not tp or tp[0].get("output") is None: return None
    try:
        return fn(tp[0]["input"]) == tp[0]["output"]
    except Exception:
        return False


def find_dirs(a1, a2):
    home=os.path.expanduser("~"); cands=[home+"/Downloads",home,os.getcwd(),home+"/gaiya_arc"]
    def hunt(pats):
        for base in cands:
            for pat in pats:
                for p in glob.glob(os.path.join(base,pat)):
                    if len(glob.glob(os.path.join(p,"*.json")))>=50: return p
        return None
    a1=a1 or hunt(["ARC-AGI-master/data/training","*AGI*/data/training"])
    a2=a2 or hunt(["ARC-AGI-2-main/data/training","*AGI-2*/data/*"])
    return a1,a2

def load(path):
    t={}
    for f in glob.glob(os.path.join(path,"*.json")):
        try: t[os.path.basename(f)[:-5]]=json.load(open(f))
        except: pass
    return t


def scan(name, tasks):
    fire=defaultdict(list); testok=defaultdict(int)
    for tid,t in tasks.items():
        if not t.get("train"): continue
        for cname, fn in CANDIDATES.items():
            if fires_train(fn, t):
                fire[cname].append(tid)
                if confirms_test(fn, t): testok[cname]+=1
    print(f"\n{'='*58}\n[{name}] {len(tasks)} tasks — invariant recurrence (train-fire)\n{'='*58}")
    print(f"{'candidate':20s} {'fires':>6s} {'test-confirmed':>15s}")
    for cname,ids in sorted(fire.items(), key=lambda x:-len(x[1])):
        print(f"  {cname:18s} {len(ids):6d} {testok[cname]:15d}")
    return fire


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--agi1"); ap.add_argument("--agi2")
    a=ap.parse_args(); a1,a2=find_dirs(a.agi1,a.agi2)
    print("ARC data:"); print(f"  AGI-1: {a1}\n  AGI-2: {a2}")
    if not a1 and not a2: raise SystemExit("No ARC folders found; pass --agi1 PATH --agi2 PATH")
    if a1: scan("AGI-1", load(a1))
    if a2: scan("AGI-2", load(a2))
    print("\nTop candidate = the most-recurring single invariant in the corpus.")
    print("A high fires count with high test-confirmed = the biggest honest jump to build.")


if __name__ == "__main__":
    main()
