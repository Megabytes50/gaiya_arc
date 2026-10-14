"""
scan_learned_connect.py — does the LEARNED connect_pairs solve real ARC tasks that
the RIGID hand-coded forms missed?

THE PAYOFF TEST (Sonia, 2026-10-14)
===================================
We trained a model on connect_pairs (join same-colour cells on a line) and showed
it generalises across surface forms. The real question: on actual ARC tasks, does
the learned model reproduce train pairs that NEITHER hand-coded form caught?
  - hand form A: pixel-level connect_pairs
  - hand form B: lattice-cell connect_pairs
A learned win only counts if it's on a task BOTH rigid forms miss. Anything the
hand forms already solve is not new reach.

HOW IT SCORES (honest)
======================
For each task, a solver "fires" if it reproduces EVERY train pair exactly (the same
honesty gate the whole engine uses). We report:
  - tasks the rigid pixel form fires on
  - tasks the rigid lattice form fires on
  - tasks the LEARNED model fires on
  - **NEW = learned fires AND neither rigid form fires** <- the result
For any NEW task we also check the model on the TEST pair (true held-out) so we
don't over-claim from train-fit alone.

SELF-LOCATING
=============
Finds the ARC task folders automatically (searches Downloads, home, cwd for the
standard ARC-AGI layouts). Prints what it found. If it finds nothing, it says so
and how to point it manually with --agi1 / --agi2.

REQUIREMENTS: pip install torch numpy    (same env as training)
USAGE: python scan_learned_connect.py
       python scan_learned_connect.py --model connect_generalize.pt
"""
from __future__ import annotations
import os, json, glob, argparse
from collections import defaultdict
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("Install torch first:  pip install torch numpy")

BG = 0
NCOL = 10
K = 2
DIRS8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]


# ── import the EXACT feature + model code from the trainer, so the loaded weights
#    match. We re-define them here to keep this script standalone. ──────────────
def _scan(grid, r, c, dr, dc):
    H, W = len(grid), len(grid[0])
    rr, cc = r + dr, c + dc
    while 0 <= rr < H and 0 <= cc < W:
        v = grid[rr][cc]
        if v != BG:
            return v, True
        rr += dr; cc += dc
    return None, False

def cell_features(grid, r, c):
    H, W = len(grid), len(grid[0])
    own = grid[r][c]
    feats = []
    oh = np.zeros(NCOL, dtype=np.float32); oh[own] = 1.0; feats.append(oh)
    for dr in range(-K, K + 1):
        for dc in range(-K, K + 1):
            w = np.zeros(NCOL, dtype=np.float32)
            rr, cc = r + dr, c + dc
            w[grid[rr][cc] if 0 <= rr < H and 0 <= cc < W else BG] = 1.0
            feats.append(w)
    term = np.zeros(len(DIRS8) * (NCOL + 1), dtype=np.float32)
    reached = []
    for i, (dr, dc) in enumerate(DIRS8):
        col, clear = _scan(grid, r, c, dr, dc)
        reached.append((col, clear))
        term[i * (NCOL + 1) + (col if col is not None else NCOL)] = 1.0
    feats.append(term)
    axes = [(0, 1), (2, 3), (4, 7), (5, 6)]
    flank = np.zeros(len(axes), dtype=np.float32)
    for j, (a, b) in enumerate(axes):
        ca, cla = reached[a]; cb, clb = reached[b]
        if ca is not None and cb is not None and cla and clb and ca == cb and own == BG:
            flank[j] = 1.0
    feats.append(flank)
    return np.concatenate(feats)

IN_DIM = NCOL + (2*K+1)**2 * NCOL + len(DIRS8) * (NCOL + 1) + 4   # 352

class CellNet(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NCOL),
        )
    def forward(self, x): return self.net(x)


def model_apply(m, grid):
    H, W = len(grid), len(grid[0])
    feats = np.array([cell_features(grid, r, c) for r in range(H) for c in range(W)],
                     dtype=np.float32)
    with torch.no_grad():
        pred = m(torch.from_numpy(feats)).argmax(1).numpy().reshape(H, W)
    return pred.tolist()


# ── rigid hand-coded forms (pixel + lattice) ─────────────────────────────────
def rigid_pixel(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pos = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != BG: pos[grid[r][c]].append((r, c))
    for col, cells in pos.items():
        byrow = defaultdict(list)
        for r, c in cells: byrow[r].append(c)
        for r, cs in byrow.items():
            if len(cs) >= 2:
                for c in range(min(cs), max(cs)+1): out[r][c] = col
    for col, cells in pos.items():
        bycol = defaultdict(list)
        for r, c in cells: bycol[c].append(r)
        for c, rs in bycol.items():
            if len(rs) >= 2:
                for r in range(min(rs), max(rs)+1): out[r][c] = col
    return out

def _bands(n, divs):
    out=[];s=None
    for i in range(n):
        if i in set(divs):
            if s is not None: out.append((s,i-1)); s=None
        else:
            if s is None: s=i
    if s is not None: out.append((s,n-1))
    return out

def rigid_lattice(grid):
    H, W = len(grid), len(grid[0])
    latt=None
    for r in range(H):
        if len(set(grid[r]))==1 and grid[r][0]!=BG: latt=grid[r][0]; break
    if latt is None:
        for c in range(W):
            col={grid[r][c] for r in range(H)}
            if len(col)==1 and next(iter(col))!=BG: latt=next(iter(col)); break
    if latt is None: return None
    dr={r for r in range(H) if all(grid[r][c]==latt for c in range(W))}
    dc={c for c in range(W) if all(grid[r][c]==latt for r in range(H))}
    rb=_bands(H,dr); cb=_bands(W,dc)
    cellcol={}
    for bi,(r0,r1) in enumerate(rb):
        for bj,(c0,c1) in enumerate(cb):
            cols=[grid[r][c] for r in range(r0,r1+1) for c in range(c0,c1+1) if grid[r][c] not in (BG,latt)]
            cellcol[(bi,bj)]=cols[0] if cols else None
    out=[row[:] for row in grid]
    def paint(bi,bj,col):
        r0,r1=rb[bi];c0,c1=cb[bj]
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                if grid[r][c]==BG: out[r][c]=col
    byrow=defaultdict(lambda:defaultdict(list)); bycol=defaultdict(lambda:defaultdict(list))
    for (bi,bj),col in cellcol.items():
        if col is not None: byrow[bi][col].append(bj); bycol[bj][col].append(bi)
    for bi,cols in byrow.items():
        for col,bjs in cols.items():
            if len(bjs)>=2:
                for bj in range(min(bjs),max(bjs)+1): paint(bi,bj,col)
    for bj,cols in bycol.items():
        for col,bis in cols.items():
            if len(bis)>=2:
                for bi in range(min(bis),max(bis)+1): paint(bi,bj,col)
    return out


def fires_all_train(solver, task):
    """Does `solver(input)` reproduce EVERY train pair exactly?"""
    for p in task["train"]:
        try:
            got = solver(p["input"])
        except Exception:
            return False
        if got is None or got != p["output"]:
            return False
    return True


# ── self-locating ARC task loader ────────────────────────────────────────────
def find_arc_dirs(agi1_arg, agi2_arg):
    home = os.path.expanduser("~")
    cands = [home + "/Downloads", home, os.getcwd(), home + "/gaiya_arc"]
    def hunt(patterns):
        for base in cands:
            for pat in patterns:
                for p in glob.glob(os.path.join(base, pat)):
                    # a dir holding many *.json ARC tasks
                    jsons = glob.glob(os.path.join(p, "*.json"))
                    if len(jsons) >= 50:
                        return p
        return None
    agi1 = agi1_arg or hunt([
        "ARC-AGI-master/data/training", "ARC-AGI-master/data/*",
        "ARC-AGI/data/training", "*AGI*/data/training", "*agi1*training*"])
    agi2 = agi2_arg or hunt([
        "ARC-AGI-2-main/data/training", "ARC-AGI-2-main/data/evaluation",
        "ARC-AGI-2*/data/*", "*AGI-2*/data/*", "*agi2*"])
    return agi1, agi2

def load_dir(path):
    tasks = {}
    if not path: return tasks
    for f in glob.glob(os.path.join(path, "*.json")):
        tid = os.path.splitext(os.path.basename(f))[0]
        try:
            with open(f) as fh:
                tasks[tid] = json.load(fh)
        except Exception:
            pass
    return tasks


def scan(name, tasks, m):
    pix = {tid for tid, t in tasks.items() if fires_all_train(rigid_pixel, t)}
    lat = {tid for tid, t in tasks.items() if fires_all_train(lambda g: rigid_lattice(g), t)}
    lrn = {tid for tid, t in tasks.items() if fires_all_train(lambda g: model_apply(m, g), t)}
    new = lrn - pix - lat
    print(f"\n[{name}] {len(tasks)} tasks")
    print(f"  rigid pixel   fires: {len(pix)}")
    print(f"  rigid lattice fires: {len(lat)}")
    print(f"  LEARNED       fires: {len(lrn)}")
    print(f"  NEW (learned only, both rigid miss): {len(new)}  {sorted(new)}")
    # verify NEW tasks on the held-out TEST pair (don't over-claim)
    for tid in sorted(new):
        t = tasks[tid]
        if t["test"] and t["test"][0].get("output") is not None:
            got = model_apply(m, t["test"][0]["input"])
            ok = got == t["test"][0]["output"]
            print(f"    {tid}: test-pair exact = {ok}")
        else:
            print(f"    {tid}: test output not in file (can't verify test)")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="connect_generalize.pt")
    ap.add_argument("--agi1", default=None)
    ap.add_argument("--agi2", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"model file not found: {args.model} (run the trainer first)")
    m = CellNet()
    m.load_state_dict(torch.load(args.model, map_location="cpu"))
    m.eval()

    agi1_dir, agi2_dir = find_arc_dirs(args.agi1, args.agi2)
    print("ARC data located:")
    print(f"  AGI-1: {agi1_dir}")
    print(f"  AGI-2: {agi2_dir}")
    if not agi1_dir and not agi2_dir:
        raise SystemExit("No ARC task folders found. Pass --agi1 PATH --agi2 PATH "
                         "(a folder of *.json task files).")

    agi1 = load_dir(agi1_dir)
    agi2 = load_dir(agi2_dir)
    new1 = scan("AGI-1", agi1, m) if agi1 else set()
    new2 = scan("AGI-2", agi2, m) if agi2 else set()

    print("\n" + "=" * 56)
    total_new = len(new1) + len(new2)
    if total_new:
        print(f"RESULT: the learned connect_pairs won {total_new} task(s) neither")
        print(f"rigid hand-form caught. Reach beyond what was given. IDs above.")
    else:
        print("RESULT: no NEW tasks — the learned model fired only where a rigid")
        print("form already did. Honest null: generalisation held on synthetic")
        print("surfaces but didn't surface a new real-ARC task this pass.")


if __name__ == "__main__":
    main()
