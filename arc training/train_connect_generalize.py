"""
train_connect_generalize.py — can a learned model generalize the connect_pairs
invariant BEYOND the surface forms we hand-coded?

THE QUESTION (Sonia, 2026-10-14)
================================
connect_pairs ("join two same-colour cells on a line by filling between them")
already recurred once BY HAND: pixel-level (task 40853293) and lattice-cell level
(06df4c85). We hand-coded both. Can a learned model, trained on the rule, GENERALISE
to surface forms we did NOT hand-code — and so win reach beyond what was given?

This is honest by construction: we HAVE the rule, so:
  * the hand-coded rule GENERATES unlimited labelled training grids (no fabricated
    signal — the construction is ground truth), and
  * the hand-coded rule is the ORACLE we check the model against.

THE GENERALISATION TEST (the whole point)
=========================================
Train ONLY on a restricted surface family (horizontal + vertical PIXEL pairs).
Test on surface forms the model NEVER saw:
  - DIAGONAL lines of same-colour cells,
  - pairs with WIDER spacing / more pairs,
  - (reported separately) whether it transfers at all to held-out configs.
If the model reproduces the rule on unseen surfaces, learning generalised the
invariant past the hand-coded forms. If it only fits what it trained on, the rigid
hand-coding was doing the real work.

MODEL (staged — per-cell first, CPU-trivial)
============================================
Per cell, predict output colour from local features that are size-invariant:
  - the cell's own colour (one-hot),
  - a KxK neighbourhood window (one-hot),
  - line-context bits: is there a same-colour cell further along each of the 8
    directions, and is the span between them currently background?
The per-cell model slides over any grid size. If it caps on harder surfaces
(thick bars, lattice), that justifies escalating to a small CNN — not before.

REQUIREMENTS (CPU): pip install torch numpy
USAGE: python train_connect_generalize.py [--epochs 60] [--train_n 4000]
"""
from __future__ import annotations
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("Install torch first:  pip install torch numpy")

np.random.seed(0)
torch.manual_seed(0)

NCOL = 10          # ARC colours 0..9
BG = 0
K = 2              # neighbourhood half-window (window is (2K+1)^2 = 25 cells)
DIRS8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]


# ── the GROUND-TRUTH rule (same as relational_ops connect_pairs) ─────────────
def connect_pairs_rule(grid, diagonal=False):
    """Join same-colour cells sharing a row/col (and diagonal if enabled) by
    filling the background cells strictly between them with that colour. Vertical
    drawn last (over horizontal), matching the hand-coded form."""
    from collections import defaultdict
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pos = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != BG:
                pos[grid[r][c]].append((r, c))
    # horizontal
    for col, cells in pos.items():
        byrow = defaultdict(list)
        for r, c in cells: byrow[r].append(c)
        for r, cs in byrow.items():
            if len(cs) >= 2:
                for c in range(min(cs), max(cs) + 1): out[r][c] = col
    # vertical (over horizontal)
    for col, cells in pos.items():
        bycol = defaultdict(list)
        for r, c in cells: bycol[c].append(r)
        for c, rs in bycol.items():
            if len(rs) >= 2:
                for r in range(min(rs), max(rs) + 1): out[r][c] = col
    if diagonal:
        # main & anti diagonals: cells sharing (r-c) or (r+c)
        for col, cells in pos.items():
            byd = defaultdict(list)
            for r, c in cells: byd[r - c].append((r, c))
            for k, cc in byd.items():
                if len(cc) >= 2:
                    cc.sort()
                    (r0, c0), (r1, c1) = cc[0], cc[-1]
                    r, c = r0, c0
                    while (r, c) != (r1, c1):
                        out[r][c] = col; r += 1; c += 1
                    out[r1][c1] = col
            bya = defaultdict(list)
            for r, c in cells: bya[r + c].append((r, c))
            for k, cc in bya.items():
                if len(cc) >= 2:
                    cc.sort()
                    (r0, c0), (r1, c1) = cc[0], cc[-1]
                    r, c = r0, c0
                    while (r, c) != (r1, c1):
                        out[r][c] = col; r += 1; c -= 1
                    out[r1][c1] = col
    return out


# ── random grid generators, by SURFACE FAMILY ───────────────────────────────
def _blank(h, w): return [[BG] * w for _ in range(h)]

def _span_clear(g, r0, c0, r1, c1, dr, dc):
    """Is every cell strictly between (r0,c0) and (r1,c1) along (dr,dc) background?
    Matches the real task: connect_pairs pairs are never obstructed."""
    r, c = r0 + dr, c0 + dc
    while (r, c) != (r1, c1):
        if g[r][c] != BG:
            return False
        r += dr; c += dc
    return True


def gen_axis(rng, h=10, w=10, npairs=2, diagonal=False):
    """Place `npairs` same-colour pairs along rows/cols (and diagonals if the
    family allows), ONLY where the span between them is clear — matching the real
    task 40853293, where same-colour pairs are never obstructed by another colour.
    Returns (input, output)."""
    g = _blank(h, w)
    placed = 0; tries = 0
    while placed < npairs and tries < 60:
        tries += 1
        col = int(rng.integers(1, NCOL))
        mode = int(rng.integers(0, 3 if diagonal else 2))
        if mode == 0:   # horizontal
            r = int(rng.integers(0, h)); a, b = sorted(rng.choice(w, 2, replace=False))
            if b - a >= 2 and g[r][a] == BG and g[r][b] == BG \
               and _span_clear(g, r, a, r, b, 0, 1):
                g[r][a] = col; g[r][b] = col; placed += 1
        elif mode == 1: # vertical
            c = int(rng.integers(0, w)); a, b = sorted(rng.choice(h, 2, replace=False))
            if b - a >= 2 and g[a][c] == BG and g[b][c] == BG \
               and _span_clear(g, a, c, b, c, 1, 0):
                g[a][c] = col; g[b][c] = col; placed += 1
        else:           # diagonal
            r0 = int(rng.integers(0, h - 3)); c0 = int(rng.integers(0, w - 3))
            length = int(rng.integers(3, min(h - r0, w - c0) + 1))
            r1, c1 = r0 + length - 1, c0 + length - 1
            if g[r0][c0] == BG and g[r1][c1] == BG \
               and _span_clear(g, r0, c0, r1, c1, 1, 1):
                g[r0][c0] = col; g[r1][c1] = col; placed += 1
    return g, connect_pairs_rule(g, diagonal=diagonal)


# ── per-cell features (size-invariant) ───────────────────────────────────────
def _scan(grid, r, c, dr, dc):
    """Scan from (r,c) in direction (dr,dc). Return (first_nonbg_colour or None,
    span_all_background_until_it: bool). Stops at the first non-background cell."""
    H, W = len(grid), len(grid[0])
    rr, cc = r + dr, c + dc
    while 0 <= rr < H and 0 <= cc < W:
        v = grid[rr][cc]
        if v != BG:
            return v, True        # reached a non-bg terminator through clear bg
        rr += dr; cc += dc
    return None, False            # ran off the grid with no terminator

def cell_features(grid, r, c):
    """Features that let a cell DECIDE connect_pairs locally:
      - own colour, KxK window (local shape),
      - for each of 8 directions: is there a non-bg terminator reachable through
        all-background cells, and (the key bit) does a terminator of a SPECIFIC
        colour exist on BOTH sides of this cell along a line? That 'flanked by the
        same colour on opposite ends, span clear' is exactly when connect_pairs
        fills a background cell. Computed honestly, not stubbed."""
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
    # directional terminators: colour reached each way (one-hot incl. 'none'),
    # plus the 4 opposite-axis 'flanked by same colour, span clear' bits — the
    # real connect_pairs predicate for a background cell.
    term = np.zeros(len(DIRS8) * (NCOL + 1), dtype=np.float32)   # +1 = 'none'
    reached = []
    for i, (dr, dc) in enumerate(DIRS8):
        col, clear = _scan(grid, r, c, dr, dc)
        reached.append((col, clear))
        term[i * (NCOL + 1) + (col if col is not None else NCOL)] = 1.0
    feats.append(term)
    # flanked bits: for the 4 axes (H, V, main-diag, anti-diag), both ends reached
    # through clear bg AND the two terminators are the SAME colour -> fill with it.
    axes = [(0, 1), (2, 3), (4, 7), (5, 6)]   # indices into DIRS8 for opposite dirs
    flank = np.zeros(len(axes), dtype=np.float32)
    for j, (a, b) in enumerate(axes):
        ca, cla = reached[a]; cb, clb = reached[b]
        if ca is not None and cb is not None and cla and clb and ca == cb and own == BG:
            flank[j] = 1.0
    feats.append(flank)
    return np.concatenate(feats)


def make_dataset(gen, n, rng, **kw):
    X, Y = [], []
    for _ in range(n):
        inp, outp = gen(rng, **kw)
        H, W = len(inp), len(inp[0])
        for r in range(H):
            for c in range(W):
                X.append(cell_features(inp, r, c))
                Y.append(outp[r][c])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


class CellNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, NCOL),
        )
    def forward(self, x): return self.net(x)


def train(X, Y, epochs=60, lr=2e-3, bs=4096):
    m = CellNet(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X); yt = torch.from_numpy(Y)
    n = len(X)
    for e in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(m(xt[idx]), yt[idx])
            loss.backward(); opt.step()
    return m


def grid_exact(m, gen, n, rng, **kw):
    """Fraction of generated grids the model reproduces EXACTLY (all cells)."""
    exact = 0; cellhit = 0; celltot = 0
    for _ in range(n):
        inp, outp = gen(rng, **kw)
        H, W = len(inp), len(inp[0])
        feats = np.array([cell_features(inp, r, c)
                          for r in range(H) for c in range(W)], dtype=np.float32)
        with torch.no_grad():
            pred = m(torch.from_numpy(feats)).argmax(1).numpy().reshape(H, W)
        tru = np.array(outp)
        if (pred == tru).all(): exact += 1
        cellhit += int((pred == tru).sum()); celltot += H * W
    return exact / n, cellhit / celltot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--train_n", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("TRAIN surface family: horizontal + vertical PIXEL pairs only")
    Xtr, Ytr = make_dataset(gen_axis, args.train_n, rng, diagonal=False)
    print(f"  {len(Xtr)} cell-examples, feat dim {Xtr.shape[1]}")
    m = train(Xtr, Ytr, epochs=args.epochs)

    print("\nGENERALISATION TEST (surfaces the model never trained on):")
    # 1) in-distribution (h/v pixel) — sanity, should be high
    e, c = grid_exact(m, gen_axis, 300, np.random.default_rng(1), diagonal=False)
    print(f"  in-dist (h/v pixel)      : grid-exact {e*100:.1f}%  cell {c*100:.1f}%")
    # 2) DIAGONAL lines — never seen
    e, c = grid_exact(m, gen_axis, 300, np.random.default_rng(2), diagonal=True)
    print(f"  diagonal lines (unseen)  : grid-exact {e*100:.1f}%  cell {c*100:.1f}%")
    # 3) more pairs / denser — never seen at this density
    e, c = grid_exact(m, lambda r, **k: gen_axis(r, npairs=4, **k),
                      300, np.random.default_rng(3), diagonal=False)
    print(f"  denser (4 pairs, unseen) : grid-exact {e*100:.1f}%  cell {c*100:.1f}%")
    # 4) bigger grids — never seen this size
    e, c = grid_exact(m, lambda r, **k: gen_axis(r, h=16, w=16, **k),
                      200, np.random.default_rng(4), diagonal=False)
    print(f"  16x16 grids (unseen size): grid-exact {e*100:.1f}%  cell {c*100:.1f}%")

    print("\nReading the result:")
    print("  - high in-dist + high on UNSEEN surfaces => learning generalised the")
    print("    invariant beyond the hand-coded forms.")
    print("  - high in-dist but low on unseen => per-cell model caps; escalate to CNN.")
    torch.save(m.state_dict(), "connect_generalize.pt")
    print("  saved -> connect_generalize.pt")


if __name__ == "__main__":
    main()
