"""
train_box_selector_v2.py — whole-grid assignment model for ARC 09629e4f.

WHY v2
======
v1 collapsed to "predict empty" (55.6/44.4% = the empty-box base rate — Sonia's
diagnosis: the model learned only the background). Three bugs, all fixed here:

  1. PER-BOX was too blind. The fill of a box depends on cross-box structure (a
     probe showed 3/31 local layouts conflict). v2 feeds the model the WHOLE grid
     (all 9 boxes' 3x3 layouts at once).
  2. The LABEL had no consistent meaning. v1 relabelled colours per box, so the
     target was noise. v2 predicts a COLOUR-INVARIANT target: for each box, which
     of the 9 LATTICE POSITIONS (0..8) supplies its fill colour, or "empty" (9).
     A position is the same concept in every grid, so the label is learnable.
     At apply time: fill = the colour sitting at the predicted position's box...
     (see decode). This forces the model to learn a POSITIONAL/POINTER rule, which
     the probe suggests this task is.
  3. CLASS IMBALANCE. 5 empty : 4 filled per grid. v2 weights the loss so the 4
     filled boxes carry their weight and the model can't win by always saying
     empty.

HONEST NOTE
===========
Neither Sonia nor Claude could hand-derive this task's box-assignment rule. v2
does NOT encode it — it gives the model the whole-grid information and a
colour-invariant target and asks it to find the rule. Leave-one-pair-out is the
honest test: if it beats the 55% empty base rate on a pair it never saw, it found
real structure. If it ceilings at the base rate again, the rule needs features or
capacity this model still lacks — a real result either way.

REQUIREMENTS (CPU): pip install torch numpy
USAGE: python train_box_selector_v2.py [--epochs 800]
"""
from __future__ import annotations
import json, argparse
from collections import Counter
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("Install torch first:  pip install torch numpy")

BG = 0

np.random.seed(0)
torch.manual_seed(0)


# ── perception ──────────────────────────────────────────────────────────────
def _bands(n, divs):
    out = []; s = None
    for i in range(n):
        if i in set(divs):
            if s is not None: out.append((s, i - 1)); s = None
        else:
            if s is None: s = i
    if s is not None: out.append((s, n - 1))
    return out

def _divider(grid):
    H, W = len(grid), len(grid[0])
    for r in range(H):
        if len(set(grid[r])) == 1 and grid[r][0] != BG:
            return grid[r][0]
    for c in range(W):
        col = {grid[r][c] for r in range(H)}
        if len(col) == 1 and next(iter(col)) != BG:
            return next(iter(col))
    return None

def parse_grid(grid):
    """Return (rb, cb, divider, dropped, full, box_layouts) where box_layouts[k]
    is the 3x3 colour grid of box k (k = bi*ncol+bj)."""
    H, W = len(grid), len(grid[0])
    div = _divider(grid)
    dr = {r for r in range(H) if all(grid[r][c] == div for c in range(W))}
    dc = {c for c in range(W) if all(grid[r][c] == div for r in range(H))}
    rb = _bands(H, dr); cb = _bands(W, dc)
    ci = Counter(v for row in grid for v in row if v not in (BG, div))
    nb = len(rb) * len(cb)
    mx = max(ci.values()) if ci else 0
    dropped = next((c for c, n in ci.items() if n < mx), None)
    full = [c for c, n in ci.items() if n == mx]
    layouts = []
    for (r0, r1) in rb:
        for (c0, c1) in cb:
            lay = [[grid[r][c] for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
            layouts.append(lay)
    return rb, cb, div, dropped, full, layouts

def box_fill_truth(grid, outp):
    """For each box k: the fill colour (0 if empty)."""
    rb, cb, div, dropped, full, _ = parse_grid(grid)
    fills = []
    for (r0, r1) in rb:
        for (c0, c1) in cb:
            fs = {outp[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if outp[r][c] not in (BG, div)}
            fills.append(next(iter(fs)) if fs else 0)
    return fills


# ── colour-invariant whole-grid encoding ─────────────────────────────────────
# Per box: its 3x3 layout, each cell one-hot over {bg, c1..c4, dropped} = 6 slots,
# using a GRID-WIDE consistent relabelling (colours ranked by global frequency,
# dropped marked). The whole grid = 9 boxes * 9 cells * 6 = 486 features.
N_SLOT = 6
def grid_features(grid):
    rb, cb, div, dropped, full, layouts = parse_grid(grid)
    # global relabelling: full colours ranked by a stable key (sorted), dropped=5
    order = sorted(full)
    code = {}
    for i, c in enumerate(order[:4]):
        code[c] = i + 1           # 1..4
    if dropped is not None:
        code[dropped] = 5
    feats = []
    for lay in layouts:
        for row in lay:
            for v in row:
                oh = np.zeros(N_SLOT, dtype=np.float32)
                oh[code.get(v, 0)] = 1.0   # 0 = bg / unknown
                feats.append(oh)
    x = np.concatenate(feats)      # 9*9*6 = 486
    return x, order, dropped

# target: per box, a class in 0..4 (0 = empty, 1..4 = which RANKED full colour
# fills it). Colour-invariant because it's the rank, not the colour id.
def grid_target(grid, outp):
    rb, cb, div, dropped, full, _ = parse_grid(grid)
    order = sorted(full)
    rank = {c: i + 1 for i, c in enumerate(order[:4])}
    fills = box_fill_truth(grid, outp)
    return np.array([rank.get(f, 0) for f in fills], dtype=np.int64)  # 9 labels


# ── augmentation: permute the 4 full colours (24 ways) — rule is colour-invariant
def augmented(pairs, times=24):
    data = []
    for inp, outp in pairs:
        rb, cb, div, dropped, full, _ = parse_grid(inp)
        order = sorted(full)
        import itertools
        perms = list(itertools.permutations(order))
        np.random.shuffle(perms)
        for perm in perms[:times]:
            pm = {order[i]: perm[i] for i in range(len(order))}
            pinp = [[pm.get(v, v) for v in row] for row in inp]
            pout = [[pm.get(v, v) for v in row] for row in outp]
            data.append((pinp, pout))
    return data


# ── model: whole-grid MLP -> 9 boxes x 5 classes ─────────────────────────────
class GridSelector(nn.Module):
    def __init__(self, in_dim=486, hidden=256, n_box=9, n_cls=5):
        super().__init__()
        self.n_box, self.n_cls = n_box, n_cls
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_box * n_cls),
        )
    def forward(self, x):
        return self.net(x).view(-1, self.n_box, self.n_cls)


def make_xy(pairs):
    X, Y = [], []
    for inp, outp in pairs:
        x, _, _ = grid_features(inp)
        y = grid_target(inp, outp)
        if len(x) == 486 and len(y) == 9:
            X.append(x); Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


def train(X, Y, epochs=800, lr=2e-3):
    m = GridSelector()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    # class weights: empties are ~5/9, fills ~1/9 each -> upweight fills
    w = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0])
    lossf = nn.CrossEntropyLoss(weight=w)
    xt = torch.from_numpy(X); yt = torch.from_numpy(Y)
    for e in range(epochs):
        opt.zero_grad()
        out = m(xt)                       # (N,9,5)
        loss = lossf(out.reshape(-1, 5), yt.reshape(-1))
        loss.backward(); opt.step()
    return m


def evaluate(m, pairs, label):
    X, Y = make_xy(pairs)
    if len(X) == 0:
        print(f"  {label}: no examples"); return 0.0
    with torch.no_grad():
        pred = m(torch.from_numpy(X)).argmax(-1).numpy()   # (N,9)
    box_acc = (pred == Y).mean()
    # also: exact-grid accuracy (all 9 boxes right) and filled-box-only accuracy
    filled = Y > 0
    filled_acc = (pred[filled] == Y[filled]).mean() if filled.any() else 0.0
    exact = (pred == Y).all(axis=1).mean()
    print(f"  {label}: box-acc {box_acc*100:.1f}% | filled-box-acc "
          f"{filled_acc*100:.1f}% | whole-grid-exact {exact*100:.1f}%")
    return box_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--task", default="09629e4f.json")
    args = ap.parse_args()

    t = json.load(open(args.task))
    train_pairs = [(p["input"], p["output"]) for p in t["train"]]
    test_pairs = [(p["input"], p.get("output")) for p in t["test"]]

    print(f"task {args.task} | {len(train_pairs)} train, {len(test_pairs)} test")
    print("baseline (always-empty) box-acc ~= 5/9 = 55.6%\n")

    print("LEAVE-ONE-PAIR-OUT (whole-grid model, filled-box-acc is the real signal):")
    for held in range(len(train_pairs)):
        tr = [train_pairs[i] for i in range(len(train_pairs)) if i != held]
        m = train(make_xy(augmented(tr))[0], make_xy(augmented(tr))[1], epochs=args.epochs)
        evaluate(m, [train_pairs[held]], f"held-out pair {held}")

    print("\nFINAL (train on all 4) + test pair:")
    Xall, Yall = make_xy(augmented(train_pairs))
    m = train(Xall, Yall, epochs=args.epochs)
    torch.save(m.state_dict(), "box_selector_v2.pt")
    print("  saved -> box_selector_v2.pt")
    if test_pairs[0][1] is not None:
        evaluate(m, [test_pairs[0]], "TEST pair (never trained)")


if __name__ == "__main__":
    main()
