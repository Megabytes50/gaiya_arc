"""
train_box_selector.py — train a small CPU model to LEARN the box-fill selector for
ARC task 09629e4f, the rule neither Sonia nor Claude could hand-derive.

WHAT THIS IS
============
A neuro-symbolic head keeps the SYMBOLIC structure fixed (divider-carved box grid;
each box fills solid with one colour or stays empty) and LEARNS the selector:
given a box, predict its fill. This script trains that selector on CPU.

INTEGRITY (read this)
=====================
The mapping box->fill is NOT hand-coded anywhere here. The model sees only:
  * the box's own 3x3 layout, colour-INVARIANT encoded (colours relabelled by
    order of appearance, the dropped colour marked, background marked) — so the
    model learns STRUCTURE, not absolute colours;
  * the box's grid position (which cell of the 3x3 lattice it is).
It predicts a colour-invariant fill label. We measure by LEAVE-ONE-PAIR-OUT: train
on 3 of the 4 train pairs, predict the 4th (held-out), and also predict the real
test pair. If it generalises to a pair it never saw, the rule was LEARNED.

A probe run established the ceiling: of 31 distinct box layouts across the 4 train
pairs, 28 map to a single fill and 3 conflict — so ~90% is learnable from the
box's own features; the rest needs grid context (included as position features).
The script reports accuracy honestly against that ceiling.

REQUIREMENTS (CPU only)
=======================
  pip install torch numpy        # torch CPU wheel is fine; no GPU needed
Runs in well under a minute on a laptop CPU.

USAGE
=====
  python train_box_selector.py                 # leave-one-out + test eval
  python train_box_selector.py --epochs 400     # override epochs

Outputs a trained model to box_selector.pt and prints per-pair accuracy. The
saved model drops into learned_selector.py as the fitted-selector replacement
(same interface: box features -> fill).
"""
from __future__ import annotations
import json, argparse, itertools
from collections import Counter
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("Install torch first:  pip install torch numpy  (CPU wheel is fine)")

LATT = 5          # lattice/divider colour in this task (detected generically below)
BG = 0
EMPTY = 0         # fill label for 'box stays empty'

torch.manual_seed(0)
np.random.seed(0)


# ──────────────────────────────────────────────────────────────────────────
# perception: carve the box grid, extract each box's 3x3 layout + position
# ──────────────────────────────────────────────────────────────────────────
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
        s = set(grid[r])
        if len(s) == 1 and next(iter(s)) != BG:
            return next(iter(s))
    for c in range(W):
        s = {grid[r][c] for r in range(H)}
        if len(s) == 1 and next(iter(s)) != BG:
            return next(iter(s))
    return None


def _dropped_colour(grid, divider, nboxes):
    cnt = Counter(v for row in grid for v in row if v not in (BG, divider))
    if not cnt:
        return None
    # the 'incomplete' colour: appears in fewer boxes than the full ones
    # (approximated by total count below the modal count)
    mx = max(cnt.values())
    low = [c for c, n in cnt.items() if n < mx]
    return low[0] if len(low) == 1 else None


def boxes_of(grid):
    """Yield (bi, bj, layout3x3, divider, dropped) for each box."""
    H, W = len(grid), len(grid[0])
    div = _divider(grid)
    dr = {r for r in range(H) if all(grid[r][c] == div for c in range(W))}
    dc = {c for c in range(W) if all(grid[r][c] == div for r in range(H))}
    rb = _bands(H, dr); cb = _bands(W, dc)
    dropped = _dropped_colour(grid, div, len(rb) * len(cb))
    out = []
    for bi, (r0, r1) in enumerate(rb):
        for bj, (c0, c1) in enumerate(cb):
            lay = [[grid[r][c] for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
            out.append((bi, bj, lay, div, dropped, (len(rb), len(cb))))
    return out


# ──────────────────────────────────────────────────────────────────────────
# colour-invariant encoding: relabel by first appearance; mark dropped & bg.
# This is what makes the learned rule generalise across palettes.
# ──────────────────────────────────────────────────────────────────────────
N_SLOT = 6        # 0=bg, 1..4 = relabelled colours by appearance, 5 = dropped

def encode_box(lay, dropped):
    mp = {}; nxt = 1
    def code(v):
        nonlocal nxt
        if v == BG: return 0
        if v == dropped: return 5
        if v not in mp:
            mp[v] = nxt; nxt = min(nxt + 1, 4)
        return mp[v]
    flat = [code(v) for row in lay for v in row]     # 9 cells
    # one-hot each of the 9 cells over N_SLOT -> 9*6 = 54 features
    oh = np.zeros((9, N_SLOT), dtype=np.float32)
    for i, c in enumerate(flat):
        oh[i, c] = 1.0
    return oh.reshape(-1), mp


def encode_fill(fill_colour, mp):
    """Fill label in the SAME relabelled space (0=empty, 1..4=relabelled)."""
    if fill_colour == 0:
        return 0
    return mp.get(fill_colour, 0)


def pos_features(bi, bj, grid_shape):
    gr, gc = grid_shape
    # normalized position + one-hot of the 9 lattice cells (assumes up to 3x3)
    feat = np.zeros(2 + 9, dtype=np.float32)
    feat[0] = bi / max(1, gr - 1)
    feat[1] = bj / max(1, gc - 1)
    idx = bi * gc + bj
    if idx < 9:
        feat[2 + idx] = 1.0
    return feat


# ──────────────────────────────────────────────────────────────────────────
# build a dataset from a set of (input, output) pairs, with colour-permutation
# augmentation (the rule must be colour-invariant, so every palette permutation
# is a valid extra example — turns 36 boxes into thousands).
# ──────────────────────────────────────────────────────────────────────────
def build_examples(pairs, augment=40):
    X, Y = [], []
    for inp, outp in pairs:
        # the actual palette (non-bg, non-divider) to permute over
        div = _divider(inp)
        palette = sorted({v for row in inp for v in row if v not in (BG, div)})
        for a in range(augment):
            if a == 0:
                perm = {c: c for c in palette}
            else:
                shuffled = palette[:]
                np.random.shuffle(shuffled)
                perm = {c: s for c, s in zip(palette, shuffled)}
            pinp = [[perm.get(v, v) for v in row] for row in inp]
            pout = [[perm.get(v, v) for v in row] for row in outp]
            for (bi, bj, lay, d, dropped, gshape) in boxes_of(pinp):
                feat_box, mp = encode_box(lay, dropped)
                feat_pos = pos_features(bi, bj, gshape)
                # the true fill of this box in pout
                r0 = bi; c0 = bj  # recompute fill directly from pout box region
                # find the box region again in pout
                fills = _box_fill(pinp, pout, bi, bj)
                label = encode_fill(fills, mp)
                X.append(np.concatenate([feat_box, feat_pos]))
                Y.append(label)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


def _box_fill(inp, outp, bi, bj):
    H, W = len(inp), len(inp[0])
    div = _divider(inp)
    dr = {r for r in range(H) if all(inp[r][c] == div for c in range(W))}
    dc = {c for c in range(W) if all(inp[r][c] == div for r in range(H))}
    rb = _bands(H, dr); cb = _bands(W, dc)
    r0, r1 = rb[bi]; c0, c1 = cb[bj]
    fs = {outp[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
          if outp[r][c] not in (BG, div)}
    return next(iter(fs)) if fs else 0


# ──────────────────────────────────────────────────────────────────────────
# the model: a small MLP. Input ~= 54 (box) + 11 (pos) = 65 features.
# Output: 5 classes (0=empty, 1..4 = relabelled fill colour). CPU-trivial.
# ──────────────────────────────────────────────────────────────────────────
class Selector(nn.Module):
    def __init__(self, in_dim, hidden=64, n_cls=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_cls),
        )
    def forward(self, x):
        return self.net(x)


def train(X, Y, epochs=300, lr=1e-3):
    m = Selector(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X); yt = torch.from_numpy(Y)
    for e in range(epochs):
        opt.zero_grad()
        out = m(xt)
        loss = lossf(out, yt)
        loss.backward(); opt.step()
    return m


def accuracy(m, X, Y):
    with torch.no_grad():
        pred = m(torch.from_numpy(X)).argmax(1).numpy()
    return (pred == Y).mean(), pred


# ──────────────────────────────────────────────────────────────────────────
# main: leave-one-pair-out on the 4 train pairs + eval on the real test pair
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--augment", type=int, default=40)
    ap.add_argument("--task", default="09629e4f.json")
    args = ap.parse_args()

    t = json.load(open(args.task))
    train_pairs = [(p["input"], p["output"]) for p in t["train"]]
    test_pair = [(p["input"], p.get("output")) for p in t["test"]]

    print(f"task: {args.task}  |  {len(train_pairs)} train pairs, {len(test_pair)} test")
    print("=" * 60)

    # LEAVE-ONE-PAIR-OUT: the honest generalisation test
    print("LEAVE-ONE-PAIR-OUT (train on 3 pairs, predict the held-out 4th):")
    loo_scores = []
    for held in range(len(train_pairs)):
        tr = [train_pairs[i] for i in range(len(train_pairs)) if i != held]
        Xtr, Ytr = build_examples(tr, augment=args.augment)
        Xte, Yte = build_examples([train_pairs[held]], augment=1)   # no aug on test
        m = train(Xtr, Ytr, epochs=args.epochs)
        acc, pred = accuracy(m, Xte, Yte)
        loo_scores.append(acc)
        print(f"  held-out pair {held}: box-fill accuracy = {acc*100:.1f}%  "
              f"({int(acc*len(Yte))}/{len(Yte)} boxes)")
    print(f"  mean leave-one-out accuracy: {np.mean(loo_scores)*100:.1f}%")
    print(f"  (probe ceiling from feature-consistency: ~90% — 28/31 layouts)")

    # FINAL: train on all 4 pairs, save, and predict the real test grid's boxes
    print("\nFINAL MODEL (train on all 4 pairs) + TEST-pair prediction:")
    Xall, Yall = build_examples(train_pairs, augment=args.augment)
    m = train(Xall, Yall, epochs=args.epochs)
    torch.save(m.state_dict(), "box_selector.pt")
    print("  saved -> box_selector.pt")

    if test_pair[0][1] is not None:
        Xte, Yte = build_examples([test_pair[0]], augment=1)
        acc, pred = accuracy(m, Xte, Yte)
        print(f"  TEST-pair box-fill accuracy = {acc*100:.1f}% "
              f"({int(acc*len(Yte))}/{len(Yte)} boxes)")
        print("  (this pair was NEVER in training — true held-out)")
    else:
        # test output not available locally; just predict
        Xte, _ = build_examples([(test_pair[0][0], test_pair[0][0])], augment=1)
        with torch.no_grad():
            pred = m(torch.from_numpy(Xte)).argmax(1).numpy()
        print(f"  predicted fills (relabelled space) for {len(pred)} test boxes: {pred.tolist()}")


if __name__ == "__main__":
    main()
