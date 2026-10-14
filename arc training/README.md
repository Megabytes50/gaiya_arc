# Learned box-selector — single-task training (09629e4f)

A CPU-only experiment: can a small learned model pick up the box-fill rule for ARC
task `09629e4f` — the rule neither of us could hand-derive?

## What it does

The symbolic part is fixed (divider-carved box grid; each box fills with one colour
or stays empty). The model LEARNS the selector: given a box's own 3×3 layout
(colour-invariant encoded) plus its grid position, predict the box's fill.

The box→fill mapping is not hand-coded anywhere. The model learns it from the 4
real train pairs (36 boxes), with colour-permutation augmentation so the rule is
forced to be colour-invariant.

## Run it

```bash
cd train_selector
pip install -r requirements.txt      # torch CPU wheel + numpy; no GPU needed
python train_box_selector.py
```

Runs in under a minute on a laptop CPU.

## What to look for

Two numbers decide whether the learned selector is real:

1. **Leave-one-pair-out accuracy** — trains on 3 of the 4 train pairs and predicts
   the held-out 4th. This is the honest generalisation test: the model never saw
   that pair. Printed per held-out pair and as a mean.

2. **Test-pair accuracy** — trains on all 4 pairs and predicts the real test grid's
   boxes (never in training).

A probe established the ceiling: of 31 distinct box layouts across the train pairs,
28 map to a single fill and 3 conflict — so **~90% is the honest ceiling** for a
model that sees each box's own features. The remaining ~10% needs grid context that
these features may not fully capture.

- If leave-one-out lands near 90%: the selector LEARNED the rule from box structure.
  That's the result — a learned parameter extending the invariant where hand-coding
  failed.
- If it lands near chance (~55%, the empty-box base rate): the rule isn't in these
  features, and the box-fill genuinely needs cross-box reasoning we haven't encoded.
  That's also a real finding — it tells us what the feature set is missing.

Either way, report back the two numbers and we calibrate from there.

## Output

Saves the trained model to `box_selector.pt`. If it generalises, it drops into
`learned_selector.py` as the fitted-selector replacement (same interface: box
features → fill), so the head, partial-solve, and shell handoff keep working with a
LEARNED selector in place of a hand rule.

## Files

- `train_box_selector.py` — the trainer (data prep + MLP + leave-one-out eval)
- `09629e4f.json` — the task (4 train pairs, 1 test)
- `requirements.txt` — torch + numpy
