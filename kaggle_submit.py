"""
kaggle_submit.py — the competition runner (Sonia, 2026-10-17).

ONE self-contained, OFFLINE, stdlib-only entry point for the ARC Prize 2026
ARC-AGI-2 Kaggle submission. It runs the FULL honest engine — the whole-invariant
library (relational_ops via recombination) AND the attribute auto-compose layer
(attributes + auto_compose) — and produces up to TWO distinct predictions per test
input (pass@2). Honest by construction: a prediction is emitted only if its
mechanism reproduced every training pair byte-exact; otherwise the slot is filled
with a safe fallback (the input unchanged) so the submission is always well-formed.

No internet, no model files, no API — compliant with Kaggle's offline evaluation.

MODES (auto-detected):
  - SCORE mode: if tasks carry test outputs (local eval/training), report the
    honest byte-exact pass@2 accuracy — the number the paper's Results section uses.
  - SUBMIT mode: write submission.json in the ARC/Kaggle shape
    {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]} (one entry per test input).

USAGE:
  python kaggle_submit.py --data <folder-of-task-json>        # score if outputs present
  python kaggle_submit.py --data <folder> --out submission.json
"""
from __future__ import annotations
import os, sys, json, glob, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---- engine imports (all stdlib-backed) -----------------------------------
import arc_data as D
import auto_compose as AC
try:
    import recombination as RC
    _HAVE_RC = True
except Exception:
    _HAVE_RC = False


def _fallback(test_input):
    return [row[:] for row in test_input]


def predictions_for_task(task, max_depth=2):
    """Up to 2 distinct byte-exact-justified predictions per test input.
    Primary: attribute auto-compose. Secondary: whole-invariant recombination.
    Each only contributes a prediction if it reproduced every train pair (its own
    honesty gate); we then de-dup and pad with the input-unchanged fallback."""
    # attribute engine (returns a single output for test[0]); generalise to all tests
    per_test = [[] for _ in task.test]

    # ENGINE 1: attribute auto-compose. auto_solve builds for test[0]; we re-run its
    # winning chain on every test input by calling the recipe search per test.
    try:
        out0, label, _ = AC.auto_solve(task, max_depth=max_depth)
    except Exception:
        out0, label = None, None
    if label is not None:
        # rebuild the winning op-chain and apply to each test input
        ops = []
        for prop in AC.PROPOSERS:
            try:
                ops.extend(prop(task))
            except Exception:
                pass
        chainmap = {nm: fn for nm, fn in ops}
        names = label.split(" ∘ ")
        for ti, tp in enumerate(task.test):
            g = tp.input
            ok = True
            for nm in names:
                fn = chainmap.get(nm)
                if fn is None:
                    ok = False; break
                try:
                    g = fn(g)
                except Exception:
                    g = None
                if g is None:
                    ok = False; break
            if ok and g is not None:
                per_test[ti].append(g)

    # ENGINE 2: whole-invariant recombination (if available)
    if _HAVE_RC:
        try:
            smap = RC.SelectionMap().seed()
            tr = RC.solve(task, smap, learn=False)
            if tr.route != "unsolved" and tr.construction is not None:
                for ti, tp in enumerate(task.test):
                    try:
                        g = RC._run_construction(tr.construction, tp.input, tr.held)
                    except Exception:
                        g = None
                    if g is not None:
                        per_test[ti].append(g)
        except Exception:
            pass

    # assemble up to 2 distinct predictions per test input, pad with fallback
    out = []
    for ti, tp in enumerate(task.test):
        uniq = []
        for g in per_test[ti] + [_fallback(tp.input)]:
            if g not in uniq:
                uniq.append(g)
            if len(uniq) >= 2:
                break
        while len(uniq) < 2:
            uniq.append(uniq[-1])
        out.append(uniq[:2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder of <task_id>.json files")
    ap.add_argument("--out", default=None, help="write submission.json here")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    AC.MAX_DEPTH = args.depth

    tasks = D.load_split(args.data)
    submission = {}
    scorable = 0
    solved = 0
    t0 = time.time()
    for tid, task in tasks.items():
        preds = predictions_for_task(task, max_depth=args.depth)
        submission[tid] = [{"attempt_1": p[0], "attempt_2": p[1]} for p in preds]
        if all(tp.output is not None for tp in task.test):
            scorable += 1
            ok = all(tp.output in preds[i] for i, tp in enumerate(task.test))
            if ok:
                solved += 1
    dt = time.time() - t0
    print(f"tasks={len(tasks)}  scorable={scorable}  solved={solved}  "
          f"accuracy={100*solved/scorable:.2f}%" if scorable else
          f"tasks={len(tasks)} (no ground truth; submission only)")
    print(f"time {dt:.0f}s  ({1000*dt/max(1,len(tasks)):.0f} ms/task)  "
          f"engines: auto_compose{'+recombination' if _HAVE_RC else ''}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(submission, f)
        print(f"wrote {args.out} ({len(submission)} tasks)")


if __name__ == "__main__":
    main()
