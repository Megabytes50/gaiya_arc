"""
run_autocompose_full.py — the REACH number for blind attribute auto-compose on the
FULL ARC corpora (Sonia, 2026-10-16).

WHAT IT ANSWERS
===============
The 150-task working set gave the SPARK (5/5 hand targets re-derived blind + 1 new,
zero wrong fires). This runs the SAME blind searcher (auto_compose.py — ONLY the
typed attribute primitives, no per-family recipe) over every task in AGI-1 and
AGI-2 and reports the honest reach:

  - FIRED      : chains that reproduced every TRAIN pair byte-exact (honesty gate)
  - TEST-EXACT : of those, how many also matched the held-out TEST pair exactly
                 (true generalisation — this is the number that counts)
  - OVERFIT    : fired on train but WRONG on test (train-fit was necessary-not-
                 sufficient; we report it honestly rather than hide it)
  - by recipe  : which attribute chains carried the solves, and at what depth

HONEST PROTOCOL
===============
A task counts as SOLVED only if the chain found from TRAIN reproduces the held-out
TEST pair exactly. We never peek at the test output while searching. OVERFIT is
reported, not suppressed.

SELF-LOCATING. Pure stdlib (no torch, no pip). Run from the gaiya_arc folder:
    python run_autocompose_full.py
or point it manually:
    python run_autocompose_full.py --agi1 PATH --agi2 PATH
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse
from collections import Counter, defaultdict

# import the attribute layer (must sit next to this file)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arc_data as D
import auto_compose as AC


def find_arc_dirs(a1, a2):
    home = os.path.expanduser("~")
    cands = [home + "/Downloads", home, os.getcwd(), HERE, home + "/gaiya_arc",
             HERE + "/arc training", home + "/gaiya_arc/arc training"]
    def hunt(pats):
        for base in cands:
            for pat in pats:
                for p in glob.glob(os.path.join(base, pat)):
                    if len(glob.glob(os.path.join(p, "*.json"))) >= 50:
                        return p
        return None
    a1 = a1 or hunt(["ARC-AGI-master/data/training", "ARC-AGI/data/training",
                     "*AGI*/data/training", "*agi1*training*"])
    a2 = a2 or hunt(["ARC-AGI-2-main/data/training", "ARC-AGI-2-main/data/evaluation",
                     "ARC-AGI-2*/data/*", "*AGI-2*/data/*"])
    return a1, a2


def run(name, folder):
    tasks = D.load_split(folder)
    n = len(tasks)
    fired = 0; test_exact = 0; overfit = 0; unscorable = 0
    by_recipe = Counter()        # label -> test-exact count
    by_depth = Counter()         # composition depth -> test-exact count
    fired_by_recipe = Counter()  # label -> fired count
    overfit_ids = []
    t0 = time.time()
    for tid, task in tasks.items():
        out, label, _considered = AC.auto_solve(task)
        if label is None:
            continue
        fired += 1
        fired_by_recipe[label] += 1
        tp = task.test[0]
        if tp.output is None:
            unscorable += 1
            continue
        if out == tp.output:
            test_exact += 1
            by_recipe[label] += 1
            by_depth[label.count("∘") + 1] += 1
        else:
            overfit += 1
            overfit_ids.append(tid)
    dt = time.time() - t0
    print(f"\n{'='*60}\n[{name}] {n} tasks   ({dt:.0f}s, {1000*dt/max(1,n):.0f} ms/task)\n{'='*60}")
    print(f"  fired (train byte-exact): {fired}")
    print(f"  TEST-EXACT (true solves): {test_exact}  = {100*test_exact/n:.1f}% of {name}")
    print(f"  overfit (train ok, test wrong): {overfit}   {overfit_ids[:12]}")
    if unscorable:
        print(f"  unscorable (no test output in file): {unscorable}")
    print(f"  solves by composition depth: {dict(sorted(by_depth.items()))}")
    print(f"  solves by recipe:")
    for label, c in by_recipe.most_common():
        print(f"    {c:4d}  {label}")
    return {"n": n, "fired": fired, "test_exact": test_exact, "overfit": overfit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agi1", default=None)
    ap.add_argument("--agi2", default=None)
    ap.add_argument("--depth", type=int, default=AC.MAX_DEPTH,
                    help="max composition depth (default %(default)s)")
    args = ap.parse_args()
    AC.MAX_DEPTH = args.depth

    a1, a2 = find_arc_dirs(args.agi1, args.agi2)
    print("ARC data located:")
    print(f"  AGI-1: {a1}")
    print(f"  AGI-2: {a2}")
    print(f"  max composition depth: {AC.MAX_DEPTH}")
    if not a1 and not a2:
        raise SystemExit("No ARC folders found. Pass --agi1 PATH --agi2 PATH "
                         "(each a folder of *.json task files).")
    tot = {"n": 0, "test_exact": 0, "overfit": 0, "fired": 0}
    for nm, fold in (("AGI-1", a1), ("AGI-2", a2)):
        if fold:
            r = run(nm, fold)
            for k in tot: tot[k] += r[k]
    print(f"\n{'#'*60}")
    print(f"COMBINED: {tot['test_exact']} true solves / {tot['n']} tasks "
          f"= {100*tot['test_exact']/max(1,tot['n']):.1f}%   "
          f"(fired {tot['fired']}, overfit {tot['overfit']})")
    print("Blind attribute auto-compose, no per-family recipe. This is the reach")
    print("of the attribute vocabulary ALONE — the headroom enrichment grows.")


if __name__ == "__main__":
    main()
