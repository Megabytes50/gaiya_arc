"""
audit.py — HONEST ACCOUNTING (Sonia, 2026-10-17).

Question that triggered this: training appears to "work" but eval is 0. Suspicion:
the tasks counted as AGI-2 solves are the SAME task IDs we hand-derived invariants
from in AGI-1 — i.e. train-on-test, not transfer.

This establishes the truth with no interpretation:
  1. ID overlap between AGI-1 training and AGI-2 training folders.
  2. Every task the engine solves (train byte-exact) on AGI-2 training, labelled:
       - also in AGI-1 training?  (potential contamination)
       - in the HAND-DERIVED list below? (definite contamination: we built its invariant)
  3. The honest count: solves on AGI-2-training tasks that are NEITHER in AGI-1
     training NOR hand-derived = genuine generalization within the training split.
  4. Eval solves (should be 0, re-confirmed here).

Run on the Mac where both corpora live:
    python audit.py
"""
from __future__ import annotations
import os, sys, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arc_data as D
import auto_compose as AC
try:
    import recombination as RC
    HAVE_RC = True
except Exception:
    HAVE_RC = False

# Tasks we explicitly hand-derived invariants from across the whole project.
# (From the session logs / INSIGHTS. If a solve is here, its invariant was fitted
#  to it — a solve on it is memorization, not generalization.)
HAND_DERIVED = {
    "22168020","23581191","0b148d64","0ca9ddb6","0962bcdd","0a938d79","178fcbfb",
    "0dfd9992","10fcaaa3","11852cab","1190e5a7","137eaa0f","150deff5","1a07d186",
    "1b60fb0c","06df4c85","05269061","017c7c7b","00d62c1b","045e512c",
    "e760a62e","90347967","8ee62060","0d3d703e",
}

def find(a1, a2):
    try:
        import run_autocompose_full as RF
        return RF.find_arc_dirs(a1, a2)
    except Exception:
        home = os.path.expanduser("~")
        g1 = glob.glob(home+"/Downloads/ARC-AGI-master/data/training")
        g2 = glob.glob(home+"/Downloads/ARC-AGI-2-main/data/training")
        return (g1[0] if g1 else a1), (g2[0] if g2 else a2)

def ids(folder):
    return set(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(folder,"*.json")))

def solves(folder):
    """IDs the engine reproduces byte-exact on TRAIN pairs (fires), via either engine."""
    tasks = D.load_split(folder); out=set()
    for tid, task in tasks.items():
        o,label,_ = AC.auto_solve(task)
        if label is not None: out.add(tid); continue
        if HAVE_RC:
            try:
                sm=RC.SelectionMap().seed(); tr=RC.solve(task,sm,learn=False)
                if tr.route!="unsolved": out.add(tid)
            except Exception: pass
    return out

def main():
    a1,a2 = find(None,None)
    a2eval = a2.replace("training","evaluation") if a2 else None
    print("AGI-1 train:",a1); print("AGI-2 train:",a2); print("AGI-2 eval:",a2eval)
    id1, id2 = ids(a1), ids(a2)
    overlap = id1 & id2
    print(f"\nID OVERLAP AGI-1-train vs AGI-2-train: {len(overlap)} shared task IDs "
          f"(AGI-1={len(id1)}, AGI-2={len(id2)})")

    s2 = solves(a2)
    print(f"\nEngine fires (train byte-exact) on {len(s2)} AGI-2-TRAIN tasks.")
    contaminated_overlap = s2 & id1
    contaminated_hand   = s2 & HAND_DERIVED
    genuine = s2 - id1 - HAND_DERIVED
    print(f"  of those, also in AGI-1 training (overlap): {len(contaminated_overlap)}")
    print(f"  of those, in our hand-derived list:         {len(contaminated_hand)}")
    print(f"  GENUINE (neither overlap nor hand-derived):  {len(genuine)}")
    print(f"    genuine IDs: {sorted(genuine)}")

    if a2eval and os.path.isdir(a2eval):
        se = solves(a2eval)
        print(f"\nEngine fires on AGI-2 EVAL: {len(se)}/{len(ids(a2eval))}  {sorted(se)}")
    print("\n=== HONEST BOTTOM LINE ===")
    print("Genuine within-training generalization + any eval solves = the real number.")
    print("Everything in overlap/hand-derived is train-on-test and must NOT be")
    print("reported as transfer.")

if __name__ == "__main__":
    main()
