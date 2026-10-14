"""
run.py — run the solver over a dataset split, score it honestly, emit submission.

WHAT IT DOES
============
For every task in a split: run the layered search, take up to 2 solving programs
(pass@2), apply them to each test input to get up to 2 candidate output grids.
Score: a task counts as SOLVED iff, for EVERY test input, one of the 2 candidates
exactly equals the true output (the ARC criterion). Also emit a submission dict
in the ARC/Kaggle shape: {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
(one entry per test input).

HONEST EVALUATION PROTOCOL
==========================
Per the ARC-AGI-2 readme: do NOT tune against the evaluation set. We develop and
tune on TRAINING, and run EVALUATION once for the reported number. This script
takes a split name so the discipline is explicit in how it's called.

Pure stdlib. Degrades gracefully: a task the solver can't crack contributes two
fallback guesses (identity, and the input itself) so the submission is always
well-formed — a wrong guess scores 0, exactly as it should.
"""
from __future__ import annotations
import time
import arc_data as D
import search as S

Grid = list


def _fallbacks(test_input: Grid) -> list:
    """Two safe, well-formed guesses when the search finds nothing: the input
    unchanged (some tasks are near-identity), and a single-cell blank. Both will
    usually score 0 — the point is a VALID submission, never a crash."""
    return [[row[:] for row in test_input], [[0]]]


def solve_task_predictions(task, max_programs: int = 2) -> list:
    """For each test input, up to 2 candidate output grids (pass@2)."""
    progs = S.solve(task, max_programs=max_programs)
    preds_per_test = []
    for tp in task.test:
        cands = []
        for prog in progs:
            try:
                cands.append(prog.run(tp.input))
            except Exception:
                pass
        # pad to 2 with fallbacks; de-dup identical candidates
        uniq = []
        for g in cands + _fallbacks(tp.input):
            if g not in uniq:
                uniq.append(g)
            if len(uniq) >= 2:
                break
        preds_per_test.append(uniq[:2])
    return preds_per_test


def run_split(tasks: dict, max_programs: int = 2, verbose: bool = True) -> dict:
    """Run over a {task_id: Task} dict. Returns a report with the honest solve
    rate (scored only where ground-truth outputs are present) and the submission.
    """
    submission = {}
    solved = 0
    scorable = 0
    solved_ids = []
    t0 = time.time()

    for tid, task in tasks.items():
        preds = solve_task_predictions(task, max_programs=max_programs)
        submission[tid] = [{"attempt_1": p[0],
                            "attempt_2": p[1] if len(p) > 1 else p[0]}
                           for p in preds]
        # score only if every test pair has a known output (training/public eval)
        if all(tp.output is not None for tp in task.test):
            scorable += 1
            ok = True
            for tp, cand in zip(task.test, preds):
                if tp.output not in cand:
                    ok = False
                    break
            if ok:
                solved += 1
                solved_ids.append(tid)

    dt = time.time() - t0
    report = {
        "n_tasks": len(tasks),
        "scorable": scorable,
        "solved": solved,
        "solve_rate": (solved / scorable) if scorable else None,
        "seconds": dt,
        "ms_per_task": (dt / len(tasks) * 1000) if tasks else 0,
        "solved_ids": solved_ids,
        "submission": submission,
    }
    if verbose:
        rate = report["solve_rate"]
        print(f"tasks={report['n_tasks']} scorable={scorable} "
              f"solved={solved} rate={rate*100:.1f}%  "
              f"({report['ms_per_task']:.0f} ms/task, {dt:.0f}s total)")
    return report


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "_testdata/training"
    tasks = D.load_split(folder)
    run_split(tasks)
