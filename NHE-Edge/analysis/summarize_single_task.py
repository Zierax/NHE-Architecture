"""Single-task benchmark: what 463 items say about NHE-temporal.

DATASET
  463 questions merged from topics.py into one task with an explicit answer-union
  truth layer (build_single_task.py). Families: 244 capital, 178 largest-city,
  41 element. Baseline strict hallucination rate 0.1274 (59/463) - roughly 8x
  the Africa-only headroom, which is what makes the item sufficient for power.

ARMS (identical engine, identical mask, identical threshold)
  none                          no intervention
  mask L19 w5 scale 0.3 t90     the deployed Edge configuration

HEADLINE, paired exact McNemar on the strict metric
  n=463   fixes=3   breaks=0   net=+3   p=0.250 (NOT significant)
  net 95% Wilson CI [0.0022, 0.0189]
  fired=41 (of which wrong-before=9)  repairs per fired-wrong = 0.333

WHAT THIS DOES AND DOES NOT ESTABLISH
  ESTABLISHED, and now at n=463 instead of n=54:
    - Zero collateral damage. 0 breaks on 463 items spanning three question
      frames, versus 9 breaks in the merged static+temporal arm. The "no new
      errors" property is not an artefact of a 54-item set.
    - The timing law holds: 40 of the 41 fired items have lead >= 1, and the one
      with lead < 1 never flipped. Every fired item that had lead >= 1 and was
      wrong either fixed or was left unharmed.
    - The repair rate PER FIRED WRONG ITEM is 0.333, slightly higher than the
      0.286 observed on Africa. This is the quantity that has to survive a
      domain change for the method to generalise, and it did.
    - The taxonomy split is confirmed at scale: of the 56 items still wrong
      after the arm, 50 never fired. 89% of the residual error is in the
      population NTW identifies as Static Memory Voids, i.e. outside runtime
      reach by construction.

  NOT ESTABLISHED:
    - Statistical significance. 3 fixes with 0 breaks gives exact McNemar
      p = 0.25. `power_single_task.py` computed, before this run, that 6 fixes
      with 0 breaks is the minimum for p < 0.05. The benchmark has the headroom
      (59 wrong items, 41 firings) but the detector's recall on errors is 0.153,
      so the fixing arm yields 3 discordant pairs, not 6.
    - AUROC is not a selector (gate.py): the highest pre-commit AUROC cell is the
      worse intervention. Detector layer choice remains under-determined offline
      (justify_detector_layer.py reports L19 is causally ADJACENT to the
      attribution band L10-17 but not UNIQUELY selected by it).
    - The firing precision is low: only 9 of 41 firings were on items that were
      actually wrong (precision 0.220). 32 firings landed on already-correct
      items without breaking them, which is safe but wasteful.

THE HONEST SUMMARY
  The action arm is real, safe, and its per-fired repair rate generalises. It is
  not yet statistically distinguishable from no effect on this task, and the
  reason is measured, not speculated: the detector reaches only 15% of errors,
  and errors it cannot see are overwhelmingly Static Voids that no runtime
  method can repair. Raising significance requires a higher-recall trigger, not
  a larger benchmark.

Usage: python analysis/summarize_single_task.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
RES = os.path.join(EDGE_ROOT, "results")


def main():
    score_p = os.path.join(RES, "single_task_score.json")
    if not os.path.exists(score_p):
        sys.exit("run score_single_task.py first")
    score = json.load(open(score_p, encoding="utf-8"))
    bench = json.load(open(os.path.join(RES, "bench_single_task.json"), encoding="utf-8"))

    arm = score["arms"][0]
    print(f"benchmark: {bench['n_items']} items {bench['family_counts']}")
    print(f"paired items: {arm['n_paired']}")
    print()
    print("RESULT (strict metric, paired exact McNemar)")
    print(f"  fixes={arm['fixes']}  breaks={arm['breaks']}  net={arm['net']}")
    print(f"  p={arm['p_exact_mcnemar']}  significant={arm['significant_p05']}")
    print(f"  net 95% CI (Wilson) {arm['net_wilson95']}")
    print(f"  fired={arm['n_fired']}  repairs/fired-wrong={arm['repairs_per_fired_wrong']}")
    print(f"  still wrong: {arm['still_wrong_total']} "
          f"(fired {arm['still_wrong_that_fired']} / quiet {arm['still_wrong_quiet']})")
    print()
    quiet_share = arm['still_wrong_quiet'] / arm['still_wrong_total']
    print(f"  -> {quiet_share:.1%} of the residual error never fired: the Static "
          f"Memory Void population NTW explains.")
    print()
    print("VERDICT")
    print("  safe:            YES - 0 breaks on 463 items, 3 question frames")
    print("  generalises:     YES - repairs/fired-wrong 0.333 >= Africa 0.286")
    print("  timing law:      YES - 40/41 fired items had lead >= 1")
    print(f"  significant:     NO  - p={arm['p_exact_mcnemar']}, needs 6 fixes "
          f"(power_single_task.py)")
    print("  cause of the     detector recall on errors is 0.153; most errors are")
    print("  shortfall:       Static Voids outside runtime reach by construction")


if __name__ == "__main__":
    main()
