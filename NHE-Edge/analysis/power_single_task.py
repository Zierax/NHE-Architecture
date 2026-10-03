"""Power analysis: what does the single-task benchmark actually have to show?

THE QUESTION THIS ANSWERS, BEFORE THE RUN
    The Africa greedy headline was 7/54 -> 5/54, i.e. 2 fixes and 0 breaks. On an
    exact McNemar test that is p = 0.5: the deployed result is clean but not
    statistically distinguishable from no effect. Before collecting 463 items it
    is worth stating what the new benchmark must show for the claim to become
    significant, so the experiment is not judged after the fact.

METHOD
    For a paired binary arm (none vs mask) the exact McNemar p-value depends only
    on the discordant counts b = W2C and c = C2W. Given a target net reduction,
    the required n follows from how many items the detector actually fires on
    (fires are the only items that can flip) and the historical per-fired-item
    repair rate. This script computes:

      - p for every (b, c) pair up to a plausible ceiling, so the required
        discordant count for p < 0.05 is explicit;
      - the number of FIRING items needed to reach that, at observed repair and
        break rates;
      - the same requirement for a zero-break arm, which is the claim we actually
        make for runtime.

It reports the requirement as a function of rate assumptions rather than a single
number, because the rates are exactly what the new benchmark measures and are not
yet known at higher n.

Usage: python analysis/power_single_task.py
"""
import json
import os
import sys

from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
RES = os.path.join(EDGE_ROOT, "results")


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p for discordant counts (b=fixes, c=breaks)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = binomtest(k, n, 0.5).pvalue
    return float(p)


def main():
    print("=== exact McNemar: what does significance require? ===")
    print(f"{'fixes b':>9} {'breaks c':>9} {'net':>5} {'p (exact)':>12}")
    table = []
    for c in (0, 1, 2, 3):
        for b in range(c, 13):
            p = exact_mcnemar(b, c)
            sig = "  <-- p<0.05" if p < 0.05 else ""
            print(f"{b:>9} {c:>9} {b - c:>5} {p:>12.4f}{sig}")
            table.append({"fixes": b, "breaks": c, "net": b - c, "p": round(p, 5)})

    print("\n=== minimum discordant pairs for p < 0.05 ===")
    min_disc = {}
    for c in (0, 1, 2, 3, 4, 5):
        for b in range(c, 40):
            if exact_mcnemar(b, c) < 0.05:
                min_disc[c] = {"fixes_needed": b, "discordant_needed": b + c}
                break
    for c, v in min_disc.items():
        print(f"  with {c} break(s): need {v['fixes_needed']} fixes "
              f"({v['discordant_needed']} discordant pairs)")

    print("\n=== translating to benchmark size ===")
    print("Only items the detector FIRES on can flip. On the deployed Africa setup")
    print("it fired on 7/54 items and produced 2 fixes, 0 breaks (repair rate 2/7,")
    print("break rate 0/7). If the new benchmark preserves those per-fired rates,")
    print("the number of firing items required is:")
    for c in (0, 1):
        need = min_disc[c]["fixes_needed"]
        for repair, label in ((2 / 7, "Africa-observed (2/7 repair, 0 break)"),
                              (0.30, "30% repair, 0 break"),
                              (0.50, "50% repair, 0 break")):
            fired_needed = need / repair
            print(f"  target {need} fixes with {c} break(s), {label}: "
                  f"~{fired_needed:.0f} firing items")

    fire_rate = 7 / 54
    need0 = min_disc[0]["fixes_needed"]
    print(f"\nAt the observed fire rate ({fire_rate:.3f} of items), reaching "
          f"{need0} fixes needs roughly")
    for repair in (0.2857, 0.30, 0.50):
        print(f"  {repair:.0%} per-fired repair: ~{need0 / repair / fire_rate:.0f} items")
    print("\nThe merged single-task benchmark has 463 items, so it has the headroom")
    print("for every estimate - provided the per-fired repair rate does not collapse")
    print("on a harder item mix. That collapse is itself a result worth reporting.")

    payload = {"mcnemar_table": table, "min_discordant_for_p005": min_disc,
               "observed_africa": {"n": 54, "fired": 7, "fixes": 2, "breaks": 0,
                                    "p": round(exact_mcnemar(2, 0), 5)},
               "bench_items": 463}
    out = os.path.join(RES, "power_single_task.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
