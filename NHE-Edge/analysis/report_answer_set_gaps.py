"""Measure the residual exposure of the token-matching metric.

`scoring.strict` matches whole token sequences, so a shorter accepted answer that
is its own token inside a longer emitted name ('Delhi' inside 'New Delhi') is
still counted correct. That is a deliberate, documented limitation: closing it in
the scorer requires a name-span parser (rejected: it needs an unaudited word list).

This script MEASURES the exposure instead of assuming it away. For every item in
the merged single-task benchmark it reports:

  - items whose accepted set contains a shorter answer that is a strict token
    sub-sequence of a longer accepted answer (the structural pre-condition for
    the limitation)
  - for those items, whether the longer form is actually present in the
    committed baseline generation (the only place this can change a number)

Usage: python analysis/report_answer_set_gaps.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "core"))

from scoring import fold, normalise_alternatives, strict, substring  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def embeds(container, part):
    n = len(part)
    return n > 0 and any(container[i:i + n] == part for i in range(len(container) - n + 1))


def main():
    bench_path = os.path.join(RES, "bench_single_task.json")
    if not os.path.exists(bench_path):
        sys.exit("run build_single_task.py first")
    bench = json.load(open(bench_path, encoding="utf-8"))

    structural = []
    for it in bench["items"]:
        alts = [a.split() for a in normalise_alternatives(it["answers"])]
        for short in alts:
            for long in alts:
                if len(long) > len(short) and embeds(long, short):
                    structural.append({
                        "id": it["id"], "question": it["question"],
                        "short": " ".join(short), "long": " ".join(long),
                    })

    print(f"items={len(bench['items'])}")
    print(f"structural exposure (a shorter accepted answer is a token sub-sequence")
    print(f"of a longer accepted answer): {len(structural)} items")

    # Attach committed baseline generations where they exist, to see whether the
    # model actually emits the longer form on any of these items.
    gen_index = {}
    for fn in os.listdir(RES):
        if fn.startswith("eval_") and fn.endswith("_baseline.json"):
            try:
                d = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for r in d.get("results", []):
                gen_index[r["question"]] = r.get("generated", "")

    affected = []
    for s in structural:
        gen = gen_index.get(s["question"])
        if gen is None:
            continue
        sent = fold(gen)
        long_tokens = s["long"].split()
        emits_long = any(sent.split()[i:i + len(long_tokens)] == long_tokens
                         for i in range(len(sent.split()) - len(long_tokens) + 1))
        strict_score = strict(gen, next(i["answers"] for i in bench["items"]
                                        if i["id"] == s["id"]))
        if emits_long:
            affected.append({**s, "emits_longer_form": True, "strict": strict_score})

    print(f"items where the model actually emits the longer form on the committed"
          f" baseline: {len(affected)}")
    for a in affected:
        print(f"  [{a['id']}] {a['question']}")
        print(f"     '{a['short']}' inside '{a['long']}' -> strict={a['strict']}")

    out = os.path.join(RES, "answer_set_gaps.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"n_items": len(bench["items"]),
                   "n_structural_exposure": len(structural),
                   "n_emitting_longer_form_on_baseline": len(affected),
                   "structural": structural, "affected": affected}, fh, indent=1)
    print(f"\nsaved {out}")
    print("Reading: the metric's exposure is the number of ITEMS where a shorter")
    print("accepted answer sits inside a longer accepted answer AND the model")
    print("emits the longer form. If that count is small, the token matcher is")
    print("safe for this benchmark and the residual risk is bounded and known.")


if __name__ == "__main__":
    main()
